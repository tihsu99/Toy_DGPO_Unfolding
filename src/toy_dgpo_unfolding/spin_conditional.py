from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from .core import load_config, make_generator, resolve_device, seed_everything
from .flow import ConditionalFlow
from .inference import binned_fisher_per_event, fit_poisson
from .spin_matrix import (
    SPIN_PARAMETER_NAMES,
    VectorScoreModel,
    _load_flow,
    _score_nmse,
    _score_reconstruction,
    _selected_scores,
    _train_vector_score,
)
from .training import reconstruct_policy, slice_events
from .ztautau import candidate_reconstruction, generate_events


EXPECTED_TARGET_SETS = {
    "Cnn": ["C_nn"],
    "Cdiag": ["C_nn", "C_rr", "C_kk"],
    "BC5": ["C_nn", "C_rr", "C_kk", "B_A_n", "B_B_n"],
}
POLICY_KEYS = {"Cnn": "cnn_only", "Cdiag": "cdiag", "BC5": "bc5"}
POLICY_LABELS = {
    "baseline": "Baseline",
    "cnn_only": r"$C_{nn}$ only",
    "cdiag": "Cdiag",
    "bc5": "BC5",
}
POLICY_COLORS = {
    "baseline": "#767676",
    "cnn_only": "#0F4D92",
    "cdiag": "#42949E",
    "bc5": "#9A4D8E",
}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


def _save_figure(fig: plt.Figure, path: Path, settings: dict[str, Any]) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=int(settings["dpi"]), bbox_inches="tight")
    if bool(settings.get("export_svg", True)):
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _set_figure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _load_conditional_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Conditional-spin configuration root must be a mapping")
    required = {
        "base_config", "cnn_output_dir", "output_dir", "target_sets",
        "primary_parameters", "training", "validation", "evaluation",
        "seed_offsets", "plots", "allow_overwrite",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Missing conditional-spin configuration sections: {sorted(missing)}")
    base_path = Path(config["base_config"])
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_config(base_path)
    actual_targets = {name: list(values) for name, values in config["target_sets"].items()}
    if actual_targets != EXPECTED_TARGET_SETS:
        raise ValueError(f"target_sets must be exactly {EXPECTED_TARGET_SETS}")
    if list(config["primary_parameters"]) != EXPECTED_TARGET_SETS["BC5"]:
        raise ValueError("primary_parameters must be the raw physical BC5 target order")
    if int(config["training"]["group_size"]) != int(base["dgpo"]["group_size"]):
        raise ValueError("The conditional study must preserve the base DGPO group size")
    if not np.isclose(float(base["physics"]["nominal_C"]), 0.60):
        raise ValueError("The conditional study requires the unchanged nominal C_nn = 0.60")
    count_keys = {
        "training": ("dgpo_epochs_per_refresh", "max_total_dgpo_epochs", "active_score_events", "policy_training_events", "monitor_events", "score_epochs", "batch_size"),
        "validation": ("score_training_events", "evaluation_events", "score_epochs", "batch_size", "early_stop_patience_attempts"),
        "evaluation": ("score_training_events", "events", "score_epochs", "batch_size", "pseudo_template_events", "pseudo_template_chunk_size", "pseudo_experiments", "events_per_pseudo_experiment"),
    }
    for section, keys in count_keys.items():
        if any(int(config[section][key]) < 1 for key in keys):
            raise ValueError(f"All configured counts in {section} must be positive")
    if float(config["validation"]["min_relative_J_improvement"]) < 0.0:
        raise ValueError("min_relative_J_improvement must be non-negative")
    if float(config["validation"]["split_noise_multiplier"]) < 0.0:
        raise ValueError("split_noise_multiplier must be non-negative")
    if not 0.0 < float(config["validation"]["learning_rate_backoff"]) <= 1.0:
        raise ValueError("learning_rate_backoff must lie in (0, 1]")
    return config, base


def conditional_objective(fisher_diagonal: np.ndarray, baseline_diagonal: np.ndarray) -> float:
    fisher = np.asarray(fisher_diagonal, dtype=np.float64)
    baseline = np.asarray(baseline_diagonal, dtype=np.float64)
    if fisher.shape != baseline.shape or fisher.ndim != 1:
        raise ValueError("Conditional Fisher diagonals must be one-dimensional and shape matched")
    if np.any(fisher <= 0.0) or np.any(baseline <= 0.0):
        raise ValueError("Conditional Fisher diagonals must be strictly positive")
    return float(np.mean(baseline / fisher))


def _torch_conditional_objective(
    fisher_diagonal: torch.Tensor, baseline_diagonal: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(fisher_diagonal.dtype).tiny
    return (baseline_diagonal / fisher_diagonal.clamp_min(floor)).mean(dim=-1)


def replacement_conditional_diagonal(
    total_reference: torch.Tensor,
    event_reference_score: torch.Tensor,
    candidate_score: torch.Tensor,
) -> torch.Tensor:
    return (
        total_reference[None, None, :]
        - event_reference_score[:, None, :].square()
        + candidate_score.square()
    )


def information_decomposition(scores: torch.Tensor, valid: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = scores.detach().to(torch.float64)
    mask = valid.detach().bool()
    efficiency = float(mask.float().mean())
    if not bool(mask.any()):
        raise RuntimeError("Cannot decompose Fisher information without valid events")
    per_valid = score[mask].square().mean(dim=0).cpu().numpy()
    fisher = score.square().mean(dim=0).cpu().numpy()
    return np.full(score.shape[-1], efficiency), per_valid, fisher


def acceptance_decision(
    candidate_J: float, accepted_J: float, relative_tolerance: float,
) -> tuple[bool, str]:
    threshold = accepted_J * relative_tolerance
    if candidate_J < accepted_J - threshold:
        return True, "accepted"
    if candidate_J > accepted_J + threshold:
        return False, "rejected_worse"
    return False, "rejected_unresolved"


def _save_policy(
    policy: ConditionalFlow, path: Path, target: str, attempt: int,
    status: str, validation_J: float,
) -> None:
    torch.save({
        "method_version": 4,
        "study": "conditional_spin_measurement",
        "target_set": target,
        "attempt": attempt,
        "status": status,
        "selection_metric": "independent_validation_J_cond",
        "validation_J_cond": validation_J,
        "state_dict": policy.state_dict(),
    }, path)


def _load_conditional_policy(path: Path, base: dict[str, Any], device: torch.device) -> ConditionalFlow:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("method_version") != 4 or payload.get("study") != "conditional_spin_measurement":
        raise RuntimeError(f"Checkpoint is not from the conditional-spin study: {path}")
    from .training import make_flow
    policy = make_flow(base, device)
    policy.load_state_dict(payload["state_dict"])
    return policy.eval()


def _load_baseline(config: dict[str, Any], base: dict[str, Any], device: torch.device) -> ConditionalFlow:
    path = Path(config["cnn_output_dir"]).expanduser().resolve() / "checkpoints" / "baseline" / "best_validation_policy.pt"
    return _load_flow(path, base, device)


def _generate_fixed_samples(
    config: dict[str, Any], base: dict[str, Any], device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    seed = int(base["seed"])
    offsets = config["seed_offsets"]
    nominal = float(base["physics"]["nominal_C"])
    specifications = {
        "monitor": (int(config["training"]["monitor_events"]), int(offsets["monitor"])),
        "validation_score": (int(config["validation"]["score_training_events"]), int(offsets["validation_score_training"])),
        "validation": (int(config["validation"]["evaluation_events"]), int(offsets["validation_evaluation"])),
        "evaluation_score": (int(config["evaluation"]["score_training_events"]), int(offsets["evaluation_score_training"])),
        "evaluation": (int(config["evaluation"]["events"]), int(offsets["evaluation_events"])),
    }
    return {
        name: generate_events(count, nominal, base, device, make_generator(device, seed + offset))
        for name, (count, offset) in specifications.items()
    }


def _fit_diagnostic_score(
    policy: ConditionalFlow,
    events: dict[str, torch.Tensor],
    parameters: list[str],
    settings: dict[str, Any],
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    description: str,
) -> tuple[VectorScoreModel, list[float]]:
    return _train_vector_score(
        policy, events, parameters, config, base, device, seed, description, settings,
    )


@torch.no_grad()
def _policy_diagnostics(
    policy: ConditionalFlow,
    baseline: ConditionalFlow,
    score_model: VectorScoreModel,
    events: dict[str, torch.Tensor],
    parameters: list[str],
    base: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    nmse, reconstructed, prediction, target = _score_nmse(
        policy, score_model, events, parameters, base, seed,
    )
    raw_mse = float((prediction - target).square().mean())
    target_scale = float(target.square().mean())
    action = reconstructed["action"]
    kl = float((
        policy.log_prob(action, events["context"])
        - baseline.log_prob(action, events["context"])
    ).mean())
    cosine = (reconstructed["k_a"] * events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        "NMSE": nmse,
        "relative_NMSE": raw_mse / max(target_scale, np.finfo(float).eps),
        "valid_efficiency": float(reconstructed["valid"].float().mean()),
        "tau_axis_error": float(torch.acos(cosine).mean()),
        "KL_to_initial": kl,
    }


@torch.no_grad()
def _measure_policy(
    policy: ConditionalFlow,
    score_model: VectorScoreModel,
    active_score: VectorScoreModel | None,
    events: dict[str, torch.Tensor],
    parameters: list[str],
    base: dict[str, Any],
    seed: int,
    direct_Cnn_bins: int,
) -> dict[str, Any]:
    reconstructed = reconstruct_policy(policy, events, base, make_generator(events["x"].device, seed))
    scores = _score_reconstruction(score_model, reconstructed)
    fisher = scores.square().mean(dim=0).cpu().numpy()
    midpoint = events["x"].numel() // 2
    half_fisher = [
        scores[:midpoint].square().mean(dim=0).cpu().numpy(),
        scores[midpoint:].square().mean(dim=0).cpu().numpy(),
    ]
    active_fisher = None
    if active_score is not None:
        active_fisher = _score_reconstruction(active_score, reconstructed).square().mean(dim=0).cpu().numpy()
    direct_Cnn = None
    if "C_nn" in parameters:
        edges = np.linspace(-1.0, 1.0, direct_Cnn_bins + 1)
        direct_Cnn = binned_fisher_per_event(
            events["x"].cpu().numpy(), reconstructed["y"].cpu().numpy(),
            reconstructed["valid"].cpu().numpy().astype(bool), edges,
            float(base["physics"]["nominal_C"]),
        )
    cosine = (reconstructed["k_a"] * events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
    efficiency, per_valid, decomposed = information_decomposition(scores, reconstructed["valid"])
    return {
        "fisher_diagonal": fisher,
        "half_fisher_diagonal": half_fisher,
        "active_fisher_diagonal": active_fisher,
        "direct_Cnn_fisher": direct_Cnn,
        "valid_efficiency": float(reconstructed["valid"].float().mean()),
        "tau_axis_error": float(torch.acos(cosine).mean()),
        "decomposition_efficiency": efficiency,
        "information_per_valid": per_valid,
        "decomposed_fisher": decomposed,
        "reconstructed": reconstructed,
        "scores": scores,
    }


def _evaluate_trial(
    policy: ConditionalFlow,
    baseline: ConditionalFlow,
    active_score: VectorScoreModel,
    parameters: list[str],
    baseline_fisher: np.ndarray,
    samples: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    description: str,
) -> dict[str, Any]:
    settings = config["validation"]
    diagnostic, losses = _fit_diagnostic_score(
        policy, samples["validation_score"], parameters, settings, config, base,
        device, seed, description,
    )
    measured = _measure_policy(
        policy, diagnostic, active_score, samples["validation"], parameters, base,
        int(base["seed"]) + int(config["seed_offsets"]["validation_reconstruction"]),
        int(settings["direct_Cnn_bins"]),
    )
    fisher = measured["fisher_diagonal"]
    half_J = [conditional_objective(item, baseline_fisher) for item in measured["half_fisher_diagonal"]]
    validation_J = conditional_objective(fisher, baseline_fisher)
    split_relative_noise = abs(half_J[0] - half_J[1]) / max(2.0 * validation_J, np.finfo(float).eps)
    relative_tolerance = max(
        float(settings["min_relative_J_improvement"]),
        float(settings["split_noise_multiplier"]) * split_relative_noise,
    )
    active_fisher = measured["active_fisher_diagonal"]
    closure = active_fisher / np.clip(fisher, np.finfo(float).tiny, None)
    monitor = _policy_diagnostics(
        policy, baseline, active_score, samples["monitor"], parameters, base,
        int(base["seed"]) + int(config["seed_offsets"]["monitor_reconstruction"]),
    )
    return {
        "active_J": conditional_objective(active_fisher, baseline_fisher),
        "validation_J": validation_J,
        "half_J": half_J,
        "split_relative_noise": split_relative_noise,
        "relative_tolerance": relative_tolerance,
        "validation_fisher_diagonal": fisher.tolist(),
        "active_fisher_diagonal": active_fisher.tolist(),
        "conditional_sigma_ratio_to_baseline": np.sqrt(baseline_fisher / fisher).tolist(),
        "closure": closure.tolist(),
        "mean_abs_log_closure": float(np.mean(np.abs(np.log(np.clip(closure, 1.0e-300, None))))),
        "max_abs_log_closure": float(np.max(np.abs(np.log(np.clip(closure, 1.0e-300, None))))),
        "direct_Cnn_fisher": measured["direct_Cnn_fisher"],
        "score_to_direct_Cnn_closure": (
            float(fisher[parameters.index("C_nn")] / measured["direct_Cnn_fisher"])
            if measured["direct_Cnn_fisher"] and "C_nn" in parameters else None
        ),
        "validation_score_loss": losses[-1],
        **monitor,
    }


def _train_local_conditional_update(
    current: ConditionalFlow,
    active_score: VectorScoreModel,
    events: dict[str, torch.Tensor],
    baseline_fisher_per_event: np.ndarray,
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    description: str,
) -> tuple[ConditionalFlow, list[dict[str, float]], float]:
    settings = config["training"]
    reconstruction_seed = int(base["seed"]) + int(config["seed_offsets"]["policy_reconstruction"]) + seed
    reference = reconstruct_policy(current, events, base, make_generator(device, reconstruction_seed))
    reference_score = _score_reconstruction(active_score, reference)
    total_reference = reference_score.square().sum(dim=0).to(torch.float64)
    baseline_total = torch.as_tensor(
        baseline_fisher_per_event * events["x"].numel(), dtype=torch.float64, device=device,
    )
    reference_J = float(_torch_conditional_objective(total_reference, baseline_total))
    policy = copy.deepcopy(current).to(device).train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, weight_decay=float(settings["weight_decay"]),
    )
    generator = make_generator(
        device, int(base["seed"]) + int(config["seed_offsets"]["policy_sampling"]) + seed,
    )
    batch_size = int(settings["batch_size"])
    group_size = int(settings["group_size"])
    rows: list[dict[str, float]] = []
    progress = tqdm(range(epochs), desc=description, unit="epoch")
    for local_epoch in progress:
        permutation = torch.randperm(events["x"].numel(), device=device, generator=generator)
        reward_sum = 0.0
        invalid_sum = 0.0
        seen = 0
        for start in range(0, permutation.numel(), batch_size):
            indices = permutation[start : start + batch_size]
            batch = slice_events(events, indices)
            actions, _ = policy.sample(batch["context"], group_size, generator)
            with torch.no_grad():
                reconstruction = candidate_reconstruction(batch, actions.detach(), base)
                candidate_score = _score_reconstruction(active_score, reconstruction)
                candidate_diagonal = replacement_conditional_diagonal(
                    total_reference, reference_score[indices], candidate_score,
                )
                candidate_J = _torch_conditional_objective(candidate_diagonal, baseline_total)
                reward = 0.5 * torch.log(
                    torch.as_tensor(reference_J, dtype=torch.float64, device=device) / candidate_J
                ) * float(settings["reward_scale"])
                advantage = reward - (reward.sum(dim=1, keepdim=True) - reward) / (group_size - 1)
            context = batch["context"][:, None, :].expand(-1, group_size, -1)
            log_probability = policy.log_prob(actions.detach(), context)
            loss = -(advantage.detach() * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            count = indices.numel()
            reward_sum += float(reward.mean()) * count
            invalid_sum += float((~reconstruction["valid"]).float().mean()) * count
            seen += count
        row = {
            "local_epoch": float(local_epoch + 1),
            "mean_reward": reward_sum / seen,
            "invalid_fraction": invalid_sum / seen,
        }
        rows.append(row)
        progress.set_postfix(reward=f"{row['mean_reward']:.4g}", invalid=f"{row['invalid_fraction']:.3f}")
    return policy.eval(), rows, reference_J


def _active_score_for_update(
    policy: ConditionalFlow,
    parameters: list[str],
    accepted_updates: int,
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    target_index: int,
) -> tuple[VectorScoreModel, list[float]]:
    seed = (
        int(base["seed"]) + int(config["seed_offsets"]["active_score"])
        + 100000 * target_index + 1000 * accepted_updates
    )
    events = generate_events(
        int(config["training"]["active_score_events"]),
        float(base["physics"]["nominal_C"]), base, device, make_generator(device, seed),
    )
    return _fit_diagnostic_score(
        policy, events, parameters, config["training"], config, base, device, seed,
        f"{parameters} active score update {accepted_updates}",
    )


def _train_target_policy(
    baseline: ConditionalFlow,
    target: str,
    parameters: list[str],
    target_index: int,
    samples: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> tuple[ConditionalFlow, list[dict[str, Any]]]:
    checkpoint_dir = output / "checkpoints" / POLICY_KEYS[target]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    active_score, active_losses = _active_score_for_update(
        baseline, parameters, 0, config, base, device, target_index,
    )
    validation_seed = (
        int(base["seed"]) + int(config["seed_offsets"]["validation_score_reconstruction"])
        + 100000 * target_index
    )
    baseline_validation_model, baseline_losses = _fit_diagnostic_score(
        baseline, samples["validation_score"], parameters, config["validation"],
        config, base, device, validation_seed, f"{target} baseline validation score",
    )
    baseline_measurement = _measure_policy(
        baseline, baseline_validation_model, active_score, samples["validation"],
        parameters, base,
        int(base["seed"]) + int(config["seed_offsets"]["validation_reconstruction"]),
        int(config["validation"]["direct_Cnn_bins"]),
    )
    baseline_fisher = baseline_measurement["fisher_diagonal"]
    baseline_active = baseline_measurement["active_fisher_diagonal"]
    baseline_closure = baseline_active / np.clip(baseline_fisher, np.finfo(float).tiny, None)
    baseline_monitor = _policy_diagnostics(
        baseline, baseline, active_score, samples["monitor"], parameters, base,
        int(base["seed"]) + int(config["seed_offsets"]["monitor_reconstruction"]),
    )
    history: list[dict[str, Any]] = [{
        "target_set": target,
        "parameters": parameters,
        "attempt": 0,
        "total_dgpo_epochs": 0,
        "accepted_update": 0,
        "status": "baseline",
        "active_J": conditional_objective(baseline_active, baseline_fisher),
        "validation_J": 1.0,
        "accepted_reference_J": 1.0,
        "relative_tolerance": None,
        "split_relative_noise": None,
        "validation_fisher_diagonal": baseline_fisher.tolist(),
        "active_fisher_diagonal": baseline_active.tolist(),
        "conditional_sigma_ratio_to_baseline": np.ones(len(parameters)).tolist(),
        "closure": baseline_closure.tolist(),
        "mean_abs_log_closure": float(np.mean(np.abs(np.log(np.clip(baseline_closure, 1.0e-300, None))))),
        "max_abs_log_closure": float(np.max(np.abs(np.log(np.clip(baseline_closure, 1.0e-300, None))))),
        "direct_Cnn_fisher": baseline_measurement["direct_Cnn_fisher"],
        "score_to_direct_Cnn_closure": (
            float(baseline_fisher[parameters.index("C_nn")] / baseline_measurement["direct_Cnn_fisher"])
            if baseline_measurement["direct_Cnn_fisher"] else None
        ),
        "validation_score_loss": baseline_losses[-1],
        "active_score_loss": active_losses[-1],
        "learning_rate": None,
        "local_epochs": 0,
        "mean_reward": None,
        "score_refreshes": 0,
        **baseline_monitor,
    }]
    with (output / f"baseline_conditional_fisher_{target}.json").open("w", encoding="utf-8") as stream:
        json.dump({"parameters": parameters, "F0_diagonal": baseline_fisher.tolist()}, stream, indent=2)
    current = copy.deepcopy(baseline).eval()
    best = copy.deepcopy(baseline).eval()
    accepted_J = 1.0
    accepted_updates = 0
    rejected_updates = 0
    consecutive_nonimproving = 0
    total_epochs = 0
    attempt = 0
    score_refreshes = 0
    max_epochs = int(config["training"]["max_total_dgpo_epochs"])
    fixed_epochs = int(config["training"]["dgpo_epochs_per_refresh"])
    patience = int(config["validation"]["early_stop_patience_attempts"])
    while total_epochs < max_epochs and consecutive_nonimproving < patience:
        accepted_this_round = False
        base_learning_rate = float(config["training"]["learning_rate"])
        retries = int(config["validation"]["max_retries_after_rejection"])
        for retry in range(retries + 1):
            if total_epochs >= max_epochs or consecutive_nonimproving >= patience:
                break
            attempt += 1
            local_epochs = fixed_epochs if retry == 0 else int(config["validation"]["retry_epochs"])
            local_epochs = min(local_epochs, max_epochs - total_epochs)
            learning_rate = base_learning_rate * float(config["validation"]["learning_rate_backoff"]) ** retry
            training_seed = (
                int(config["seed_offsets"]["policy_training"]) + 100000 * target_index + 1000 * attempt
            )
            training_events = generate_events(
                int(config["training"]["policy_training_events"]),
                float(base["physics"]["nominal_C"]), base, device,
                make_generator(device, int(base["seed"]) + training_seed),
            )
            trial, local_rows, active_reference_J = _train_local_conditional_update(
                current, active_score, training_events, baseline_fisher, config, base,
                device, training_seed, local_epochs, learning_rate,
                f"{target} attempt {attempt}" + (" backoff" if retry else ""),
            )
            total_epochs += local_epochs
            validation = _evaluate_trial(
                trial, baseline, active_score, parameters, baseline_fisher, samples,
                config, base, device,
                validation_seed,
                f"{target} validation score attempt {attempt}",
            )
            accepted, status = acceptance_decision(
                validation["validation_J"], accepted_J, validation["relative_tolerance"],
            )
            if accepted:
                current = trial
                best = copy.deepcopy(trial).eval()
                accepted_J = float(validation["validation_J"])
                accepted_updates += 1
                consecutive_nonimproving = 0
                accepted_this_round = True
            else:
                rejected_updates += 1
                consecutive_nonimproving += 1
            row = {
                "target_set": target,
                "parameters": parameters,
                "attempt": attempt,
                "total_dgpo_epochs": total_epochs,
                "accepted_update": accepted_updates,
                "status": status,
                "retry": retry,
                "active_reference_J": active_reference_J,
                "accepted_reference_J": accepted_J if accepted else history[-1]["accepted_reference_J"],
                "learning_rate": learning_rate,
                "local_epochs": local_epochs,
                "mean_reward": float(np.mean([item["mean_reward"] for item in local_rows])),
                "score_refreshes": score_refreshes,
                **validation,
            }
            history.append(row)
            checkpoint_name = f"attempt_{attempt:03d}_{'accepted' if accepted else 'rejected'}.pt"
            _save_policy(trial, checkpoint_dir / checkpoint_name, target, attempt, status, validation["validation_J"])
            if accepted:
                _save_policy(best, checkpoint_dir / "best_validation_policy.pt", target, attempt, status, accepted_J)
                _save_policy(best, checkpoint_dir / f"accepted_{accepted_updates:03d}.pt", target, attempt, status, accepted_J)
                if total_epochs < max_epochs:
                    active_score, active_losses = _active_score_for_update(
                        current, parameters, accepted_updates, config, base, device, target_index,
                    )
                    score_refreshes += 1
                break
        if not accepted_this_round and (total_epochs >= max_epochs or consecutive_nonimproving >= patience):
            break
    _save_policy(best, checkpoint_dir / "best_validation_policy.pt", target, attempt, "best_accepted", accepted_J)
    _save_policy(current, checkpoint_dir / "final_policy.pt", target, attempt, "final_accepted", accepted_J)
    for row in history:
        row["final_accepted_updates"] = accepted_updates
        row["final_rejected_updates"] = rejected_updates
        row["final_score_refreshes"] = score_refreshes
    payload = {
        "target_set": target,
        "parameters": parameters,
        "selection_metric": "minimum independently validated conditional J_cond",
        "best_validation_J": accepted_J,
        "accepted_updates": accepted_updates,
        "rejected_updates": rejected_updates,
        "score_refreshes": score_refreshes,
        "history": history,
    }
    with (output / f"conditional_training_{target}.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    _write_rows(output / f"conditional_training_{target}_attempts", history)
    return best, history


def _load_training_histories(config: dict[str, Any], output: Path) -> dict[str, list[dict[str, Any]]]:
    histories = {}
    for target in config["target_sets"]:
        path = output / f"conditional_training_{target}.json"
        with path.open(encoding="utf-8") as stream:
            histories[target] = json.load(stream)["history"]
    return histories


def _load_study_policies(
    baseline: ConditionalFlow,
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, ConditionalFlow]:
    policies = {"baseline": baseline}
    for target, key in POLICY_KEYS.items():
        policies[key] = _load_conditional_policy(
            output / "checkpoints" / key / "best_validation_policy.pt", base, device,
        )
    return policies


def _evaluate_full15(
    policies: dict[str, ConditionalFlow],
    samples: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> dict[str, dict[str, Any]]:
    parameters = list(SPIN_PARAMETER_NAMES)
    results: dict[str, dict[str, Any]] = {}
    for name, policy in policies.items():
        score_seed = int(base["seed"]) + int(config["seed_offsets"]["evaluation_score_reconstruction"])
        model, losses = _fit_diagnostic_score(
            policy, samples["evaluation_score"], parameters, config["evaluation"],
            config, base, device, score_seed, f"{name} Full-15 diagnostic score",
        )
        torch.save({
            "method_version": 1,
            "study": "conditional_spin_full15_diagnostic",
            "policy": name,
            "parameters": parameters,
            "state_dict": model.state_dict(),
        }, output / "checkpoints" / f"full15_score_{name}.pt")
        measured = _measure_policy(
            policy, model, None, samples["evaluation"], parameters, base,
            int(base["seed"]) + int(config["seed_offsets"]["evaluation_reconstruction"]),
            int(config["evaluation"]["direct_Cnn_bins"]),
        )
        results[name] = {
            "policy": name,
            "parameters": parameters,
            "individual_fisher_per_generated_event": measured["fisher_diagonal"].tolist(),
            "conditional_sigma_per_sqrt_event": (1.0 / np.sqrt(measured["fisher_diagonal"])).tolist(),
            "valid_efficiency": measured["valid_efficiency"],
            "tau_axis_error": measured["tau_axis_error"],
            "direct_Cnn_fisher_per_generated_event": measured["direct_Cnn_fisher"],
            "score_to_direct_Cnn_closure": float(
                measured["fisher_diagonal"][parameters.index("C_nn")] / measured["direct_Cnn_fisher"]
            ),
            "information_per_valid": measured["information_per_valid"].tolist(),
            "decomposition_efficiency": measured["decomposition_efficiency"].tolist(),
            "score_loss": losses[-1],
        }
    baseline_fisher = np.asarray(results["baseline"]["individual_fisher_per_generated_event"])
    for result in results.values():
        fisher = np.asarray(result["individual_fisher_per_generated_event"])
        result["conditional_sigma_ratio_to_baseline"] = np.sqrt(baseline_fisher / fisher).tolist()
        result["conditional_precision_gain"] = (np.sqrt(fisher / baseline_fisher) - 1.0).tolist()
    with (output / "full15_conditional_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def _plot_conditional_multitarget(
    results: dict[str, dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    parameters = list(config["primary_parameters"])
    policy_order = ["cnn_only", "cdiag", "bc5"]
    positions = np.arange(len(parameters))
    width = 0.18
    rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    plotted: list[np.ndarray] = []
    for policy_index, name in enumerate(policy_order):
        full_ratio = np.asarray(results[name]["conditional_sigma_ratio_to_baseline"])
        ratio = np.asarray([full_ratio[list(SPIN_PARAMETER_NAMES).index(parameter)] for parameter in parameters])
        plotted.append(ratio)
        axis.plot(
            positions + (policy_index - 1) * width, ratio,
            color=POLICY_COLORS[name], marker="o", linewidth=1.0,
            label=POLICY_LABELS[name],
        )
        rows.extend({
            "policy": name,
            "parameter": parameter,
            "conditional_sigma_ratio_to_baseline": float(ratio[index]),
            "precision_gain": float(1.0 / ratio[index] - 1.0),
        } for index, parameter in enumerate(parameters))
    axis.axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=0.9)
    axis.set(
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in parameters],
        ylabel=r"Conditional $\sigma_a$ / baseline",
        title="Conditional precision across the five physical targets",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    extent = max(float(np.max(np.abs(np.concatenate(plotted) - 1.0))), 0.02)
    axis.set_ylim(max(0.0, 1.0 - 1.35 * extent), 1.0 + 1.35 * extent)
    _save_figure(fig, output / "conditional_multitarget_summary", config["plots"])
    _write_rows(output / "conditional_multitarget_summary", rows)


def _plot_full15_transfer(
    results: dict[str, dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    policy_order = ["cnn_only", "cdiag", "bc5"]
    positions = np.arange(len(SPIN_PARAMETER_NAMES))
    width = 0.18
    rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(10.8, 4.1), constrained_layout=True)
    plotted: list[np.ndarray] = []
    for policy_index, name in enumerate(policy_order):
        ratio = np.asarray(results[name]["conditional_sigma_ratio_to_baseline"])
        plotted.append(ratio)
        trained = set(config["target_sets"][[key for key, value in POLICY_KEYS.items() if value == name][0]])
        shifted = positions + (policy_index - 1) * width
        axis.plot(
            shifted, ratio, color=POLICY_COLORS[name], linewidth=0.8,
            label=POLICY_LABELS[name], zorder=2,
        )
        for parameter_index, parameter in enumerate(SPIN_PARAMETER_NAMES):
            is_trained = parameter in trained
            axis.scatter(
                shifted[parameter_index], ratio[parameter_index], s=28,
                facecolor=POLICY_COLORS[name] if is_trained else "white",
                edgecolor=POLICY_COLORS[name], linewidth=1.0, zorder=3,
            )
        rows.extend({
            "policy": name,
            "parameter": parameter,
            "trained_target": parameter in trained,
            "conditional_sigma_ratio_to_baseline": float(ratio[index]),
            "precision_gain": float(1.0 / ratio[index] - 1.0),
        } for index, parameter in enumerate(SPIN_PARAMETER_NAMES))
    axis.axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=0.9)
    axis.set(
        xticks=positions,
        xticklabels=[name.replace("_", " ") for name in SPIN_PARAMETER_NAMES],
        ylabel=r"Conditional $\sigma_a$ / baseline",
        title="Full-15 passive transfer: conditional precision (other coefficients fixed; not jointly profiled)",
    )
    axis.tick_params(axis="x", rotation=45, labelsize=7)
    axis.legend(ncol=3, loc="upper center")
    axis.text(
        0.995, 0.02, "Filled markers: explicit training targets", transform=axis.transAxes,
        ha="right", va="bottom", color="#4D4D4D", fontsize=7,
    )
    extent = max(float(np.max(np.abs(np.concatenate(plotted) - 1.0))), 0.03)
    axis.set_ylim(max(0.0, 1.0 - 1.3 * extent), 1.0 + 1.3 * extent)
    _save_figure(fig, output / "full15_conditional_passive_transfer", config["plots"])
    _write_rows(output / "full15_conditional_passive_transfer", rows)


def _plot_accept_reject(
    histories: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True, sharey=False)
    rows: list[dict[str, Any]] = []
    for axis, (target, history) in zip(axes, histories.items()):
        attempts = np.asarray([int(row["attempt"]) for row in history])
        active = np.asarray([float(row["active_J"]) for row in history])
        validation = np.asarray([float(row["validation_J"]) for row in history])
        accepted_reference = np.asarray([float(row["accepted_reference_J"]) for row in history])
        axis.plot(attempts, active, color="#A8A8A8", marker="o", label="Active surrogate")
        axis.plot(attempts, validation, color=POLICY_COLORS[POLICY_KEYS[target]], marker="o", label="Independent validation")
        axis.step(attempts, accepted_reference, where="post", color="#272727", linestyle="--", label="Accepted reference")
        for row in history[1:]:
            accepted = row["status"] == "accepted"
            axis.scatter(
                row["attempt"], row["validation_J"], s=45,
                marker="o" if accepted else "x",
                color="#2E9E44" if accepted else "#B64342", zorder=5,
            )
            if not accepted:
                axis.scatter(
                    row["attempt"], row["accepted_reference_J"], s=55,
                    marker="v", facecolor="none", edgecolor="#B64342", zorder=5,
                )
        best_row = min(
            (row for row in history if row["status"] in {"baseline", "accepted"}),
            key=lambda row: float(row["validation_J"]),
        )
        axis.scatter(best_row["attempt"], best_row["validation_J"], marker="*", s=100,
                     color="#FFD700", edgecolor="#272727", zorder=6)
        axis.axhline(1.0, color="#767676", linewidth=0.8, linestyle=":")
        axis.set(title=target, xlabel="DGPO attempt")
        rows.extend({
            "target_set": target,
            "attempt": row["attempt"],
            "active_J": row["active_J"],
            "validation_J": row["validation_J"],
            "accepted_reference_J": row["accepted_reference_J"],
            "status": row["status"],
            "relative_tolerance": row["relative_tolerance"],
        } for row in history)
    axes[0].set_ylabel(r"$J_{cond}$")
    axes[0].legend(fontsize=7)
    fig.suptitle("Independent conditional-Fisher validation controls acceptance and rollback")
    _save_figure(fig, output / "measurement_accept_reject_training", config["plots"])
    _write_rows(output / "measurement_accept_reject_training", rows)


def _plot_score_closure(
    histories: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True, sharey=False)
    rows: list[dict[str, Any]] = []
    colors = plt.get_cmap("tab10").colors
    for axis, (target, history) in zip(axes, histories.items()):
        selected = [row for row in history if row["status"] in {"baseline", "accepted"}]
        parameters = list(config["target_sets"][target])
        for parameter_index, parameter in enumerate(parameters):
            values = [row["closure"][parameter_index] for row in selected]
            updates = [row["accepted_update"] for row in selected]
            axis.plot(updates, values, marker="o", color=colors[parameter_index], label=parameter.replace("_", " "))
            rows.extend({
                "target_set": target,
                "accepted_update": row["accepted_update"],
                "parameter": parameter,
                "active_to_validation_fisher": row["closure"][parameter_index],
            } for row in selected)
        axis.axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=0.8)
        axis.set(title=target, xlabel="Accepted update")
        axis.legend(fontsize=6)
    axes[0].set_ylabel(r"$F_{active,aa}/F_{val,aa}$")
    fig.suptitle("Per-target conditional-score closure at accepted checkpoints")
    _save_figure(fig, output / "conditional_score_closure", config["plots"])
    _write_rows(output / "conditional_score_closure", rows)


def _plot_information_decomposition(
    results: dict[str, dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    parameters = list(config["primary_parameters"])
    policies = ["baseline", "cnn_only", "cdiag", "bc5"]
    baseline_efficiency = float(results["baseline"]["valid_efficiency"])
    baseline_per_valid = np.asarray(results["baseline"]["information_per_valid"])
    rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    positions = np.arange(len(parameters))
    width = 0.18
    for policy_index, name in enumerate(policies):
        efficiency_ratio = float(results[name]["valid_efficiency"]) / baseline_efficiency
        info = np.asarray(results[name]["information_per_valid"])
        info_ratio = np.asarray([
            info[list(SPIN_PARAMETER_NAMES).index(parameter)]
            / baseline_per_valid[list(SPIN_PARAMETER_NAMES).index(parameter)]
            for parameter in parameters
        ])
        offset = (policy_index - 1.5) * width
        axes[0].bar(positions + offset, np.full(len(parameters), efficiency_ratio), width,
                    color=POLICY_COLORS[name], label=POLICY_LABELS[name])
        axes[1].bar(positions + offset, info_ratio, width, color=POLICY_COLORS[name])
        rows.extend({
            "policy": name,
            "parameter": parameter,
            "valid_efficiency": float(results[name]["valid_efficiency"]),
            "valid_efficiency_ratio_to_baseline": efficiency_ratio,
            "information_per_valid": float(info[list(SPIN_PARAMETER_NAMES).index(parameter)]),
            "information_per_valid_ratio_to_baseline": float(info_ratio[index]),
            "fisher_ratio_to_baseline": float(efficiency_ratio * info_ratio[index]),
        } for index, parameter in enumerate(parameters))
    for axis, title, ylabel in zip(
        axes,
        ("Valid-efficiency contribution", "Information per valid event"),
        (r"$\epsilon_{valid}/\epsilon_{valid}^{base}$", r"$E[s_a^2|valid]/E[s_a^2|valid]_{base}$"),
    ):
        axis.axhline(1.0, color="#4D4D4D", linestyle="--", linewidth=0.8)
        axis.set(xticks=positions, xticklabels=[name.replace("_", " ") for name in parameters],
                 title=title, ylabel=ylabel)
        axis.tick_params(axis="x", rotation=25)
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(r"Conditional information decomposition: $F_{aa}/N=\epsilon_{valid}E[s_a^2|valid]$")
    _save_figure(fig, output / "target_information_decomposition", config["plots"])
    _write_rows(output / "target_information_decomposition", rows)


def _summary_rows(
    results: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    parameters = list(config["primary_parameters"])
    rows: list[dict[str, Any]] = []
    reverse_keys = {value: key for key, value in POLICY_KEYS.items()}
    for policy in ("baseline", "cnn_only", "cdiag", "bc5"):
        result = results[policy]
        fisher = np.asarray(result["individual_fisher_per_generated_event"])
        baseline_fisher = np.asarray(results["baseline"]["individual_fisher_per_generated_event"])
        gains = {
            parameter: float(np.sqrt(
                fisher[list(SPIN_PARAMETER_NAMES).index(parameter)]
                / baseline_fisher[list(SPIN_PARAMETER_NAMES).index(parameter)]
            ) - 1.0)
            for parameter in parameters
        }
        primary_fisher = np.asarray([fisher[list(SPIN_PARAMETER_NAMES).index(parameter)] for parameter in parameters])
        primary_baseline = np.asarray([baseline_fisher[list(SPIN_PARAMETER_NAMES).index(parameter)] for parameter in parameters])
        if policy == "baseline":
            final_history = histories["Cnn"][0]
            accepted = rejected = refreshes = 0
        else:
            history = histories[reverse_keys[policy]]
            accepted_rows = [row for row in history if row["status"] == "accepted"]
            final_history = accepted_rows[-1] if accepted_rows else history[0]
            accepted = sum(row["status"] == "accepted" for row in history)
            rejected = sum(str(row["status"]).startswith("rejected") for row in history)
            refreshes = int(history[-1].get("final_score_refreshes", 0))
        rows.append({
            "policy": policy,
            "C_nn conditional precision gain": gains["C_nn"],
            "C_rr conditional precision gain": gains["C_rr"],
            "C_kk conditional precision gain": gains["C_kk"],
            "B_A_n conditional precision gain": gains["B_A_n"],
            "B_B_n conditional precision gain": gains["B_B_n"],
            "Mean conditional precision gain": float(np.mean(list(gains.values()))),
            "J_cond": conditional_objective(primary_fisher, primary_baseline),
            "Valid efficiency": float(result["valid_efficiency"]),
            "Tau-axis error": float(result["tau_axis_error"]),
            "Number of accepted updates": accepted,
            "Number of rejected updates": rejected,
            "Number of score refreshes": refreshes,
            "Active/validation closure": (
                f"{float(final_history['mean_abs_log_closure']):.3f}/"
                f"{float(final_history['max_abs_log_closure']):.3f}"
            ),
        })
    return rows


def _plot_summary_table(
    rows: list[dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    short_columns = [
        r"$C_{nn}$ gain", r"$C_{rr}$ gain", r"$C_{kk}$ gain",
        r"$B_{A,n}$ gain", r"$B_{B,n}$ gain", "Mean gain", r"$J_{cond}$",
        r"$\epsilon_{valid}$", "Axis error", "Accepted", "Rejected", "Refreshes",
        r"Closure $\langle|\log c|\rangle/\max$",
    ]
    keys = list(rows[0])[1:]
    cells = []
    for row in rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:+.3f}" if "gain" in key.lower() else f"{value:.3f}")
            else:
                values.append(str(value))
        cells.append(values)
    fig, axis = plt.subplots(figsize=(15.2, 2.7), constrained_layout=True)
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        rowLabels=[POLICY_LABELS[row["policy"]] for row in rows],
        colLabels=short_columns,
        cellLoc="center", rowLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.55)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#E0E0F0")
            cell.set_text_props(weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    axis.set_title(
        "Conditional-precision summary (gain = sqrt(Fpolicy/Fbaseline) - 1; closure shows mean/max |log ratio|)",
        pad=8,
    )
    _save_figure(fig, output / "final_conditional_summary_table", config["plots"])
    _write_rows(output / "final_conditional_summary_table", rows)


def _run_Cnn_pseudoexperiments(
    policies: dict[str, ConditionalFlow],
    results: dict[str, dict[str, Any]],
    config: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> list[dict[str, Any]]:
    settings = config["evaluation"]
    offsets = config["seed_offsets"]
    nominal = float(base["physics"]["nominal_C"])
    total_template_events = int(settings["pseudo_template_events"])
    chunk_size = int(settings["pseudo_template_chunk_size"])
    bins = int(settings["pseudo_bins"])
    edges = np.linspace(-1.0, 1.0, bins + 1)
    scan_bounds = base["inference"]["scan_range"]
    scan = np.linspace(float(scan_bounds[0]), float(scan_bounds[1]), int(base["inference"]["scan_points"]))
    counts = {name: np.zeros(bins, dtype=np.float64) for name in policies}
    derivatives = {name: np.zeros(bins, dtype=np.float64) for name in policies}
    chunks = (total_template_events + chunk_size - 1) // chunk_size
    progress = tqdm(total=chunks * len(policies), desc="conditional Cnn templates", unit="policy-chunk")
    for chunk in range(chunks):
        count = min(chunk_size, total_template_events - chunk * chunk_size)
        events = generate_events(
            count, nominal, base, device,
            make_generator(device, int(base["seed"]) + int(offsets["pseudo_templates"]) + chunk),
        )
        truth_score = events["x"].cpu().numpy() / (1.0 + nominal * events["x"].cpu().numpy())
        for name, policy in policies.items():
            reconstructed = reconstruct_policy(
                policy, events, base,
                make_generator(device, int(base["seed"]) + int(offsets["pseudo_template_reconstruction"]) + chunk),
            )
            valid = reconstructed["valid"].cpu().numpy().astype(bool)
            indices = np.clip(
                np.searchsorted(edges, reconstructed["y"].cpu().numpy()[valid], side="right") - 1,
                0, bins - 1,
            )
            counts[name] += np.bincount(indices, minlength=bins)
            derivatives[name] += np.bincount(indices, weights=truth_score[valid], minlength=bins)
            progress.update()
    progress.close()
    exposure = int(settings["events_per_pseudo_experiment"])
    scale = exposure / total_template_events
    # This is the exact nominal weight ratio
    # (1 + C*x) / (1 + C0*x) = 1 + (C - C0)*x/(1 + C0*x), aggregated by bin.
    templates = {
        name: scale * (
            counts[name][None, :] + (scan[:, None] - nominal) * derivatives[name][None, :]
        )
        for name in policies
    }
    estimates = {name: [] for name in policies}
    fit_rows: list[dict[str, Any]] = []
    for experiment in tqdm(range(int(settings["pseudo_experiments"])), desc="conditional Cnn pseudo-experiments", unit="toy"):
        events = generate_events(
            exposure, nominal, base, device,
            make_generator(device, int(base["seed"]) + int(offsets["pseudo_events"]) + experiment),
        )
        for name, policy in policies.items():
            reconstructed = reconstruct_policy(
                policy, events, base,
                make_generator(device, int(base["seed"]) + int(offsets["pseudo_reconstruction"]) + experiment),
            )
            valid = reconstructed["valid"].cpu().numpy().astype(bool)
            observed, _ = np.histogram(reconstructed["y"].cpu().numpy()[valid], bins=edges)
            estimate, lower, upper, _ = fit_poisson(observed, templates[name], scan)
            estimates[name].append(estimate)
            fit_rows.append({
                "policy": name, "pseudo_experiment": experiment, "C_hat": estimate,
                "lower_error": lower, "upper_error": upper,
            })
    _write_rows(output / "final_Cnn_pseudoexperiment_fits", fit_rows)
    result_rows: list[dict[str, Any]] = []
    Cnn_index = list(SPIN_PARAMETER_NAMES).index("C_nn")
    for name in policies:
        fisher = float(results[name]["individual_fisher_per_generated_event"][Cnn_index])
        predicted_sigma = float(1.0 / np.sqrt(fisher * exposure))
        direct_fisher = float(results[name]["direct_Cnn_fisher_per_generated_event"])
        observed_std = float(np.std(estimates[name], ddof=1))
        result_rows.append({
            "policy": name,
            "nominal_Cnn": nominal,
            "template_events": total_template_events,
            "pseudo_experiments": int(settings["pseudo_experiments"]),
            "events_per_pseudo_experiment": exposure,
            "score_Fisher_per_event": fisher,
            "direct_binned_Fisher_per_event": direct_fisher,
            "predicted_conditional_sigma": predicted_sigma,
            "pseudoexperiment_Std_C_hat": observed_std,
            "PE_to_predicted_sigma_closure": observed_std / predicted_sigma,
            "mean_C_hat": float(np.mean(estimates[name])),
        })
    _write_rows(output / "final_Cnn_pseudoexperiment_closure", result_rows)
    return result_rows


def _write_interpretation(
    rows: list[dict[str, Any]], results: dict[str, dict[str, Any]],
    pseudo_rows: list[dict[str, Any]], output: Path,
) -> None:
    summary = {row["policy"]: row for row in rows}
    cdiag = summary["cdiag"]
    bc5 = summary["bc5"]
    untargeted = [name for name in SPIN_PARAMETER_NAMES if name not in EXPECTED_TARGET_SETS["BC5"]]
    cdiag_ratio = np.asarray(results["cdiag"]["conditional_sigma_ratio_to_baseline"])
    bc5_ratio = np.asarray(results["bc5"]["conditional_sigma_ratio_to_baseline"])
    untargeted_delta = np.asarray([
        bc5_ratio[list(SPIN_PARAMETER_NAMES).index(name)]
        - cdiag_ratio[list(SPIN_PARAMETER_NAMES).index(name)]
        for name in untargeted
    ])
    b_gains = [
        float(cdiag_ratio[list(SPIN_PARAMETER_NAMES).index(name)]
              / bc5_ratio[list(SPIN_PARAMETER_NAMES).index(name)] - 1.0)
        for name in ("B_A_n", "B_B_n")
    ]
    c_preserved = [
        bc5[f"{name} conditional precision gain"] - cdiag[f"{name} conditional precision gain"]
        for name in ("C_nn", "C_rr", "C_kk")
    ]
    efficiency_ratio = float(results["bc5"]["valid_efficiency"] / results["cdiag"]["valid_efficiency"])
    cdiag_per_valid = np.asarray(results["cdiag"]["information_per_valid"])
    bc5_per_valid = np.asarray(results["bc5"]["information_per_valid"])
    per_valid_ratios = [
        float(bc5_per_valid[list(SPIN_PARAMETER_NAMES).index(name)]
              / cdiag_per_valid[list(SPIN_PARAMETER_NAMES).index(name)])
        for name in EXPECTED_TARGET_SETS["BC5"]
    ]
    tradeoff = [
        float(cdiag_ratio[list(SPIN_PARAMETER_NAMES).index(name)]
              / bc5_ratio[list(SPIN_PARAMETER_NAMES).index(name)] - 1.0)
        for name in EXPECTED_TARGET_SETS["BC5"]
    ]
    lines = [
        "# Conditional spin-measurement interpretation",
        "",
        "All quoted uncertainties are conditional, with the other spin coefficients fixed.",
        "The smoke configuration is software validation only when fewer than 50 pseudo-experiments are used.",
        "",
        "## Cdiag versus BC5",
        "",
        f"1. BC5-versus-Cdiag B-target precision gains are {b_gains[0]:+.4f} (B_A_n) and {b_gains[1]:+.4f} (B_B_n).",
        f"2. BC5 minus Cdiag precision-gain changes for C_nn, C_rr, C_kk are {c_preserved[0]:+.4f}, {c_preserved[1]:+.4f}, {c_preserved[2]:+.4f}.",
        f"3. Across untargeted Full-15 components, the largest BC5-minus-Cdiag sigma-ratio change is {float(np.max(np.abs(untargeted_delta))):.4f}.",
        f"4. J_cond changes from {cdiag['J_cond']:.6f} (Cdiag) to {bc5['J_cond']:.6f} (BC5).",
        f"5. BC5/Cdiag valid efficiency is {efficiency_ratio:.4f}; information-per-valid ratios for C_nn, C_rr, C_kk, B_A_n, B_B_n are "
        + ", ".join(f"{value:.4f}" for value in per_valid_ratios) + ".",
        "",
        "BC5-versus-Cdiag per-target precision changes are "
        + ", ".join(f"{name} {value:+.4f}" for name, value in zip(EXPECTED_TARGET_SETS["BC5"], tradeoff)) + ".",
        "",
        "## Final C_nn closure",
        "",
    ]
    lines.extend(
        f"- {POLICY_LABELS[row['policy']]}: predicted sigma {row['predicted_conditional_sigma']:.5f}, "
        f"pseudo-experiment Std(C_hat) {row['pseudoexperiment_Std_C_hat']:.5f}, closure {row['PE_to_predicted_sigma_closure']:.3f}."
        for row in pseudo_rows
    )
    (output / "conditional_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_new_training_output(output: Path, config: dict[str, Any]) -> None:
    existing = [
        output / "checkpoints" / key / "best_validation_policy.pt"
        for key in POLICY_KEYS.values()
    ]
    if any(path.exists() for path in existing) and not bool(config["allow_overwrite"]):
        raise FileExistsError(
            f"Conditional checkpoints already exist under {output}; use a new output directory "
            "or explicitly set allow_overwrite: true"
        )


def run_spin_conditional(
    spin_config_path: str | Path,
    mode: str,
    device_override: str | None = None,
    output_override: str | None = None,
    cnn_output_override: str | None = None,
) -> None:
    config, base = _load_conditional_config(spin_config_path)
    if device_override is not None:
        base["device"] = device_override
    if output_override is not None:
        config["output_dir"] = output_override
    if cnn_output_override is not None:
        config["cnn_output_dir"] = cnn_output_override
    seed_everything(int(base["seed"]))
    device = resolve_device(str(base.get("device", "auto")))
    output = Path(config["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (output / "resolved_spin_conditional_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    with (output / "resolved_base_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(base, stream, sort_keys=False)
    print(f"Using device: {device}", flush=True)
    baseline = _load_baseline(config, base, device)
    samples = _generate_fixed_samples(config, base, device)
    histories: dict[str, list[dict[str, Any]]]
    if mode in {"spin-conditional-run", "spin-conditional-train"}:
        _check_new_training_output(output, config)
        histories = {}
        for target_index, (target, parameters) in enumerate(config["target_sets"].items()):
            _, histories[target] = _train_target_policy(
                baseline, target, list(parameters), target_index, samples,
                config, base, device, output,
            )
    else:
        histories = _load_training_histories(config, output)
    if mode in {"spin-conditional-run", "spin-conditional-evaluate"}:
        policies = _load_study_policies(baseline, config, base, device, output)
        results = _evaluate_full15(policies, samples, config, base, device, output)
        _set_figure_style()
        _plot_conditional_multitarget(results, config, output)
        _plot_full15_transfer(results, config, output)
        _plot_accept_reject(histories, config, output)
        _plot_score_closure(histories, config, output)
        _plot_information_decomposition(results, config, output)
        summary = _summary_rows(results, histories, config)
        _plot_summary_table(summary, config, output)
        pseudo_rows = _run_Cnn_pseudoexperiments(
            policies, results, config, base, device, output,
        )
        _write_interpretation(summary, results, pseudo_rows, output)
    print(f"Conditional-spin results written to {output}", flush=True)
