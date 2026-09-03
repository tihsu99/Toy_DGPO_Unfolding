from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import yaml

from .core import load_config, make_generator, resolve_device, seed_everything
from .flow import ConditionalFlow
from .training import make_flow, reconstruct_policy, slice_events
from .ztautau import candidate_reconstruction, generate_events


SPIN_PARAMETER_NAMES = (
    "B_A_r", "B_A_n", "B_A_k", "B_B_r", "B_B_n", "B_B_k",
    "C_rr", "C_rn", "C_rk", "C_nr", "C_nn", "C_nk", "C_kr", "C_kn", "C_kk",
)
SPIN_PARAMETER_INDEX = {name: index for index, name in enumerate(SPIN_PARAMETER_NAMES)}


class VectorScoreModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int, layers: int):
        super().__init__()
        modules: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            modules.extend((nn.Linear(current, width), nn.SiLU()))
            current = width
        modules.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*modules)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def truth_spin_observables(events: dict[str, torch.Tensor]) -> torch.Tensor:
    h_a = events["h_a_truth"]
    h_b = events["h_b_truth"]
    correlations = (h_a[..., :, None] * h_b[..., None, :]).reshape(*h_a.shape[:-1], 9)
    return torch.cat((h_a, h_b, correlations), dim=-1)


def truth_spin_scores(events: dict[str, torch.Tensor], nominal_C: float) -> torch.Tensor:
    observables = truth_spin_observables(events)
    g0 = 1.0 + nominal_C * observables[..., SPIN_PARAMETER_INDEX["C_nn"]]
    return observables / g0[..., None]


def _selected_scores(events: dict[str, torch.Tensor], nominal_C: float, names: list[str]) -> torch.Tensor:
    indices = [SPIN_PARAMETER_INDEX[name] for name in names]
    return truth_spin_scores(events, nominal_C)[..., indices]


def _load_spin_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spin_path = Path(path).resolve()
    with spin_path.open(encoding="utf-8") as stream:
        spin = yaml.safe_load(stream)
    if not isinstance(spin, dict):
        raise ValueError("Spin-matrix configuration root must be a mapping")
    required = {"base_config", "cnn_output_dir", "output_dir", "study", "multi_training", "seed_offsets", "plots"}
    missing = required.difference(spin)
    if missing:
        raise ValueError(f"Missing spin-matrix configuration sections: {sorted(missing)}")
    base_path = Path(spin["base_config"])
    if not base_path.is_absolute():
        base_path = spin_path.parent / base_path
    base = load_config(base_path)
    targets = spin["multi_training"]["target_sets"]
    for target_name, parameter_names in targets.items():
        unknown = sorted(set(parameter_names).difference(SPIN_PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"Unknown parameters in target set {target_name}: {unknown}")
    enabled = list(spin["multi_training"]["enabled_targets"])
    if spin["multi_training"]["primary_target"] not in enabled:
        raise ValueError("multi_training.primary_target must be enabled")
    if any(name not in targets for name in enabled):
        raise ValueError("Every enabled multi-measurement target must be defined")
    if not bool(spin["multi_training"]["equal_weights"]):
        raise ValueError("This first study requires equal normalized A-optimal weights")
    if not 0.0 <= float(spin["multi_training"]["lambda_num"]) <= 1.0e-6:
        raise ValueError("multi_training.lambda_num must remain a numerical ridge no larger than 1e-6")
    for section, keys in (
        ("study", ("score_events", "evaluation_events", "score_epochs", "batch_size", "null_pseudo_experiments", "null_events_per_experiment")),
        ("multi_training", ("max_refresh_rounds", "dgpo_epochs_per_round", "score_events", "training_events", "validation_events", "score_epochs", "batch_size")),
    ):
        if any(int(spin[section][key]) < 1 for key in keys):
            raise ValueError(f"All configured counts in {section} must be positive")
    if int(spin["multi_training"]["group_size"]) < 2:
        raise ValueError("multi_training.group_size must be at least two")
    if float(spin["multi_training"]["condition_number_max"]) <= 1.0:
        raise ValueError("multi_training.condition_number_max must exceed one")
    return spin, base


def _load_flow(path: Path, base: dict[str, Any], device: torch.device) -> ConditionalFlow:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("method_version") not in {1, 3}:
        raise RuntimeError(f"Unsupported policy checkpoint: {path}")
    flow = make_flow(base, device)
    flow.load_state_dict(payload["state_dict"])
    return flow.eval()


def _load_cnn_policies(
    spin: dict[str, Any], base: dict[str, Any], device: torch.device,
) -> dict[str, ConditionalFlow]:
    checkpoints = Path(spin["cnn_output_dir"]).expanduser().resolve() / "checkpoints"
    return {
        "baseline": _load_flow(checkpoints / "baseline" / "best_validation_policy.pt", base, device),
        "cnn_only": _load_flow(
            checkpoints / "iterative_refresh_no_trust" / "best_validation_policy.pt", base, device,
        ),
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


def _train_vector_score(
    policy: ConditionalFlow,
    events: dict[str, torch.Tensor],
    parameter_names: list[str],
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    description: str,
    training_settings: dict[str, Any],
) -> tuple[VectorScoreModel, list[float]]:
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1)
    reconstructed = reconstruct_policy(policy, events, base, make_generator(device, seed))
    target = _selected_scores(events, float(base["physics"]["nominal_C"]), parameter_names)
    valid_indices = torch.nonzero(reconstructed["valid"], as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError("The policy produced no valid events for vector-score training")
    model = VectorScoreModel(
        6, len(parameter_names), int(training_settings["score_hidden_width"]),
        int(training_settings["score_hidden_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(base["training"]["learning_rate"]),
        weight_decay=float(base["training"]["weight_decay"]),
    )
    batch_size = int(training_settings["batch_size"])
    losses: list[float] = []
    progress = tqdm(range(int(training_settings["score_epochs"])), desc=description, unit="epoch")
    for _ in progress:
        permutation = valid_indices[torch.randperm(valid_indices.numel(), device=device)]
        total = 0.0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            loss = (model(reconstructed["spin_features"][index]) - target[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * index.numel()
        mean_loss = total / permutation.numel()
        losses.append(mean_loss)
        progress.set_postfix(mse=f"{mean_loss:.5f}", valid=f"{valid_indices.numel() / events['x'].numel():.3f}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, losses


@torch.no_grad()
def _score_reconstruction(
    model: VectorScoreModel, reconstructed: dict[str, torch.Tensor], dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    score = model(reconstructed["spin_features"]).to(dtype)
    return score * reconstructed["valid"].to(dtype)[..., None]


def fisher_matrix_per_generated_event(scores: torch.Tensor) -> torch.Tensor:
    flat = scores.reshape(-1, scores.shape[-1]).to(torch.float64)
    return flat.T @ flat / flat.shape[0]


def replacement_fisher_matrix(
    total_reference: torch.Tensor,
    event_reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    return total_reference[None, None, :, :] - event_reference[:, None, :, :] + candidate


def _matrix_stats(matrix: np.ndarray) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    positive = eigenvalues[eigenvalues > 0.0]
    condition = float(eigenvalues[-1] / positive[0]) if positive.size == eigenvalues.size else float("inf")
    return {
        "eigenvalues": eigenvalues.tolist(),
        "smallest_eigenvalue": float(eigenvalues[0]),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "condition_number": condition,
    }


def _stable_inverse(
    matrix: np.ndarray, condition_max: float, relative_tolerance: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    stats = _matrix_stats(matrix)
    stable = (
        stats["smallest_eigenvalue"] > relative_tolerance * stats["largest_eigenvalue"]
        and stats["condition_number"] <= condition_max
    )
    stats["stable"] = bool(stable)
    return (np.linalg.inv(matrix) if stable else None), stats


def _subset_result(
    fisher: np.ndarray, parameter_names: list[str], subset: list[str], settings: dict[str, Any],
) -> dict[str, Any]:
    indices = [parameter_names.index(name) for name in subset]
    selected = fisher[np.ix_(indices, indices)]
    covariance, stats = _stable_inverse(
        selected, float(settings["condition_number_max"]),
        float(settings["eigenvalue_relative_tolerance"]),
    )
    result: dict[str, Any] = {"parameters": subset, "fisher": selected.tolist(), **stats}
    if covariance is not None:
        sigma = np.sqrt(np.diag(covariance))
        result.update({
            "covariance_per_inverse_event": covariance.tolist(),
            "profiled_sigma_per_sqrt_event": sigma.tolist(),
            "correlation": (covariance / np.outer(sigma, sigma)).tolist(),
        })
    return result


def _evaluate_policy_spin_matrix(
    name: str,
    policy: ConditionalFlow,
    model: VectorScoreModel,
    events: dict[str, torch.Tensor],
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    reconstructed = reconstruct_policy(policy, events, base, make_generator(device, seed))
    scores = _score_reconstruction(model, reconstructed)
    fisher = fisher_matrix_per_generated_event(scores).cpu().numpy()
    covariance, stats = _stable_inverse(
        fisher, float(spin["study"]["condition_number_max"]),
        float(spin["study"]["eigenvalue_relative_tolerance"]),
    )
    conditional = 1.0 / np.sqrt(np.clip(np.diag(fisher), 1.0e-300, None))
    cosine = (reconstructed["k_a"] * events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
    result: dict[str, Any] = {
        "policy": name,
        "parameters": list(SPIN_PARAMETER_NAMES),
        "fisher_per_generated_event": fisher.tolist(),
        "individual_fisher": np.diag(fisher).tolist(),
        "conditional_sigma_per_sqrt_event": conditional.tolist(),
        "valid_efficiency": float(reconstructed["valid"].float().mean()),
        "tau_axis_error": float(torch.acos(cosine).mean()),
        **stats,
        "Cdiag": _subset_result(
            fisher, list(SPIN_PARAMETER_NAMES), ["C_nn", "C_rr", "C_kk"], spin["study"],
        ),
        "BC5": _subset_result(
            fisher, list(SPIN_PARAMETER_NAMES),
            ["C_nn", "C_rr", "C_kk", "B_A_n", "B_B_n"], spin["study"],
        ),
    }
    if covariance is not None:
        sigma = np.sqrt(np.diag(covariance))
        result.update({
            "covariance_per_inverse_event": covariance.tolist(),
            "profiled_sigma_per_sqrt_event": sigma.tolist(),
            "correlation": (covariance / np.outer(sigma, sigma)).tolist(),
        })
    return result, reconstructed


def _run_null_test(
    policies: dict[str, ConditionalFlow],
    models: dict[str, VectorScoreModel],
    calibration: dict[str, tuple[np.ndarray, np.ndarray]],
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> list[dict[str, Any]]:
    settings = spin["study"]
    offsets = spin["seed_offsets"]
    experiments = int(settings["null_pseudo_experiments"])
    events_per = int(settings["null_events_per_experiment"])
    nominal_C = float(base["physics"]["nominal_C"])
    shifts = {name: [] for name in policies}
    progress = tqdm(total=experiments * len(policies), desc="spin null cross-talk", unit="policy-toy")
    for experiment in range(experiments):
        events = generate_events(
            events_per, nominal_C, base, device,
            make_generator(device, int(base["seed"]) + int(offsets["null_experiments"]) + experiment),
        )
        for name, policy in policies.items():
            reconstructed = reconstruct_policy(
                policy, events, base,
                make_generator(device, int(base["seed"]) + int(offsets["null_reconstruction"]) + experiment),
            )
            scores = _score_reconstruction(models[name], reconstructed).cpu().numpy()
            fisher, compensator = calibration[name]
            full_score = scores.sum(axis=0) - events_per * compensator
            shifts[name].append(full_score / (events_per * np.diag(fisher)))
            progress.update()
    progress.close()
    rows: list[dict[str, Any]] = []
    for name in policies:
        values = np.asarray(shifts[name])
        fisher, _ = calibration[name]
        expected_sigma = 1.0 / np.sqrt(events_per * np.diag(fisher))
        for index, parameter in enumerate(SPIN_PARAMETER_NAMES):
            if parameter == "C_nn":
                continue
            rows.append({
                "policy": name,
                "parameter": parameter,
                "mean_local_shift": float(values[:, index].mean()),
                "expected_statistical_sigma": float(expected_sigma[index]),
                "normalized_pull": float(values[:, index].mean() / expected_sigma[index]),
                "pull_width": float((values[:, index] / expected_sigma[index]).std(ddof=1)),
                "pseudo_experiments": experiments,
            })
    _write_rows(output / "spin_matrix_null_crosstalk", rows)
    return rows


def _plot_information_transfer(results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path) -> None:
    baseline = np.asarray(results["baseline"]["individual_fisher"])
    optimized = np.asarray(results["cnn_only"]["individual_fisher"])
    information_ratio = optimized / baseline
    sigma_ratio = np.sqrt(baseline / optimized)
    positions = np.arange(len(SPIN_PARAMETER_NAMES))
    highlight = SPIN_PARAMETER_INDEX["C_nn"]
    colors = ["tab:red" if index == highlight else "tab:blue" for index in positions]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharex=True)
    axes[0].bar(positions, information_ratio, color=colors)
    axes[1].bar(positions, sigma_ratio, color=colors)
    axes[0].axhline(1.0, color="black", linestyle="--")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[0].set(ylabel=r"$F_{aa}^{C_{nn}\ policy}/F_{aa}^{baseline}$",
                title=r"Passive transfer from $C_{nn}$-only optimization; red is the trained target")
    axes[1].set(ylabel="Conditional sigma ratio", xticks=positions, xticklabels=SPIN_PARAMETER_NAMES)
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "spin_matrix_information_transfer.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    rows = [{
        "parameter": name,
        "information_ratio_cnn_to_baseline": float(information_ratio[index]),
        "conditional_sigma_ratio_cnn_to_baseline": float(sigma_ratio[index]),
    } for index, name in enumerate(SPIN_PARAMETER_NAMES)]
    _write_rows(output / "spin_matrix_information_transfer", rows)


def _plot_correlations(results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path) -> None:
    subsets = ("Cdiag", "BC5")
    policies = ("baseline", "cnn_only")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), constrained_layout=True)
    image = None
    for row_index, policy in enumerate(policies):
        for column_index, subset in enumerate(subsets):
            axis = axes[row_index, column_index]
            result = results[policy][subset]
            labels = result["parameters"]
            if not result["stable"]:
                axis.axis("off")
                axis.text(0.5, 0.5, f"Ill-conditioned\ncond={result['condition_number']:.3g}",
                          ha="center", va="center")
            else:
                correlation = np.asarray(result["correlation"])
                image = axis.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
                axis.set(xticks=range(len(labels)), xticklabels=labels,
                         yticks=range(len(labels)), yticklabels=labels)
                axis.tick_params(axis="x", rotation=45)
            axis.set_title(f"{policy.replace('_', ' ')}: {subset}")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label=r"$\rho_{ab}$", shrink=0.8, pad=0.03)
    fig.suptitle("Profiled spin-parameter correlations")
    fig.savefig(output / "spin_matrix_correlations.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)


def run_passive_study(
    policies: dict[str, ConditionalFlow], spin: dict[str, Any], base: dict[str, Any],
    device: torch.device, output: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, VectorScoreModel], dict[str, dict[str, torch.Tensor]]]:
    settings = spin["study"]
    offsets = spin["seed_offsets"]
    nominal_C = float(base["physics"]["nominal_C"])
    score_events = generate_events(
        int(settings["score_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["passive_score"])),
    )
    evaluation_events = generate_events(
        int(settings["evaluation_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["passive_evaluation"])),
    )
    models: dict[str, VectorScoreModel] = {}
    results: dict[str, dict[str, Any]] = {}
    reconstructions: dict[str, dict[str, torch.Tensor]] = {}
    calibration: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy_index, (name, policy) in enumerate(policies.items()):
        seed = int(base["seed"]) + int(offsets["passive_score"]) + 1000 * policy_index
        model, losses = _train_vector_score(
            policy, score_events, list(SPIN_PARAMETER_NAMES), spin, base, device, seed,
            f"15D diagnostic score: {name}", settings,
        )
        torch.save({
            "method_version": 1, "diagnostic_only": True,
            "parameters": list(SPIN_PARAMETER_NAMES), "state_dict": model.state_dict(),
        }, output / f"diagnostic_spin_score_{name}.pt")
        result, reconstructed = _evaluate_policy_spin_matrix(
            name, policy, model, evaluation_events, spin, base, device,
            int(base["seed"]) + int(offsets["passive_evaluation"]) + 1000 * policy_index,
        )
        result["score_final_loss"] = losses[-1]
        models[name] = model
        results[name] = result
        reconstructions[name] = reconstructed
        scores = _score_reconstruction(model, reconstructed).cpu().numpy()
        calibration[name] = (
            np.asarray(result["fisher_per_generated_event"]), scores.mean(axis=0),
        )
    with (output / "spin_matrix_diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    if "baseline" in results and "cnn_only" in results:
        _plot_information_transfer(results, spin, output)
        _plot_correlations(results, spin, output)
        _run_null_test(
            {name: policies[name] for name in ("baseline", "cnn_only")},
            {name: models[name] for name in ("baseline", "cnn_only")},
            {name: calibration[name] for name in ("baseline", "cnn_only")},
            spin, base, device, output,
        )
    return results, models, reconstructions


def _torch_a_objective(
    fisher: torch.Tensor, baseline_variances: torch.Tensor, lambda_num: float,
) -> torch.Tensor:
    dimension = fisher.shape[-1]
    scale = torch.diagonal(fisher, dim1=-2, dim2=-1).mean(dim=-1)
    ridge = lambda_num * scale
    identity = torch.eye(dimension, dtype=fisher.dtype, device=fisher.device)
    regularized = fisher + ridge[..., None, None] * identity
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    if torch.any(info != 0):
        raise RuntimeError("Fisher matrix is not positive definite under the configured numerical ridge")
    covariance = torch.cholesky_inverse(cholesky)
    normalized_variance = torch.diagonal(covariance, dim1=-2, dim2=-1) / baseline_variances
    return normalized_variance.mean(dim=-1)


def _numpy_objectives(
    fisher: np.ndarray, baseline_variances: np.ndarray, compute_d_optimal: bool,
) -> tuple[float, float | None, np.ndarray]:
    covariance = np.linalg.inv(fisher)
    a_objective = float(np.mean(np.diag(covariance) / baseline_variances))
    d_objective = None
    if compute_d_optimal:
        scales = np.sqrt(baseline_variances)
        normalized_fisher = fisher * np.outer(scales, scales)
        sign, log_determinant = np.linalg.slogdet(normalized_fisher)
        d_objective = float(-log_determinant) if sign > 0 else float("inf")
    return a_objective, d_objective, covariance


def _multi_policy_name(target_name: str, primary_target: str) -> str:
    return "iterative_refresh_multi_no_trust" if target_name == primary_target else f"iterative_refresh_multi_no_trust_{target_name}"


def _save_multi_policy(
    policy: ConditionalFlow, path: Path, target_name: str, round_index: int,
) -> None:
    torch.save({
        "method_version": 1, "policy": "iterative_refresh_multi_no_trust",
        "target_set": target_name, "round": round_index, "state_dict": policy.state_dict(),
    }, path)


def _train_multi_local_update(
    current: ConditionalFlow,
    active_score: VectorScoreModel,
    events: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    fisher_reference: torch.Tensor,
    baseline_variances: torch.Tensor,
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    description: str,
) -> tuple[ConditionalFlow, list[dict[str, float]]]:
    settings = spin["multi_training"]
    policy = copy.deepcopy(current).to(device).train()
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    current.eval()
    for parameter in current.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    batch_size = int(settings["batch_size"])
    group_size = int(settings["group_size"])
    lambda_num = float(settings["lambda_num"])
    reference_scores = reference["vector_score"].to(torch.float64)
    reference_outer = reference_scores[..., :, None] * reference_scores[..., None, :]
    reference_objective = _torch_a_objective(
        fisher_reference, baseline_variances, lambda_num,
    ).detach()
    generator = make_generator(device, seed)
    history: list[dict[str, float]] = []
    progress = tqdm(
        range(int(settings["dgpo_epochs_per_round"])), desc=description, unit="epoch",
    )
    for epoch in progress:
        permutation = torch.randperm(events["x"].numel(), device=device, generator=generator)
        reward_total = 0.0
        invalid_total = 0.0
        seen = 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            batch = slice_events(events, index)
            actions, _ = policy.sample(batch["context"], group_size, generator)
            with torch.no_grad():
                reconstruction = candidate_reconstruction(batch, actions.detach(), base)
                candidate_score = _score_reconstruction(active_score, reconstruction)
                candidate_outer = candidate_score[..., :, None] * candidate_score[..., None, :]
                candidate_fisher = replacement_fisher_matrix(
                    fisher_reference, reference_outer[index], candidate_outer,
                )
                candidate_objective = _torch_a_objective(
                    candidate_fisher, baseline_variances, lambda_num,
                )
                reward = 0.5 * torch.log(reference_objective / candidate_objective)
                reward = reward * float(settings["reward_scale"])
                advantage = reward - (reward.sum(dim=1, keepdim=True) - reward) / (group_size - 1)
            expanded_context = batch["context"][:, None, :].expand(-1, group_size, -1)
            log_probability = policy.log_prob(actions.detach(), expanded_context)
            loss = -(advantage.detach() * log_probability).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings["grad_clip_norm"]))
            optimizer.step()
            count = index.numel()
            reward_total += reward.mean().item() * count
            invalid_total += (~reconstruction["valid"]).float().mean().item() * count
            seen += count
        row = {
            "local_epoch": float(epoch + 1),
            "reward": reward_total / seen,
            "invalid_fraction": invalid_total / seen,
        }
        history.append(row)
        progress.set_postfix(reward=f"{row['reward']:.4g}", invalid=f"{row['invalid_fraction']:.3f}")
    return policy.eval(), history


def train_multi_measurement_policy(
    baseline: ConditionalFlow,
    target_name: str,
    parameter_names: list[str],
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> tuple[ConditionalFlow, list[dict[str, Any]]]:
    settings = spin["multi_training"]
    offsets = spin["seed_offsets"]
    nominal_C = float(base["physics"]["nominal_C"])
    policy_name = _multi_policy_name(target_name, str(settings["primary_target"]))
    checkpoint_dir = output / "checkpoints" / policy_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    score_events = generate_events(
        int(settings["score_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["multi_score"])),
    )
    validation_events = generate_events(
        int(settings["validation_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["multi_validation"])),
    )
    active_score, _ = _train_vector_score(
        baseline, score_events, parameter_names, spin, base, device,
        int(base["seed"]) + int(offsets["multi_score"]),
        f"{target_name} vector score pi0", settings,
    )
    validation_seed = int(base["seed"]) + int(offsets["multi_reconstruction"])
    baseline_validation = reconstruct_policy(
        baseline, validation_events, base, make_generator(device, validation_seed),
    )
    baseline_validation_fisher = fisher_matrix_per_generated_event(
        _score_reconstruction(active_score, baseline_validation),
    ).cpu().numpy()
    baseline_covariance, baseline_stats = _stable_inverse(
        baseline_validation_fisher, float(settings["condition_number_max"]), 0.0,
    )
    if baseline_covariance is None:
        raise RuntimeError(f"Baseline {target_name} validation Fisher is ill-conditioned: {baseline_stats}")
    baseline_validation_variances = np.diag(baseline_covariance)
    current = copy.deepcopy(baseline).eval()
    best_policy = copy.deepcopy(current).eval()
    best_objective = 1.0
    best_round = 0
    patience_reference = 1.0
    rounds_without_improvement = 0
    rows: list[dict[str, Any]] = []
    _save_multi_policy(current, checkpoint_dir / "pi_00_policy.pt", target_name, 0)
    for round_index in range(int(settings["max_refresh_rounds"])):
        training_events = generate_events(
            int(settings["training_events"]), nominal_C, base, device,
            make_generator(
                device, int(base["seed"]) + int(offsets["multi_training"]) + 10000 * round_index,
            ),
        )
        reference = reconstruct_policy(
            current, training_events, base,
            make_generator(device, int(base["seed"]) + int(offsets["multi_reconstruction"]) + 10000 * round_index),
        )
        reference["vector_score"] = _score_reconstruction(active_score, reference)
        fisher_reference = (
            reference["vector_score"].T @ reference["vector_score"]
        ).to(torch.float64)
        fisher_reference_per_event = (fisher_reference / training_events["x"].numel()).cpu().numpy()
        train_covariance, train_stats = _stable_inverse(
            fisher_reference_per_event, float(settings["condition_number_max"]), 0.0,
        )
        if train_covariance is None:
            rows.append({
                "round": round_index, "status": "stopped_ill_conditioned_training_fisher", **train_stats,
            })
            break
        baseline_train_variances = torch.as_tensor(
            baseline_validation_variances / training_events["x"].numel(),
            device=device, dtype=torch.float64,
        )
        train_objective, _, _ = _numpy_objectives(
            fisher_reference_per_event, baseline_validation_variances, False,
        )
        next_policy, local_history = _train_multi_local_update(
            current, active_score, training_events, reference, fisher_reference,
            baseline_train_variances, spin, base, device,
            int(base["seed"]) + int(offsets["multi_training"]) + 20000 * round_index,
            f"{policy_name} round {round_index + 1}",
        )
        next_score, score_losses = _train_vector_score(
            next_policy, score_events, parameter_names, spin, base, device,
            int(base["seed"]) + int(offsets["multi_score"]) + 1000 * (round_index + 1),
            f"{target_name} vector score pi{round_index + 1}", settings,
        )
        updated_training = reconstruct_policy(
            next_policy, training_events, base,
            make_generator(device, int(base["seed"]) + int(offsets["multi_reconstruction"]) + 10000 * round_index),
        )
        updated_training_fisher = fisher_matrix_per_generated_event(
            _score_reconstruction(next_score, updated_training),
        ).cpu().numpy()
        updated_train_covariance, updated_train_stats = _stable_inverse(
            updated_training_fisher, float(settings["condition_number_max"]), 0.0,
        )
        if updated_train_covariance is None:
            rows.append({
                "round": round_index + 1,
                "status": "stopped_ill_conditioned_refreshed_training_fisher",
                **updated_train_stats,
            })
            current = next_policy
            active_score = next_score
            break
        updated_train_objective, _, _ = _numpy_objectives(
            updated_training_fisher, baseline_validation_variances, False,
        )
        validation = reconstruct_policy(
            next_policy, validation_events, base, make_generator(device, validation_seed),
        )
        validation_fisher = fisher_matrix_per_generated_event(
            _score_reconstruction(next_score, validation),
        ).cpu().numpy()
        validation_covariance, validation_stats = _stable_inverse(
            validation_fisher, float(settings["condition_number_max"]), 0.0,
        )
        if validation_covariance is None:
            rows.append({
                "round": round_index + 1, "status": "stopped_ill_conditioned_validation_fisher",
                **validation_stats,
            })
            current = next_policy
            active_score = next_score
            break
        validation_objective, d_objective, validation_covariance = _numpy_objectives(
            validation_fisher, baseline_validation_variances,
            bool(settings["compute_d_optimal_diagnostic"]),
        )
        conditional_sigma = 1.0 / np.sqrt(np.diag(validation_fisher))
        profiled_sigma = np.sqrt(np.diag(validation_covariance))
        baseline_conditional = 1.0 / np.sqrt(np.diag(baseline_validation_fisher))
        baseline_profiled = np.sqrt(baseline_validation_variances)
        row = {
            "round": round_index + 1,
            "total_dgpo_epoch": (round_index + 1) * int(settings["dgpo_epochs_per_round"]),
            "status": "ok",
            "J_train_before_update": train_objective,
            "J_train_after_refresh": updated_train_objective,
            "J_validation": validation_objective,
            "D_validation": d_objective,
            "validation_fisher_per_generated_event": validation_fisher.tolist(),
            "conditional_sigma_per_sqrt_event": conditional_sigma.tolist(),
            "profiled_sigma_per_sqrt_event": profiled_sigma.tolist(),
            "conditional_sigma_ratio_to_baseline": (conditional_sigma / baseline_conditional).tolist(),
            "profiled_sigma_ratio_to_baseline": (profiled_sigma / baseline_profiled).tolist(),
            "score_loss": score_losses[-1],
            "mean_reward": float(np.mean([item["reward"] for item in local_history])),
            "invalid_fraction": float(1.0 - validation["valid"].float().mean()),
            "lambda_num_relative": float(settings["lambda_num"]),
            "numerical_ridge_absolute": float(
                settings["lambda_num"] * torch.trace(fisher_reference).item() / len(parameter_names)
            ),
            "train_reference_smallest_eigenvalue": train_stats["smallest_eigenvalue"],
            "train_reference_largest_eigenvalue": train_stats["largest_eigenvalue"],
            "train_reference_condition_number": train_stats["condition_number"],
            "train_refreshed_smallest_eigenvalue": updated_train_stats["smallest_eigenvalue"],
            "train_refreshed_largest_eigenvalue": updated_train_stats["largest_eigenvalue"],
            "train_refreshed_condition_number": updated_train_stats["condition_number"],
            "validation_eigenvalues": validation_stats["eigenvalues"],
            "validation_smallest_eigenvalue": validation_stats["smallest_eigenvalue"],
            "validation_largest_eigenvalue": validation_stats["largest_eigenvalue"],
            "validation_condition_number": validation_stats["condition_number"],
        }
        rows.append(row)
        completed_round = round_index + 1
        _save_multi_policy(next_policy, checkpoint_dir / f"pi_{completed_round:02d}_policy.pt", target_name, completed_round)
        torch.save({
            "method_version": 1, "target_set": target_name, "parameters": parameter_names,
            "round": completed_round, "state_dict": next_score.state_dict(),
        }, checkpoint_dir / f"pi_{completed_round:02d}_score.pt")
        if validation_objective < best_objective:
            best_objective = validation_objective
            best_round = completed_round
            best_policy = copy.deepcopy(next_policy).eval()
        threshold = patience_reference * (1.0 - float(settings["min_relative_objective_improvement"]))
        if validation_objective < threshold:
            patience_reference = validation_objective
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        current = next_policy
        active_score = next_score
        if rounds_without_improvement >= int(settings["early_stop_patience_rounds"]):
            break
    final_round = max((int(row["round"]) for row in rows), default=0)
    _save_multi_policy(current, checkpoint_dir / "final_policy.pt", target_name, final_round)
    _save_multi_policy(best_policy, checkpoint_dir / "best_validation_policy.pt", target_name, best_round)
    root_checkpoint = output / "checkpoints" / f"{policy_name}.pt"
    _save_multi_policy(best_policy, root_checkpoint, target_name, best_round)
    with (output / f"multi_training_{target_name}.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "policy": policy_name, "target_set": target_name, "parameters": parameter_names,
            "selection_metric": "minimum_validation_normalized_A_optimal_objective",
            "best_round": best_round, "best_J_validation": best_objective,
            "final_round": final_round, "rounds": rows,
        }, stream, indent=2)
    return best_policy, rows


def _plot_multi_training(
    histories: dict[str, list[dict[str, Any]]], spin: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    for target_name, raw_rows in histories.items():
        rows = [row for row in raw_rows if row.get("status") == "ok"]
        if not rows:
            continue
        epochs = [row["total_dgpo_epoch"] for row in rows]
        axes[0].plot(epochs, [row["J_train_before_update"] for row in rows], marker="o",
                     linestyle=":", label=f"{target_name} train before update")
        axes[0].plot(epochs, [row["J_train_after_refresh"] for row in rows], marker="o",
                     linestyle="--", label=f"{target_name} train after refresh")
        axes[0].plot(epochs, [row["J_validation"] for row in rows], marker="s",
                     label=f"{target_name} validation")
        parameters = list(spin["multi_training"]["target_sets"][target_name])
        for index, parameter in enumerate(parameters):
            axes[1].plot(epochs, [row["conditional_sigma_ratio_to_baseline"][index] for row in rows],
                         marker="o", label=f"{target_name}: {parameter}")
            axes[2].plot(epochs, [row["profiled_sigma_ratio_to_baseline"][index] for row in rows],
                         marker="o", label=f"{target_name}: {parameter}")
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xlabel("Total DGPO epoch")
        axis.legend(frameon=False, fontsize=7)
    axes[0].set(ylabel="Normalized A-optimal objective J", title="Training and independent validation")
    axes[1].set(ylabel="Conditional sigma ratio", title=r"$1/\sqrt{F_{aa}}$")
    axes[2].set(ylabel="Profiled sigma ratio", title=r"$\sqrt{(F^{-1})_{aa}}$")
    fig.suptitle("Iterative multi-measurement DGPO without KL trust")
    fig.tight_layout()
    fig.savefig(output / "multi_measurement_training.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)


def _plot_multi_tradeoff(
    results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    target_parameters = []
    for target_name in spin["multi_training"]["enabled_targets"]:
        for parameter in spin["multi_training"]["target_sets"][target_name]:
            if parameter not in target_parameters:
                target_parameters.append(parameter)
    policy_order = ["baseline", "cnn_only"] + [
        f"multi_{target_name}" for target_name in spin["multi_training"]["enabled_targets"]
    ]
    policies = [name for name in policy_order if name in results]
    baseline_fisher = np.asarray(results["baseline"]["fisher_per_generated_event"])
    baseline_subset = _subset_result(
        baseline_fisher, list(SPIN_PARAMETER_NAMES), target_parameters, spin["study"],
    )
    baseline_conditional = 1.0 / np.sqrt(np.diag(baseline_fisher)[
        [SPIN_PARAMETER_INDEX[name] for name in target_parameters]
    ])
    baseline_profiled = np.asarray(baseline_subset.get("profiled_sigma_per_sqrt_event", []))
    positions = np.arange(len(target_parameters))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    rows: list[dict[str, Any]] = []
    for policy_index, name in enumerate(policies):
        fisher = np.asarray(results[name]["fisher_per_generated_event"])
        indices = [SPIN_PARAMETER_INDEX[item] for item in target_parameters]
        conditional = 1.0 / np.sqrt(np.diag(fisher)[indices])
        subset = _subset_result(fisher, list(SPIN_PARAMETER_NAMES), target_parameters, spin["study"])
        profiled = np.asarray(subset.get("profiled_sigma_per_sqrt_event", np.full(len(indices), np.nan)))
        conditional_ratio = conditional / baseline_conditional
        profiled_ratio = profiled / baseline_profiled if baseline_profiled.size else np.full(len(indices), np.nan)
        offset = (policy_index - (len(policies) - 1) / 2.0) * width
        axes[0].bar(positions + offset, conditional_ratio, width, label=name.replace("_", " "))
        axes[1].bar(positions + offset, profiled_ratio, width, label=name.replace("_", " "))
        for index, parameter in enumerate(target_parameters):
            rows.append({
                "policy": name, "parameter": parameter,
                "conditional_sigma_ratio_to_baseline": float(conditional_ratio[index]),
                "profiled_sigma_ratio_to_baseline": float(profiled_ratio[index]),
            })
    for axis in axes:
        axis.axhline(1.0, color="black", linestyle="--")
        axis.set(xticks=positions, xticklabels=target_parameters, ylabel="Sigma ratio to baseline")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_title("One-parameter conditional precision")
    axes[1].set_title("Jointly profiled precision")
    fig.suptitle("Cnn-only versus joint-measurement reconstruction tradeoff")
    fig.tight_layout()
    fig.savefig(output / "multi_measurement_tradeoff.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "multi_measurement_tradeoff", rows)


def _response_matrix(x: np.ndarray, y: np.ndarray, valid: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(y[valid], x[valid], bins=(edges, edges))
    return counts / np.clip(counts.sum(axis=0, keepdims=True), 1.0, None)


def _plot_cnn_vs_multitask_response(
    events: dict[str, torch.Tensor], reconstructions: dict[str, dict[str, torch.Tensor]],
    spin: dict[str, Any], output: Path,
) -> None:
    names = [name for name in ("baseline", "cnn_only", "multi_Cdiag") if name in reconstructions]
    x = events["x"].cpu().numpy()
    bins = int(spin["study"]["response_bins"])
    matrices = {}
    for name in names:
        reco = reconstructions[name]
        matrices[name] = _response_matrix(
            x, reco["y"].cpu().numpy(), reco["valid"].cpu().numpy().astype(bool), bins,
        )
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.6 * len(names), 4.4), squeeze=False, constrained_layout=True,
    )
    image = None
    for index, name in enumerate(names):
        image = axes[0, index].imshow(
            matrices[name], origin="lower", extent=(-1, 1, -1, 1), aspect="auto",
            cmap="magma", vmin=0.0, vmax=maximum,
        )
        axes[0, index].set(title=name.replace("_", " "), xlabel=r"Truth $C_{nn}$ analyzer $x$")
        axes[0, index].set_ylabel(r"Reconstructed $c_{nn}$")
    if image is not None:
        fig.colorbar(
            image, ax=axes.ravel().tolist(), label="Probability per reco bin", shrink=0.82, pad=0.03,
        )
    fig.suptitle(r"Policy-specific $c_{nn}$ response; full spin features drive the multi-task score")
    fig.savefig(output / "cnn_vs_multitask_response.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)


def _plot_spin_summary(
    results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    target_parameters = ["C_nn", "C_rr", "C_kk"]
    baseline_fisher = np.asarray(results["baseline"]["fisher_per_generated_event"])
    baseline_subset = _subset_result(
        baseline_fisher, list(SPIN_PARAMETER_NAMES), target_parameters, spin["study"],
    )
    baseline_sigma = np.asarray(baseline_subset.get("profiled_sigma_per_sqrt_event", [np.nan] * 3))
    columns = [
        "C_nn precision gain", "C_rr precision gain", "C_kk precision gain",
        "Average normalized variance", "Cdiag condition", "Valid efficiency", "Axis error [deg]",
    ]
    rows = []
    labels = []
    policy_order = ["cnn_only"] + [
        f"multi_{target_name}" for target_name in spin["multi_training"]["enabled_targets"]
    ]
    for name in policy_order:
        if name not in results:
            continue
        subset = results[name]["Cdiag"]
        sigma = np.asarray(subset.get("profiled_sigma_per_sqrt_event", [np.nan] * 3))
        gains = baseline_sigma / sigma - 1.0
        average_variance = float(np.mean((sigma / baseline_sigma) ** 2))
        rows.append([
            f"{100 * gains[0]:+.2f}%", f"{100 * gains[1]:+.2f}%", f"{100 * gains[2]:+.2f}%",
            f"{average_variance:.4f}", f"{subset['condition_number']:.4g}",
            f"{results[name]['valid_efficiency']:.3f}", f"{np.degrees(results[name]['tau_axis_error']):.3f}",
        ])
        labels.append(name.replace("_", " "))
    fig, axis = plt.subplots(figsize=(15.5, 4.2))
    axis.axis("off")
    table = axis.table(cellText=rows, rowLabels=labels, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    axis.set_title("Spin-matrix summary: jointly profiled Cdiag precision", pad=20)
    fig.tight_layout()
    fig.savefig(output / "spin_matrix_summary.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)


def _load_multi_histories(spin: dict[str, Any], output: Path) -> dict[str, list[dict[str, Any]]]:
    histories = {}
    for target_name in spin["multi_training"]["enabled_targets"]:
        path = output / f"multi_training_{target_name}.json"
        with path.open(encoding="utf-8") as stream:
            histories[target_name] = json.load(stream)["rounds"]
    return histories


def run_spin_matrix(
    spin_config_path: str | Path,
    mode: str,
    device_override: str | None = None,
    output_override: str | None = None,
    cnn_output_override: str | None = None,
) -> None:
    spin, base = _load_spin_config(spin_config_path)
    if device_override is not None:
        base["device"] = device_override
    if output_override is not None:
        spin["output_dir"] = output_override
    if cnn_output_override is not None:
        spin["cnn_output_dir"] = cnn_output_override
    seed_everything(int(base["seed"]))
    device = resolve_device(str(base.get("device", "auto")))
    output = Path(spin["output_dir"]).expanduser().resolve()
    (output / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (output / "resolved_spin_matrix_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(spin, stream, sort_keys=False)
    with (output / "resolved_base_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(base, stream, sort_keys=False)
    print(f"Using device: {device}", flush=True)
    policies = _load_cnn_policies(spin, base, device)
    histories: dict[str, list[dict[str, Any]]] = {}
    if mode in {"spin-run", "spin-train"}:
        for target_name in spin["multi_training"]["enabled_targets"]:
            parameters = list(spin["multi_training"]["target_sets"][target_name])
            policy, rows = train_multi_measurement_policy(
                policies["baseline"], target_name, parameters, spin, base, device, output,
            )
            histories[target_name] = rows
            policies[f"multi_{target_name}"] = policy
        _plot_multi_training(histories, spin, output)
    elif mode == "spin-evaluate":
        histories = _load_multi_histories(spin, output)
        _plot_multi_training(histories, spin, output)
        primary = str(spin["multi_training"]["primary_target"])
        for target_name in spin["multi_training"]["enabled_targets"]:
            policy_name = _multi_policy_name(str(target_name), primary)
            policies[f"multi_{target_name}"] = _load_flow(
                output / "checkpoints" / policy_name / "best_validation_policy.pt", base, device,
            )
    if mode in {"spin-run", "spin-evaluate", "spin-passive"}:
        results, _, reconstructions = run_passive_study(policies, spin, base, device, output)
        if "multi_Cdiag" in results:
            _plot_multi_tradeoff(results, spin, output)
            _plot_spin_summary(results, spin, output)
            evaluation_events = generate_events(
                int(spin["study"]["evaluation_events"]), float(base["physics"]["nominal_C"]),
                base, device,
                make_generator(device, int(base["seed"]) + int(spin["seed_offsets"]["passive_evaluation"])),
            )
            _plot_cnn_vs_multitask_response(evaluation_events, reconstructions, spin, output)
    print(f"Spin-matrix results written to {output}", flush=True)
