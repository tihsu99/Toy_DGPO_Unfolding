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
DERIVED_PARAMETER_NAMES = ("B_plus_n", "B_minus_n")
TRAINING_PARAMETER_NAMES = SPIN_PARAMETER_NAMES + DERIVED_PARAMETER_NAMES


def _parameter_projection(source_names: list[str], target_names: list[str]) -> np.ndarray:
    projection = np.zeros((len(target_names), len(source_names)), dtype=np.float64)
    for row, name in enumerate(target_names):
        if name in source_names:
            projection[row, source_names.index(name)] = 1.0
        elif name in DERIVED_PARAMETER_NAMES:
            scale = 1.0 / np.sqrt(2.0)
            projection[row, source_names.index("B_A_n")] = scale
            projection[row, source_names.index("B_B_n")] = scale if name == "B_plus_n" else -scale
        else:
            raise ValueError(f"Unknown spin parameter: {name}")
    return projection


def _project_scores(scores: torch.Tensor, source_names: list[str], target_names: list[str]) -> torch.Tensor:
    projection = torch.as_tensor(
        _parameter_projection(source_names, target_names), dtype=scores.dtype, device=scores.device,
    )
    return scores @ projection.T


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
    return _project_scores(truth_spin_scores(events, nominal_C), list(SPIN_PARAMETER_NAMES), names)


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
        unknown = sorted(set(parameter_names).difference(TRAINING_PARAMETER_NAMES))
        if unknown:
            raise ValueError(f"Unknown parameters in target set {target_name}: {unknown}")
    enabled = list(spin["multi_training"]["enabled_targets"])
    if spin["multi_training"]["primary_target"] not in enabled:
        raise ValueError("multi_training.primary_target must be enabled")
    if any(name not in targets for name in enabled):
        raise ValueError("Every enabled multi-measurement target must be defined")
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
    refresh_modes = list(spin["multi_training"].get("refresh_modes", ["fixed"]))
    if not refresh_modes or set(refresh_modes).difference({"fixed", "adaptive"}):
        raise ValueError("multi_training.refresh_modes may contain only fixed and adaptive")
    primary_refresh = str(spin["multi_training"].get("primary_refresh_mode", refresh_modes[0]))
    if primary_refresh not in refresh_modes:
        raise ValueError("multi_training.primary_refresh_mode must be enabled")
    if "adaptive" in refresh_modes:
        required_adaptive = {
            "monitor_events", "direct_validation_events", "direct_fisher_bins_per_dimension",
            "min_epochs_between_refresh", "max_epochs_between_refresh", "nmse_ratio_warning",
            "fisher_closure_log_threshold", "warm_start_check_interval",
        }
        missing_adaptive = required_adaptive.difference(spin["multi_training"])
        if missing_adaptive:
            raise ValueError(f"Missing adaptive-refresh settings: {sorted(missing_adaptive)}")
        minimum = int(spin["multi_training"]["min_epochs_between_refresh"])
        maximum = int(spin["multi_training"]["max_epochs_between_refresh"])
        if not 1 <= minimum <= maximum:
            raise ValueError("Adaptive refresh epochs must satisfy 1 <= min <= max")
        if int(spin["multi_training"]["direct_fisher_bins_per_dimension"]) < 2:
            raise ValueError("direct_fisher_bins_per_dimension must be at least two")
        if float(spin["multi_training"]["nmse_ratio_warning"]) <= 1.0:
            raise ValueError("nmse_ratio_warning must exceed one")
        if float(spin["multi_training"]["fisher_closure_log_threshold"]) <= 0.0:
            raise ValueError("fisher_closure_log_threshold must be positive")
        if any(int(spin["multi_training"][key]) < 1 for key in ("monitor_events", "direct_validation_events")):
            raise ValueError("Adaptive monitor and direct-validation samples must be non-empty")
        required_offsets = {
            "multi_monitor", "multi_direct_validation", "multi_monitor_reconstruction",
            "multi_direct_reconstruction",
        }
        missing_offsets = required_offsets.difference(spin["seed_offsets"])
        if missing_offsets:
            raise ValueError(f"Missing adaptive-refresh seed offsets: {sorted(missing_offsets)}")
    return spin, base


def _load_flow(path: Path, base: dict[str, Any], device: torch.device) -> ConditionalFlow:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("method_version") not in {1, 2, 3}:
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
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
    initial_state_dict: dict[str, torch.Tensor] | None = None,
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
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
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
    symmetric = 0.5 * (np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = eigenvalues[eigenvalues > 0.0]
    condition = float(eigenvalues[-1] / positive[0]) if positive.size == eigenvalues.size else float("inf")
    sign, log_determinant = np.linalg.slogdet(symmetric)
    return {
        "eigenvalues": eigenvalues.tolist(),
        "eigenvectors": eigenvectors.tolist(),
        "smallest_eigenvalue": float(eigenvalues[0]),
        "largest_eigenvalue": float(eigenvalues[-1]),
        "condition_number": condition,
        "log_determinant": float(log_determinant) if sign > 0 else float("-inf"),
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
    if not stable:
        return None, stats
    cholesky = np.linalg.cholesky(0.5 * (matrix + matrix.T))
    identity = np.eye(matrix.shape[0], dtype=np.float64)
    return np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, identity)), stats


def _subset_result(
    fisher: np.ndarray, parameter_names: list[str], subset: list[str], settings: dict[str, Any],
) -> dict[str, Any]:
    projection = _parameter_projection(parameter_names, subset)
    selected = projection @ fisher @ projection.T
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
    bc5_parameters = ["C_nn", "C_rr", "C_kk", "B_plus_n", "B_minus_n"]
    direct_bc5_fisher = direct_binned_fisher_per_generated_event(
        events, reconstructed, bc5_parameters, float(base["physics"]["nominal_C"]),
        int(spin["multi_training"].get("direct_fisher_bins_per_dimension", 4)),
    )
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
            fisher, list(SPIN_PARAMETER_NAMES), bc5_parameters, spin["study"],
        ),
        "BC5_direct": _subset_result(
            direct_bc5_fisher, bc5_parameters, bc5_parameters, spin["study"],
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
    fisher: torch.Tensor, baseline_fisher: torch.Tensor, lambda_num: float,
    return_ridge_fraction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    dimension = fisher.shape[-1]
    scale = torch.diagonal(fisher, dim1=-2, dim2=-1).mean(dim=-1)
    ridge = lambda_num * scale
    identity = torch.eye(dimension, dtype=fisher.dtype, device=fisher.device)
    cholesky, info = torch.linalg.cholesky_ex(fisher)
    needs_ridge = info != 0
    if torch.any(needs_ridge):
        applied_ridge = torch.where(needs_ridge, ridge, torch.zeros_like(ridge))
        regularized = fisher + applied_ridge[..., None, None] * identity
        cholesky, info = torch.linalg.cholesky_ex(regularized)
    if torch.any(info != 0):
        raise RuntimeError("Fisher matrix is not positive definite under the configured numerical ridge")
    if baseline_fisher.ndim == 1:
        covariance = torch.cholesky_inverse(cholesky)
        objective = (
            torch.diagonal(covariance, dim1=-2, dim2=-1) / baseline_fisher
        ).mean(dim=-1)
    else:
        solved = torch.cholesky_solve(baseline_fisher.expand(fisher.shape), cholesky)
        objective = torch.diagonal(solved, dim1=-2, dim2=-1).sum(dim=-1) / dimension
    if return_ridge_fraction:
        return objective, float(needs_ridge.to(torch.float64).mean())
    return objective


def _numpy_objectives(
    fisher: np.ndarray, baseline_fisher: np.ndarray, compute_d_optimal: bool,
) -> tuple[float, float | None, np.ndarray]:
    cholesky = np.linalg.cholesky(0.5 * (fisher + fisher.T))
    identity = np.eye(fisher.shape[0], dtype=np.float64)
    covariance = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, identity))
    if baseline_fisher.ndim == 1:
        a_objective = float(np.mean(np.diag(covariance) / baseline_fisher))
    else:
        a_objective = float(np.trace(baseline_fisher @ covariance) / fisher.shape[0])
    d_objective = None
    if compute_d_optimal:
        if baseline_fisher.ndim == 1:
            scales = np.sqrt(baseline_fisher)
            normalized_fisher = fisher * np.outer(scales, scales)
            sign, log_determinant = np.linalg.slogdet(normalized_fisher)
            d_objective = float(-log_determinant) if sign > 0 else float("inf")
        else:
            baseline_sign, baseline_logdet = np.linalg.slogdet(baseline_fisher)
            sign, log_determinant = np.linalg.slogdet(fisher)
            d_objective = (
                float(baseline_logdet - log_determinant)
                if baseline_sign > 0 and sign > 0 else float("inf")
            )
    return a_objective, d_objective, covariance


def _reconstructed_spin_observables(reconstructed: dict[str, torch.Tensor]) -> torch.Tensor:
    h_a = reconstructed["h_a_reco"]
    h_b = reconstructed["h_b_reco"]
    correlations = (h_a[..., :, None] * h_b[..., None, :]).reshape(*h_a.shape[:-1], 9)
    return torch.cat((h_a, h_b, correlations), dim=-1)


def _selected_reconstructed_observables(
    reconstructed: dict[str, torch.Tensor], parameter_names: list[str],
) -> torch.Tensor:
    return _project_scores(
        _reconstructed_spin_observables(reconstructed), list(SPIN_PARAMETER_NAMES), parameter_names,
    )


def direct_binned_fisher_per_generated_event(
    events: dict[str, torch.Tensor],
    reconstructed: dict[str, torch.Tensor],
    parameter_names: list[str],
    nominal_C: float,
    bins_per_dimension: int,
) -> np.ndarray:
    """Return a joint reconstructed-level Fisher from fixed multidimensional bins."""
    if bins_per_dimension < 2:
        raise ValueError("bins_per_dimension must be at least two")
    valid = reconstructed["valid"].detach().cpu().numpy().astype(bool)
    count = valid.size
    if not valid.any():
        raise RuntimeError("The policy produced no valid direct-validation events")
    features = _selected_reconstructed_observables(
        reconstructed, parameter_names,
    ).detach().cpu().numpy().astype(np.float64)[valid]
    truth = _selected_scores(events, nominal_C, parameter_names).detach().cpu().numpy().astype(np.float64)[valid]
    bin_indices = []
    for index, name in enumerate(parameter_names):
        limit = np.sqrt(2.0) if name in DERIVED_PARAMETER_NAMES else 1.0
        edges = np.linspace(-limit, limit, bins_per_dimension + 1)
        bin_indices.append(np.clip(np.searchsorted(edges, features[:, index], side="right") - 1, 0, bins_per_dimension - 1))
    flat_bins = np.ravel_multi_index(tuple(bin_indices), (bins_per_dimension,) * len(parameter_names))
    _, inverse, counts = np.unique(flat_bins, return_inverse=True, return_counts=True)
    score_sums = np.zeros((counts.size, len(parameter_names)), dtype=np.float64)
    np.add.at(score_sums, inverse, truth)
    return (score_sums.T @ (score_sums / counts[:, None])) / count


def _refresh_decision(
    epochs_since_refresh: int,
    nmse_ratio: float,
    closure_log_drift: float | None,
    settings: dict[str, Any],
) -> tuple[bool, str | None, bool]:
    if epochs_since_refresh >= int(settings["max_epochs_between_refresh"]):
        return True, "maximum_interval", False
    if epochs_since_refresh < int(settings["min_epochs_between_refresh"]):
        return False, None, False
    threshold = float(settings["fisher_closure_log_threshold"])
    if closure_log_drift is not None and abs(closure_log_drift) > threshold:
        return True, "fisher_closure_drift", False
    if nmse_ratio > float(settings["nmse_ratio_warning"]):
        if closure_log_drift is None:
            return False, None, True
        if abs(closure_log_drift) > threshold:
            return True, "nmse_warning_confirmed_by_fisher", False
    return False, None, False


def _multi_policy_name(
    target_name: str, primary_target: str, refresh_mode: str = "fixed", explicit_modes: bool = False,
) -> str:
    if explicit_modes:
        return f"{refresh_mode}_refresh_{target_name}_no_trust"
    return (
        "iterative_refresh_multi_no_trust"
        if target_name == primary_target else f"iterative_refresh_multi_no_trust_{target_name}"
    )


def _save_multi_policy(
    policy: ConditionalFlow, path: Path, target_name: str, round_index: int,
    refresh_mode: str = "fixed", epoch: int | None = None,
) -> None:
    legacy = refresh_mode == "fixed" and epoch is None
    torch.save({
        "method_version": 1 if legacy else 2,
        "policy": "iterative_refresh_multi_no_trust" if legacy else f"{refresh_mode}_refresh_multi_no_trust",
        "target_set": target_name, "refresh_mode": refresh_mode, "round": round_index,
        "epoch": epoch, "state_dict": policy.state_dict(),
    }, path)


def _train_multi_local_update(
    current: ConditionalFlow,
    active_score: VectorScoreModel,
    events: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    fisher_reference: torch.Tensor,
    baseline_fisher: torch.Tensor,
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    seed: int,
    description: str,
    epochs: int | None = None,
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
    reference_objective, reference_ridge_fraction = _torch_a_objective(
        fisher_reference, baseline_fisher, lambda_num, return_ridge_fraction=True,
    )
    reference_objective = reference_objective.detach()
    generator = make_generator(device, seed)
    history: list[dict[str, float]] = []
    progress = tqdm(
        range(int(settings["dgpo_epochs_per_round"]) if epochs is None else epochs),
        desc=description, unit="epoch",
    )
    for epoch in progress:
        permutation = torch.randperm(events["x"].numel(), device=device, generator=generator)
        reward_total = 0.0
        invalid_total = 0.0
        ridge_fraction_total = 0.0
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
                candidate_objective, candidate_ridge_fraction = _torch_a_objective(
                    candidate_fisher, baseline_fisher, lambda_num, return_ridge_fraction=True,
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
            ridge_fraction_total += candidate_ridge_fraction * count
            seen += count
        row = {
            "local_epoch": float(epoch + 1),
            "reward": reward_total / seen,
            "invalid_fraction": invalid_total / seen,
            "candidate_ridge_fraction": ridge_fraction_total / seen,
            "reference_ridge_fraction": reference_ridge_fraction,
        }
        history.append(row)
        progress.set_postfix(reward=f"{row['reward']:.4g}", invalid=f"{row['invalid_fraction']:.3f}")
    return policy.eval(), history


def _train_legacy_multi_measurement_policy(
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


@torch.no_grad()
def _score_nmse(
    policy: ConditionalFlow,
    score_model: VectorScoreModel,
    events: dict[str, torch.Tensor],
    parameter_names: list[str],
    base: dict[str, Any],
    seed: int,
) -> tuple[float, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    reconstructed = reconstruct_policy(
        policy, events, base, make_generator(events["x"].device, seed),
    )
    prediction = _score_reconstruction(score_model, reconstructed)
    target = _selected_scores(events, float(base["physics"]["nominal_C"]), parameter_names).to(torch.float64)
    mse = (prediction - target).square().sum(dim=-1).mean()
    centered = target - target.mean(dim=0, keepdim=True)
    denominator = centered.square().sum(dim=-1).mean().clamp_min(torch.finfo(torch.float64).eps)
    return float(mse / denominator), reconstructed, prediction, target


@torch.no_grad()
def _measurement_validation(
    policy: ConditionalFlow,
    score_model: VectorScoreModel,
    events: dict[str, torch.Tensor],
    parameter_names: list[str],
    baseline_score_fisher: np.ndarray,
    baseline_direct_fisher: np.ndarray,
    spin: dict[str, Any],
    base: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    settings = spin["multi_training"]
    reconstructed = reconstruct_policy(
        policy, events, base, make_generator(events["x"].device, seed),
    )
    score_fisher = fisher_matrix_per_generated_event(
        _score_reconstruction(score_model, reconstructed),
    ).cpu().numpy()
    direct_fisher = direct_binned_fisher_per_generated_event(
        events, reconstructed, parameter_names, float(base["physics"]["nominal_C"]),
        int(settings["direct_fisher_bins_per_dimension"]),
    )
    _, score_stats = _stable_inverse(
        score_fisher, float(settings["condition_number_max"]),
        float(settings.get("eigenvalue_relative_tolerance", 0.0)),
    )
    _, direct_stats = _stable_inverse(
        direct_fisher, float(settings["condition_number_max"]),
        float(settings.get("eigenvalue_relative_tolerance", 0.0)),
    )
    if not score_stats["stable"] or not direct_stats["stable"]:
        raise RuntimeError(
            f"Unstable validation Fisher: score={score_stats}, direct={direct_stats}"
        )
    score_objective, _, score_covariance = _numpy_objectives(
        score_fisher, baseline_score_fisher, False,
    )
    direct_objective, _, direct_covariance = _numpy_objectives(
        direct_fisher, baseline_direct_fisher, False,
    )
    cosine = (reconstructed["k_a"] * events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        "J_score": score_objective,
        "J_direct": direct_objective,
        "closure_ratio": score_objective / direct_objective,
        "score_fisher_per_generated_event": score_fisher.tolist(),
        "direct_fisher_per_generated_event": direct_fisher.tolist(),
        "score_profiled_sigma_per_sqrt_event": np.sqrt(np.diag(score_covariance)).tolist(),
        "direct_profiled_sigma_per_sqrt_event": np.sqrt(np.diag(direct_covariance)).tolist(),
        "score_fisher_stats": score_stats,
        "direct_fisher_stats": direct_stats,
        "valid_efficiency": float(reconstructed["valid"].float().mean()),
        "tau_axis_error": float(torch.acos(cosine).mean()),
    }


def _score_curve_signature(
    prediction: torch.Tensor,
    reconstructed: dict[str, torch.Tensor],
    parameter_names: list[str],
    bins: int,
) -> list[list[float]]:
    features = _selected_reconstructed_observables(
        reconstructed, parameter_names,
    ).detach().cpu().numpy()
    values = prediction.detach().cpu().numpy()
    valid = reconstructed["valid"].detach().cpu().numpy().astype(bool)
    curves: list[list[float]] = []
    for index, name in enumerate(parameter_names):
        limit = np.sqrt(2.0) if name in DERIVED_PARAMETER_NAMES else 1.0
        edges = np.linspace(-limit, limit, bins + 1)
        assigned = np.clip(np.searchsorted(edges, features[:, index], side="right") - 1, 0, bins - 1)
        counts = np.bincount(assigned[valid], minlength=bins)
        sums = np.bincount(assigned[valid], weights=values[valid, index], minlength=bins)
        curves.append((sums / np.clip(counts, 1, None)).tolist())
    return curves


def _warm_start_diagnostic(
    policy: ConditionalFlow,
    warm_model: VectorScoreModel,
    fresh_model: VectorScoreModel,
    monitor_events: dict[str, torch.Tensor],
    closure_events: dict[str, torch.Tensor],
    parameter_names: list[str],
    baseline_score_fisher: np.ndarray,
    baseline_direct_fisher: np.ndarray,
    spin: dict[str, Any],
    base: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    warm_nmse, reconstructed, warm_prediction, _ = _score_nmse(
        policy, warm_model, monitor_events, parameter_names, base, seed,
    )
    fresh_nmse, _, fresh_prediction, _ = _score_nmse(
        policy, fresh_model, monitor_events, parameter_names, base, seed,
    )
    warm_validation = _measurement_validation(
        policy, warm_model, closure_events, parameter_names, baseline_score_fisher,
        baseline_direct_fisher, spin, base, seed + 1,
    )
    fresh_validation = _measurement_validation(
        policy, fresh_model, closure_events, parameter_names, baseline_score_fisher,
        baseline_direct_fisher, spin, base, seed + 1,
    )
    bins = int(spin["multi_training"].get("score_curve_bins", 20))
    warm_curves = _score_curve_signature(warm_prediction, reconstructed, parameter_names, bins)
    fresh_curves = _score_curve_signature(fresh_prediction, reconstructed, parameter_names, bins)
    curve_scale = np.sqrt(np.mean(np.square(fresh_prediction.detach().cpu().numpy())))
    curve_rms = float(np.sqrt(np.mean((np.asarray(warm_curves) - np.asarray(fresh_curves)) ** 2)))
    relative_curve_rms = curve_rms / max(float(curve_scale), np.finfo(float).eps)
    mse_ratio = warm_nmse / max(fresh_nmse, np.finfo(float).eps)
    closure_log_difference = float(np.log(
        warm_validation["closure_ratio"] / fresh_validation["closure_ratio"]
    ))
    settings = spin["multi_training"]
    disagrees = (
        abs(np.log(mse_ratio)) > float(settings.get("warm_start_mse_log_tolerance", 0.10))
        or abs(closure_log_difference) > float(settings.get("warm_start_closure_log_tolerance", 0.05))
        or relative_curve_rms > float(settings.get("warm_start_curve_rms_tolerance", 0.10))
    )
    return {
        "warm_nmse": warm_nmse,
        "fresh_nmse": fresh_nmse,
        "warm_to_fresh_nmse_ratio": mse_ratio,
        "warm_closure_ratio": warm_validation["closure_ratio"],
        "fresh_closure_ratio": fresh_validation["closure_ratio"],
        "closure_log_difference": closure_log_difference,
        "score_curve_relative_rms": relative_curve_rms,
        "score_curves_warm": warm_curves,
        "score_curves_fresh": fresh_curves,
        "significant_disagreement": bool(disagrees),
    }


def _train_controlled_multi_measurement_policy(
    baseline: ConditionalFlow,
    target_name: str,
    parameter_names: list[str],
    refresh_mode: str,
    spin: dict[str, Any],
    base: dict[str, Any],
    device: torch.device,
    output: Path,
) -> tuple[ConditionalFlow, list[dict[str, Any]]]:
    settings = spin["multi_training"]
    offsets = spin["seed_offsets"]
    nominal_C = float(base["physics"]["nominal_C"])
    explicit_modes = "refresh_modes" in settings
    policy_name = _multi_policy_name(
        target_name, str(settings["primary_target"]), refresh_mode, explicit_modes,
    )
    checkpoint_dir = output / "checkpoints" / policy_name
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()) and not bool(settings.get("allow_overwrite", False)):
        raise FileExistsError(
            f"Refusing to overwrite existing {target_name} {refresh_mode} checkpoints: {checkpoint_dir}"
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mode_seed = 0 if refresh_mode == "fixed" else 500_000
    monitor_events = generate_events(
        int(settings["monitor_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["multi_monitor"])),
    )
    closure_events = generate_events(
        int(settings["direct_validation_events"]), nominal_C, base, device,
        make_generator(device, int(base["seed"]) + int(offsets["multi_direct_validation"])),
    )
    current = copy.deepcopy(baseline).eval()
    active_score: VectorScoreModel | None = None
    score_training_epochs = 0
    diagnostic_score_training_epochs = 0
    warm_diagnostics: list[dict[str, Any]] = []

    def train_score(refresh_index: int) -> VectorScoreModel:
        nonlocal score_training_epochs, diagnostic_score_training_epochs
        score_seed = int(base["seed"]) + int(offsets["multi_score"]) + 1000 * refresh_index
        events = generate_events(
            int(settings["score_events"]), nominal_C, base, device,
            make_generator(device, score_seed),
        )
        warm_state = active_score.state_dict() if active_score is not None and bool(settings.get("score_warm_start", True)) else None
        model, _ = _train_vector_score(
            current, events, parameter_names, spin, base, device, score_seed,
            f"{target_name} {refresh_mode} score {refresh_index}", settings, warm_state,
        )
        score_training_epochs += int(settings["score_epochs"])
        interval = int(settings["warm_start_check_interval"])
        if warm_state is not None and interval > 0 and refresh_index % interval == 0:
            fresh_model, _ = _train_vector_score(
                current, events, parameter_names, spin, base, device, score_seed + 700_000,
                f"{target_name} fresh-init diagnostic {refresh_index}", settings,
            )
            diagnostic_score_training_epochs += int(settings["score_epochs"])
            diagnostic = _warm_start_diagnostic(
                current, model, fresh_model, monitor_events, closure_events, parameter_names,
                baseline_score_fisher, baseline_direct_fisher, spin, base,
                int(base["seed"]) + int(offsets["multi_monitor_reconstruction"]),
            )
            diagnostic["refresh"] = refresh_index
            warm_diagnostics.append(diagnostic)
            if diagnostic["significant_disagreement"]:
                print(f"WARNING: {target_name} warm/fresh score disagreement at refresh {refresh_index}", flush=True)
        return model

    active_score = train_score(0)
    monitor_seed = int(base["seed"]) + int(offsets["multi_monitor_reconstruction"])
    closure_seed = int(base["seed"]) + int(offsets["multi_direct_reconstruction"])
    initial_nmse, _, _, _ = _score_nmse(
        current, active_score, monitor_events, parameter_names, base, monitor_seed,
    )
    initial_reconstruction = reconstruct_policy(
        current, closure_events, base, make_generator(device, closure_seed),
    )
    baseline_score_fisher = fisher_matrix_per_generated_event(
        _score_reconstruction(active_score, initial_reconstruction),
    ).cpu().numpy()
    baseline_direct_fisher = direct_binned_fisher_per_generated_event(
        closure_events, initial_reconstruction, parameter_names, nominal_C,
        int(settings["direct_fisher_bins_per_dimension"]),
    )
    baseline_score_covariance, baseline_score_stats = _stable_inverse(
        baseline_score_fisher, float(settings["condition_number_max"]),
        float(settings.get("eigenvalue_relative_tolerance", 0.0)),
    )
    baseline_direct_covariance, baseline_direct_stats = _stable_inverse(
        baseline_direct_fisher, float(settings["condition_number_max"]),
        float(settings.get("eigenvalue_relative_tolerance", 0.0)),
    )
    if baseline_score_covariance is None or baseline_direct_covariance is None:
        raise RuntimeError(
            f"Baseline {target_name} Fisher is unstable: score={baseline_score_stats}, direct={baseline_direct_stats}"
        )
    baseline_diagnostics = {
        "parameters": parameter_names,
        "score_fisher_per_generated_event": baseline_score_fisher.tolist(),
        "direct_fisher_per_generated_event": baseline_direct_fisher.tolist(),
        "score_profiled_sigma_per_sqrt_event": np.sqrt(np.diag(baseline_score_covariance)).tolist(),
        "direct_profiled_sigma_per_sqrt_event": np.sqrt(np.diag(baseline_direct_covariance)).tolist(),
        "score_stats": baseline_score_stats,
        "direct_stats": baseline_direct_stats,
        "sigma_B_plus_n": float(np.sqrt(np.diag(baseline_direct_covariance))[parameter_names.index("B_plus_n")]) if "B_plus_n" in parameter_names else None,
        "sigma_B_minus_n": float(np.sqrt(np.diag(baseline_direct_covariance))[parameter_names.index("B_minus_n")]) if "B_minus_n" in parameter_names else None,
    }
    with (output / f"baseline_fisher_{target_name}_{refresh_mode}.json").open("w", encoding="utf-8") as stream:
        json.dump(baseline_diagnostics, stream, indent=2)

    best_policy = copy.deepcopy(current).eval()
    best_objective = 1.0
    best_epoch = 0
    best_refresh = 0
    patience_reference = 1.0
    checks_without_improvement = 0
    rows: list[dict[str, Any]] = []
    refresh_count = 0
    total_epoch = 0
    max_epochs = int(settings.get(
        "max_dgpo_epochs",
        int(settings["max_refresh_rounds"]) * int(settings["dgpo_epochs_per_round"]),
    ))
    _save_multi_policy(current, checkpoint_dir / "pi_00_policy.pt", target_name, 0, refresh_mode, 0)
    torch.save({
        "method_version": 2, "target_set": target_name, "parameters": parameter_names,
        "refresh_mode": refresh_mode, "refresh": 0, "state_dict": active_score.state_dict(),
    }, checkpoint_dir / "pi_00_score.pt")

    round_start_nmse = initial_nmse
    round_start_validation = _measurement_validation(
        current, active_score, closure_events, parameter_names, baseline_score_fisher,
        baseline_direct_fisher, spin, base, closure_seed,
    )
    round_start_closure = float(round_start_validation["closure_ratio"])
    stop_reason = "maximum_dgpo_epochs"
    while total_epoch < max_epochs:
        training_seed = int(base["seed"]) + int(offsets["multi_training"]) + mode_seed + 10000 * refresh_count
        training_events = generate_events(
            int(settings["training_events"]), nominal_C, base, device,
            make_generator(device, training_seed),
        )
        reference = reconstruct_policy(
            current, training_events, base,
            make_generator(device, int(base["seed"]) + int(offsets["multi_reconstruction"]) + mode_seed + 10000 * refresh_count),
        )
        reference["vector_score"] = _score_reconstruction(active_score, reference)
        fisher_reference = (reference["vector_score"].T @ reference["vector_score"]).to(torch.float64)
        fisher_reference_per_event = (fisher_reference / training_events["x"].numel()).cpu().numpy()
        _, reference_stats = _stable_inverse(
            fisher_reference_per_event, float(settings["condition_number_max"]),
            float(settings.get("eigenvalue_relative_tolerance", 0.0)),
        )
        if not reference_stats["stable"]:
            stop_reason = "ill_conditioned_round_reference"
            break
        baseline_training_fisher = torch.as_tensor(
            baseline_score_fisher * training_events["x"].numel(), device=device, dtype=torch.float64,
        )
        round_start_objective, _, _ = _numpy_objectives(
            fisher_reference_per_event, baseline_score_fisher, False,
        )
        epochs_since_refresh = 0
        refresh_triggered = False
        while total_epoch < max_epochs:
            next_policy, local_history = _train_multi_local_update(
                current, active_score, training_events, reference, fisher_reference,
                baseline_training_fisher, spin, base, device,
                training_seed + 20000 + total_epoch,
                f"{policy_name} epoch {total_epoch + 1}", epochs=1,
            )
            current = next_policy
            total_epoch += 1
            epochs_since_refresh += 1
            nmse, _, _, _ = _score_nmse(
                current, active_score, monitor_events, parameter_names, base, monitor_seed,
            )
            nmse_ratio = nmse / max(round_start_nmse, np.finfo(float).eps)
            validation_interval = int(settings.get("direct_validation_interval_epochs", 1))
            check_validation = total_epoch % validation_interval == 0 or nmse_ratio > float(settings["nmse_ratio_warning"])
            validation = None
            closure_drift = None
            if check_validation:
                validation = _measurement_validation(
                    current, active_score, closure_events, parameter_names, baseline_score_fisher,
                    baseline_direct_fisher, spin, base, closure_seed,
                )
                closure_drift = float(np.log(validation["closure_ratio"] / round_start_closure))
            if refresh_mode == "adaptive":
                refresh_triggered, trigger_reason, needs_check = _refresh_decision(
                    epochs_since_refresh, nmse_ratio, closure_drift, settings,
                )
                if needs_check:
                    validation = _measurement_validation(
                        current, active_score, closure_events, parameter_names,
                        baseline_score_fisher, baseline_direct_fisher, spin, base, closure_seed,
                    )
                    closure_drift = float(np.log(validation["closure_ratio"] / round_start_closure))
                    refresh_triggered, trigger_reason, _ = _refresh_decision(
                        epochs_since_refresh, nmse_ratio, closure_drift, settings,
                    )
            else:
                refresh_triggered = epochs_since_refresh >= int(settings.get(
                    "fixed_refresh_epochs", settings["dgpo_epochs_per_round"],
                ))
                trigger_reason = "fixed_interval" if refresh_triggered else None
            row: dict[str, Any] = {
                "round": refresh_count,
                "total_dgpo_epoch": total_epoch,
                "local_epoch": epochs_since_refresh,
                "status": "ok",
                "refresh_mode": refresh_mode,
                "refresh_triggered": bool(refresh_triggered and total_epoch < max_epochs),
                "refresh_reason": trigger_reason if total_epoch < max_epochs else None,
                "NMSE_current": nmse,
                "NMSE_start": round_start_nmse,
                "r_NMSE": nmse_ratio,
                "closure_start": round_start_closure,
                "delta_closure": closure_drift,
                "J_train_before_update": round_start_objective,
                "mean_reward": local_history[0]["reward"],
                "invalid_fraction_training": local_history[0]["invalid_fraction"],
                "candidate_ridge_fraction": local_history[0]["candidate_ridge_fraction"],
                "reference_ridge_fraction": local_history[0]["reference_ridge_fraction"],
                "score_refreshes": refresh_count,
                "score_training_epochs": score_training_epochs,
                "diagnostic_score_training_epochs": diagnostic_score_training_epochs,
                "configured_ridge_relative_to_mean_eigenvalue": float(settings["lambda_num"]),
                "configured_ridge_absolute_per_event": float(settings["lambda_num"] * np.trace(fisher_reference_per_event) / len(parameter_names)),
                "configured_ridge_to_largest_eigenvalue": float(settings["lambda_num"] * np.trace(fisher_reference_per_event) / len(parameter_names) / reference_stats["largest_eigenvalue"]),
                "reference_fisher_stats": reference_stats,
            }
            if validation is not None:
                row.update(validation)
                objective = float(validation["J_direct"])
                if objective < best_objective:
                    best_objective = objective
                    best_epoch = total_epoch
                    best_refresh = refresh_count
                    best_policy = copy.deepcopy(current).eval()
                improvement_threshold = patience_reference * (
                    1.0 - float(settings["min_relative_objective_improvement"])
                )
                if objective < improvement_threshold:
                    patience_reference = objective
                    checks_without_improvement = 0
                else:
                    checks_without_improvement += 1
            rows.append(row)
            if checks_without_improvement >= int(settings.get(
                "early_stop_patience_checks", settings["early_stop_patience_rounds"],
            )):
                stop_reason = "direct_validation_early_stopping"
                refresh_triggered = False
                break
            if refresh_triggered or total_epoch >= max_epochs:
                break
        if stop_reason == "direct_validation_early_stopping" or total_epoch >= max_epochs:
            break
        refresh_count += 1
        _save_multi_policy(
            current, checkpoint_dir / f"pi_{refresh_count:02d}_policy.pt",
            target_name, refresh_count, refresh_mode, total_epoch,
        )
        active_score = train_score(refresh_count)
        torch.save({
            "method_version": 2, "target_set": target_name, "parameters": parameter_names,
            "refresh_mode": refresh_mode, "refresh": refresh_count,
            "epoch": total_epoch, "state_dict": active_score.state_dict(),
        }, checkpoint_dir / f"pi_{refresh_count:02d}_score.pt")
        round_start_nmse, _, _, _ = _score_nmse(
            current, active_score, monitor_events, parameter_names, base, monitor_seed,
        )
        round_start_validation = _measurement_validation(
            current, active_score, closure_events, parameter_names, baseline_score_fisher,
            baseline_direct_fisher, spin, base, closure_seed,
        )
        round_start_closure = float(round_start_validation["closure_ratio"])
        print(
            f"Refresh {refresh_count}: min={round_start_validation['score_fisher_stats']['smallest_eigenvalue']:.4g} "
            f"max={round_start_validation['score_fisher_stats']['largest_eigenvalue']:.4g} "
            f"cond={round_start_validation['score_fisher_stats']['condition_number']:.4g} "
            f"logdet={round_start_validation['score_fisher_stats']['log_determinant']:.4g} "
            f"J={round_start_validation['J_score']:.4g}", flush=True,
        )

    final_refresh = refresh_count
    _save_multi_policy(current, checkpoint_dir / "final_policy.pt", target_name, final_refresh, refresh_mode, total_epoch)
    _save_multi_policy(best_policy, checkpoint_dir / "best_validation_policy.pt", target_name, best_refresh, refresh_mode, best_epoch)
    _save_multi_policy(best_policy, output / "checkpoints" / f"{policy_name}.pt", target_name, best_refresh, refresh_mode, best_epoch)
    summary = {
        "policy": policy_name,
        "target_set": target_name,
        "parameters": parameter_names,
        "refresh_mode": refresh_mode,
        "selection_metric": "minimum_independent_direct_validation_J",
        "best_epoch": best_epoch,
        "best_refresh": best_refresh,
        "best_J_direct_validation": best_objective,
        "final_epoch": total_epoch,
        "score_refreshes": refresh_count,
        "score_training_epochs": score_training_epochs,
        "diagnostic_score_training_epochs": diagnostic_score_training_epochs,
        "total_score_training_epochs": score_training_epochs + diagnostic_score_training_epochs,
        "stop_reason": stop_reason,
        "warm_start_diagnostics": warm_diagnostics,
        "epochs": rows,
    }
    with (output / f"multi_training_{target_name}_{refresh_mode}.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _write_rows(output / f"adaptive_refresh_history_{target_name}_{refresh_mode}", rows)
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


def _finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.9", linewidth=0.6)


def _bc5_measurement_result(result: dict[str, Any]) -> dict[str, Any]:
    return result.get("BC5_direct", result["BC5"])


def _plot_adaptive_refresh_diagnostics(
    rows: list[dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    if not rows:
        return
    settings = spin["multi_training"]
    epochs = np.asarray([row["total_dgpo_epoch"] for row in rows])
    refresh_epochs = [row["total_dgpo_epoch"] for row in rows if row["refresh_triggered"]]
    checked = [row for row in rows if row.get("J_direct") is not None]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(epochs, [row["r_NMSE"] for row in rows], color="#4C78A8", marker="o")
    axes[0, 0].axhline(float(settings["nmse_ratio_warning"]), color="#D55E00", linestyle="--")
    axes[0, 0].set(ylabel=r"$r_{\mathrm{NMSE}}$", title="Cheap score-regression warning")
    if checked:
        checked_epochs = [row["total_dgpo_epoch"] for row in checked]
        axes[0, 1].plot(checked_epochs, [row["delta_closure"] for row in checked], color="#CC79A7", marker="o")
        threshold = float(settings["fisher_closure_log_threshold"])
        axes[0, 1].axhline(threshold, color="#D55E00", linestyle="--")
        axes[0, 1].axhline(-threshold, color="#D55E00", linestyle="--")
        axes[1, 0].plot(checked_epochs, [row["J_direct"] for row in checked], color="#009E73", marker="o", label="Direct validation")
        axes[1, 0].plot(checked_epochs, [row["J_score"] for row in checked], color="#4C78A8", marker="s", label="Active surrogate")
        axes[1, 1].plot(checked_epochs, [row["closure_ratio"] for row in checked], color="#CC79A7", marker="o")
    axes[0, 1].set(ylabel=r"$\Delta\log(J_{score}/J_{direct})$", title="Relative Fisher-closure drift")
    axes[1, 0].set(ylabel="Baseline-whitened J", title="Measurement objective")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].axhline(1.0, color="0.35", linestyle="--")
    axes[1, 1].set(ylabel=r"$J_{score}/J_{direct}$", title="Direct-surrogate closure")
    for axis in axes.ravel():
        for epoch in refresh_epochs:
            axis.axvline(epoch, color="0.55", linewidth=0.8, alpha=0.65)
        axis.set_xlabel("DGPO epoch")
        _finish_axis(axis)
    fig.suptitle("Adaptive score refresh: warning, closure, and measurement validation")
    fig.savefig(output / "adaptive_refresh_diagnostics.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "adaptive_refresh_diagnostics", rows)


def _plot_fixed_vs_adaptive(
    histories: dict[str, list[dict[str, Any]]],
    baseline_sigma: np.ndarray,
    parameter_names: list[str],
    spin: dict[str, Any],
    output: Path,
) -> None:
    if not {"fixed", "adaptive"}.issubset(histories):
        return
    colors = {"fixed": "#7F7F7F", "adaptive": "#7B61A8"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    final_rows: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for mode in ("fixed", "adaptive"):
        rows = histories[mode]
        checked = [row for row in rows if row.get("J_direct") is not None]
        if not checked:
            continue
        epochs = [row["total_dgpo_epoch"] for row in checked]
        final_rows[mode] = checked[-1]
        axes[0, 0].plot(epochs, [row["J_direct"] for row in checked], color=colors[mode], marker="o", label=mode)
        axes[0, 1].plot(epochs, [row["closure_ratio"] for row in checked], color=colors[mode], marker="o", label=mode)
        all_epochs = [row["total_dgpo_epoch"] for row in rows]
        cumulative = np.cumsum([int(row["refresh_triggered"]) for row in rows])
        axes[1, 0].step(all_epochs, cumulative, where="post", color=colors[mode], label=mode)
        final = checked[-1]
        summary_rows.append({
            "refresh_mode": mode,
            "final_J_direct": final["J_direct"],
            "final_closure_ratio": final["closure_ratio"],
            "score_refreshes": int(sum(row["refresh_triggered"] for row in rows)),
            "score_training_epochs": final["score_training_epochs"],
            "final_profiled_sigma": final["direct_profiled_sigma_per_sqrt_event"],
        })
    positions = np.arange(len(parameter_names))
    width = 0.36
    for index, mode in enumerate(("fixed", "adaptive")):
        if mode not in final_rows:
            continue
        sigma = np.asarray(final_rows[mode]["direct_profiled_sigma_per_sqrt_event"])
        axes[1, 1].bar(positions + (index - 0.5) * width, sigma / baseline_sigma, width,
                       color=colors[mode], label=mode)
    axes[0, 0].set(ylabel="Direct validation J", title="Measurement objective")
    axes[0, 1].axhline(1.0, color="0.35", linestyle="--")
    axes[0, 1].set(ylabel=r"$J_{score}/J_{direct}$", title="Final Fisher closure")
    axes[1, 0].set(ylabel="Cumulative score refreshes", title="Score-training cost")
    axes[1, 1].axhline(1.0, color="0.35", linestyle="--")
    axes[1, 1].set(xticks=positions, xticklabels=parameter_names, ylabel="Profiled sigma / baseline",
                   title="Final jointly profiled precision")
    axes[1, 1].tick_params(axis="x", rotation=30)
    for axis in axes.ravel():
        axis.set_xlabel("DGPO epoch" if axis is not axes[1, 1] else "")
        axis.legend(frameon=False)
        _finish_axis(axis)
    fig.suptitle("Fixed versus adaptive score refresh at matched DGPO budget")
    fig.savefig(output / "fixed_vs_adaptive_refresh.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    with (output / "fixed_vs_adaptive_refresh.json").open("w", encoding="utf-8") as stream:
        json.dump({"summary": summary_rows, "histories": histories}, stream, indent=2)


def _plot_bc5_profiled_precision(
    results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    order = [("baseline", "Baseline"), ("cnn_only", r"$C_{nn}$ only"),
             ("cdiag", "Cdiag"), ("bc5_adaptive", "BC5")]
    available = [(key, label) for key, label in order if key in results]
    if len(available) < 2:
        return
    parameters = ["C_nn", "C_rr", "C_kk", "B_plus_n", "B_minus_n"]
    baseline_sigma = np.asarray(_bc5_measurement_result(results["baseline"]).get("profiled_sigma_per_sqrt_event", []))
    if baseline_sigma.size != len(parameters):
        return
    positions = np.arange(len(parameters))
    width = 0.8 / len(available)
    colors = ["#B5B5B5", "#4C78A8", "#E69F00", "#7B61A8"]
    rows: list[dict[str, Any]] = []
    fig, axis = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
    for index, (key, label) in enumerate(available):
        sigma = np.asarray(_bc5_measurement_result(results[key]).get("profiled_sigma_per_sqrt_event", np.full(5, np.nan)))
        ratio = sigma / baseline_sigma
        axis.bar(positions + (index - (len(available) - 1) / 2) * width, ratio, width,
                 color=colors[index], label=label)
        rows.extend({"policy": key, "parameter": name, "profiled_sigma_ratio_to_baseline": float(ratio[item])}
                    for item, name in enumerate(parameters))
    axis.axhline(1.0, color="0.3", linestyle="--")
    axis.set(xticks=positions, xticklabels=parameters, ylabel="Jointly profiled sigma / baseline",
             title="Jointly profiled BC5 precision")
    axis.legend(frameon=False, ncol=len(available))
    _finish_axis(axis)
    fig.savefig(output / "BC5_profiled_precision.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "BC5_profiled_precision", rows)


def _match_eigenmodes(reference_vectors: np.ndarray, candidate_vectors: np.ndarray) -> list[int]:
    overlaps = np.abs(reference_vectors.T @ candidate_vectors)
    remaining = set(range(overlaps.shape[1]))
    matched = []
    for reference_index in range(overlaps.shape[0]):
        selected = max(remaining, key=lambda index: overlaps[reference_index, index])
        matched.append(selected)
        remaining.remove(selected)
    return matched


def _plot_bc5_eigenmodes(
    results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    policies = [("baseline", "Baseline"), ("cdiag", "Cdiag"), ("bc5_adaptive", "BC5")]
    policies = [(key, label) for key, label in policies if key in results]
    if len(policies) < 2:
        return
    baseline = np.asarray(_bc5_measurement_result(results["baseline"])["fisher"])
    baseline_values, baseline_vectors = np.linalg.eigh(0.5 * (baseline + baseline.T))
    polarization_weight = np.square(baseline_vectors[3:, :]).sum(axis=0)
    weak_polarization_mode = int(np.argmax(polarization_weight / np.clip(baseline_values, 1.0e-300, None)))
    positions = np.arange(baseline_values.size)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    rows: list[dict[str, Any]] = []
    for key, label in policies:
        fisher = np.asarray(_bc5_measurement_result(results[key])["fisher"])
        values, vectors = np.linalg.eigh(0.5 * (fisher + fisher.T))
        matched = _match_eigenmodes(baseline_vectors, vectors)
        values = values[matched]
        axes[0].plot(positions, values, marker="o", label=label)
        axes[1].plot(positions, values / baseline_values, marker="o", label=label)
        rows.extend({
            "policy": key, "baseline_mode": int(index), "matched_policy_mode": int(matched[index]),
            "eigenvalue": float(values[index]), "relative_eigenvalue_gain": float(values[index] / baseline_values[index]),
            "weak_polarization_mode": index == weak_polarization_mode,
        } for index in range(values.size))
    for axis in axes:
        axis.axvspan(weak_polarization_mode - 0.25, weak_polarization_mode + 0.25,
                     color="#CC79A7", alpha=0.18, label="Weak polarization mode")
        axis.set_xlabel("Baseline-matched Fisher eigenmode")
        axis.legend(frameon=False)
        _finish_axis(axis)
    axes[0].set_yscale("log")
    axes[0].set(ylabel="Fisher eigenvalue", title="Absolute eigenvalue spectrum")
    axes[1].axhline(1.0, color="0.3", linestyle="--")
    axes[1].set(ylabel="Eigenvalue / baseline", title="Matched eigenmode gain")
    fig.suptitle("BC5 Fisher eigenmodes and weak-polarization response")
    fig.savefig(output / "BC5_fisher_eigenmodes.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "BC5_fisher_eigenmodes", rows)


def _plot_full_spin_passive_transfer(
    results: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    policies = [("cnn_only", r"$C_{nn}$ only"), ("cdiag", "Cdiag"), ("bc5_adaptive", "BC5")]
    policies = [(key, label) for key, label in policies if key in results]
    if not policies:
        return
    baseline = np.asarray(results["baseline"]["conditional_sigma_per_sqrt_event"])
    positions = np.arange(len(SPIN_PARAMETER_NAMES))
    width = 0.8 / len(policies)
    fig, axis = plt.subplots(figsize=(13.0, 4.8), constrained_layout=True)
    rows: list[dict[str, Any]] = []
    for index, (key, label) in enumerate(policies):
        sigma = np.asarray(results[key]["conditional_sigma_per_sqrt_event"])
        ratio = sigma / baseline
        axis.bar(positions + (index - (len(policies) - 1) / 2) * width, ratio, width, label=label)
        rows.extend({"policy": key, "parameter": name, "conditional_sigma_ratio_to_baseline": float(ratio[item]),
                     "explicit_BC5_span": name in {"C_nn", "C_rr", "C_kk", "B_A_n", "B_B_n"}}
                    for item, name in enumerate(SPIN_PARAMETER_NAMES))
    for index, name in enumerate(SPIN_PARAMETER_NAMES):
        if name in {"C_nn", "C_rr", "C_kk", "B_A_n", "B_B_n"}:
            axis.axvspan(index - 0.48, index + 0.48, color="#7B61A8", alpha=0.07)
    axis.axhline(1.0, color="0.3", linestyle="--")
    axis.set(xticks=positions, xticklabels=SPIN_PARAMETER_NAMES,
             ylabel="Conditional sigma / baseline",
             title="Full 15-component passive transfer; shaded components span the BC5 target")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False, ncol=len(policies))
    _finish_axis(axis)
    fig.savefig(output / "full_spin_passive_transfer_after_BC5.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "full_spin_passive_transfer_after_BC5", rows)


def _plot_multi_measurement_summary(
    results: dict[str, dict[str, Any]], histories: dict[str, list[dict[str, Any]]],
    training_metadata: dict[str, dict[str, Any]], spin: dict[str, Any], output: Path,
) -> None:
    order = [("baseline", "Baseline"), ("cnn_only", "Cnn-only"),
             ("cdiag", "Cdiag"), ("bc5_adaptive", "BC5 adaptive")]
    baseline_fisher = np.asarray(_bc5_measurement_result(results["baseline"])["fisher"])
    baseline_sigma = np.asarray(_bc5_measurement_result(results["baseline"]).get("profiled_sigma_per_sqrt_event", np.full(5, np.nan)))
    table_rows = []
    source_rows: list[dict[str, Any]] = []
    for key, label in order:
        if key not in results:
            continue
        subset = _bc5_measurement_result(results[key])
        fisher = np.asarray(subset["fisher"])
        objective, _, covariance = _numpy_objectives(fisher, baseline_fisher, False)
        sigma = np.sqrt(np.diag(covariance))
        history = histories.get("adaptive", []) if key == "bc5_adaptive" else []
        checked = [row for row in history if row.get("J_direct") is not None]
        metadata = training_metadata.get(key, {})
        refreshes = metadata.get("score_refreshes")
        closure = checked[-1]["closure_ratio"] if checked else None
        mean_gain = float(np.nanmean(baseline_sigma / sigma - 1.0))
        row = {
            "policy": key, "mean_profiled_precision_gain": mean_gain, "J": objective,
            "condition_number": subset["condition_number"],
            "valid_efficiency": results[key]["valid_efficiency"],
            "tau_axis_error_degrees": float(np.degrees(results[key]["tau_axis_error"])),
            "score_refreshes": refreshes,
            "total_score_training_epochs": metadata.get("total_score_training_epochs"),
            "total_dgpo_epochs": metadata.get("total_dgpo_epochs"),
            "final_direct_surrogate_closure": closure,
        }
        source_rows.append(row)
        table_rows.append([
            f"{100 * mean_gain:+.1f}%", f"{objective:.3f}", f"{subset['condition_number']:.2g}",
            f"{results[key]['valid_efficiency']:.3f}", f"{np.degrees(results[key]['tau_axis_error']):.2f}",
            "n/a" if refreshes is None else str(refreshes), "n/a" if closure is None else f"{closure:.3f}",
        ])
    columns = ["Mean precision gain", "J", "Condition", "Valid eff.", "Axis error [deg]", "Refreshes", "Closure"]
    fig, axis = plt.subplots(figsize=(11.5, 3.8), constrained_layout=True)
    axis.axis("off")
    table = axis.table(cellText=table_rows, rowLabels=[label for key, label in order if key in results],
                       colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.65)
    axis.set_title("Multi-measurement decision summary", pad=18)
    fig.savefig(output / "multi_measurement_summary.png", dpi=int(spin["plots"]["dpi"]))
    plt.close(fig)
    _write_rows(output / "multi_measurement_summary", source_rows)


def _comparison_training_metadata(
    spin: dict[str, Any], base: dict[str, Any], histories: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {
        "baseline": {"score_refreshes": 0, "total_score_training_epochs": 0, "total_dgpo_epochs": 0},
    }
    cnn_path = (
        Path(spin["cnn_output_dir"]).expanduser().resolve()
        / "refresh_invariants_iterative_refresh_no_trust.json"
    )
    if cnn_path.exists():
        with cnn_path.open(encoding="utf-8") as stream:
            values = json.load(stream)
        rounds = int(values["actual_refresh_rounds"])
        metadata["cnn_only"] = {
            "score_refreshes": rounds,
            "total_score_training_epochs": rounds * int(base["refresh"]["score_epochs"]),
            "total_dgpo_epochs": int(values["actual_total_dgpo_epochs"]),
        }
    cdiag_dir = spin.get("cdiag_output_dir")
    if cdiag_dir is not None:
        cdiag_path = Path(cdiag_dir).expanduser().resolve() / "multi_training_Cdiag.json"
        cdiag_config_path = Path(cdiag_dir).expanduser().resolve() / "resolved_spin_matrix_config.yaml"
        if cdiag_path.exists():
            with cdiag_path.open(encoding="utf-8") as stream:
                values = json.load(stream)
            score_epochs = int(spin["multi_training"]["score_epochs"])
            dgpo_epochs_per_round = int(spin["multi_training"]["dgpo_epochs_per_round"])
            if cdiag_config_path.exists():
                with cdiag_config_path.open(encoding="utf-8") as stream:
                    cdiag_settings = yaml.safe_load(stream)["multi_training"]
                score_epochs = int(cdiag_settings["score_epochs"])
                dgpo_epochs_per_round = int(cdiag_settings["dgpo_epochs_per_round"])
            completed = int(values["final_round"])
            metadata["cdiag"] = {
                "score_refreshes": completed,
                "total_score_training_epochs": (completed + 1) * score_epochs,
                "total_dgpo_epochs": completed * dgpo_epochs_per_round,
            }
    adaptive = histories.get("adaptive", [])
    if adaptive:
        final = adaptive[-1]
        metadata["bc5_adaptive"] = {
            "score_refreshes": int(sum(row["refresh_triggered"] for row in adaptive)),
            "total_score_training_epochs": int(final["score_training_epochs"] + final["diagnostic_score_training_epochs"]),
            "total_dgpo_epochs": int(final["total_dgpo_epoch"]),
        }
    return metadata


def _load_multi_histories(spin: dict[str, Any], output: Path) -> dict[str, list[dict[str, Any]]]:
    histories = {}
    for target_name in spin["multi_training"]["enabled_targets"]:
        path = output / f"multi_training_{target_name}.json"
        with path.open(encoding="utf-8") as stream:
            histories[target_name] = json.load(stream)["rounds"]
    return histories


def _load_controlled_histories(
    spin: dict[str, Any], output: Path,
) -> dict[str, list[dict[str, Any]]]:
    target = str(spin["multi_training"]["primary_target"])
    histories = {}
    for refresh_mode in spin["multi_training"]["refresh_modes"]:
        with (output / f"multi_training_{target}_{refresh_mode}.json").open(encoding="utf-8") as stream:
            histories[str(refresh_mode)] = json.load(stream)["epochs"]
    return histories


def _load_cdiag_comparison_policy(
    spin: dict[str, Any], base: dict[str, Any], device: torch.device,
) -> ConditionalFlow | None:
    configured = spin.get("cdiag_output_dir")
    if configured is None:
        return None
    checkpoint = (
        Path(configured).expanduser().resolve() / "checkpoints"
        / str(spin.get("cdiag_checkpoint_name", "iterative_refresh_multi_no_trust"))
        / "best_validation_policy.pt"
    )
    if not checkpoint.exists():
        if bool(spin.get("require_cdiag_checkpoint", True)):
            raise FileNotFoundError(f"Required independent Cdiag checkpoint is missing: {checkpoint}")
        return None
    return _load_flow(checkpoint, base, device)


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
    controlled = "refresh_modes" in spin["multi_training"]
    if mode in {"spin-run", "spin-train"}:
        if controlled:
            for target_name in spin["multi_training"]["enabled_targets"]:
                parameters = list(spin["multi_training"]["target_sets"][target_name])
                for refresh_mode in spin["multi_training"]["refresh_modes"]:
                    policy, rows = _train_controlled_multi_measurement_policy(
                        policies["baseline"], target_name, parameters, str(refresh_mode),
                        spin, base, device, output,
                    )
                    histories[str(refresh_mode)] = rows
                    policies[f"{str(target_name).lower()}_{refresh_mode}"] = policy
            if "adaptive" in histories:
                _plot_adaptive_refresh_diagnostics(histories["adaptive"], spin, output)
            baseline_path = output / f"baseline_fisher_{spin['multi_training']['primary_target']}_adaptive.json"
            if not baseline_path.exists():
                baseline_path = output / f"baseline_fisher_{spin['multi_training']['primary_target']}_fixed.json"
            with baseline_path.open(encoding="utf-8") as stream:
                baseline_sigma = np.asarray(json.load(stream)["direct_profiled_sigma_per_sqrt_event"])
            _plot_fixed_vs_adaptive(
                histories, baseline_sigma,
                list(spin["multi_training"]["target_sets"][spin["multi_training"]["primary_target"]]),
                spin, output,
            )
        else:
            for target_name in spin["multi_training"]["enabled_targets"]:
                parameters = list(spin["multi_training"]["target_sets"][target_name])
                policy, rows = _train_legacy_multi_measurement_policy(
                    policies["baseline"], target_name, parameters, spin, base, device, output,
                )
                histories[target_name] = rows
                policies[f"multi_{target_name}"] = policy
            _plot_multi_training(histories, spin, output)
    elif mode == "spin-evaluate":
        if controlled:
            histories = _load_controlled_histories(spin, output)
            target_name = str(spin["multi_training"]["primary_target"])
            for refresh_mode in spin["multi_training"]["refresh_modes"]:
                policy_name = _multi_policy_name(target_name, target_name, str(refresh_mode), True)
                policies[f"{target_name.lower()}_{refresh_mode}"] = _load_flow(
                    output / "checkpoints" / policy_name / "best_validation_policy.pt", base, device,
                )
            _plot_adaptive_refresh_diagnostics(histories.get("adaptive", []), spin, output)
            baseline_path = output / f"baseline_fisher_{target_name}_adaptive.json"
            if not baseline_path.exists():
                baseline_path = output / f"baseline_fisher_{target_name}_fixed.json"
            with baseline_path.open(encoding="utf-8") as stream:
                baseline_sigma = np.asarray(json.load(stream)["direct_profiled_sigma_per_sqrt_event"])
            _plot_fixed_vs_adaptive(
                histories, baseline_sigma,
                list(spin["multi_training"]["target_sets"][target_name]), spin, output,
            )
        else:
            histories = _load_multi_histories(spin, output)
            _plot_multi_training(histories, spin, output)
            primary = str(spin["multi_training"]["primary_target"])
            for target_name in spin["multi_training"]["enabled_targets"]:
                policy_name = _multi_policy_name(str(target_name), primary)
                policies[f"multi_{target_name}"] = _load_flow(
                    output / "checkpoints" / policy_name / "best_validation_policy.pt", base, device,
                )
    if controlled and mode == "spin-passive":
        target_name = str(spin["multi_training"]["primary_target"])
        histories = _load_controlled_histories(spin, output)
        for refresh_mode in spin["multi_training"]["refresh_modes"]:
            policy_name = _multi_policy_name(target_name, target_name, str(refresh_mode), True)
            policies[f"{target_name.lower()}_{refresh_mode}"] = _load_flow(
                output / "checkpoints" / policy_name / "best_validation_policy.pt", base, device,
            )
    if mode in {"spin-run", "spin-evaluate", "spin-passive"}:
        cdiag_policy = _load_cdiag_comparison_policy(spin, base, device)
        if cdiag_policy is not None:
            policies["cdiag"] = cdiag_policy
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
        if "bc5_adaptive" in results:
            _plot_bc5_profiled_precision(results, spin, output)
            _plot_bc5_eigenmodes(results, spin, output)
            _plot_full_spin_passive_transfer(results, spin, output)
            _plot_multi_measurement_summary(
                results, histories, _comparison_training_metadata(spin, base, histories),
                spin, output,
            )
    print(f"Spin-matrix results written to {output}", flush=True)
