from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from .core import (
    ScoreModel, comparison_policy_names, load_config, make_generator, resolve_device,
    seed_everything,
)
from .flow import ConditionalFlow
from .inference import binned_fisher_per_event, fit_poisson, reweighted_templates
from .training import (
    make_flow, make_reference_reconstruction, reconstruct_policy, slice_events,
    train_baseline, train_dgpo, train_iterative_refresh, train_reference_score,
    validation_fisher_metrics,
)
from .ztautau import generate_events


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    output = Path(config["output_dir"]).expanduser().resolve()
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    return output, checkpoints


def _split_nominal(events: dict[str, torch.Tensor], config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    count = events["x"].numel()
    train_end = int(count * float(config["data"]["train_fraction"]))
    score_end = train_end + int(count * float(config["data"]["score_fraction"]))
    if not 0 < train_end < score_end < count:
        raise ValueError("Nominal split fractions must leave non-empty train, score, and calibration samples")
    return slice_events(events, slice(0, train_end)), slice_events(events, slice(train_end, score_end)), slice_events(events, slice(score_end, count))


def _nominal_events(config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return generate_events(
        int(config["data"]["nominal_events"]), float(config["physics"]["nominal_C"]),
        config, device, make_generator(device, int(config["seed"])),
    )


def _scan(config: dict[str, Any]) -> np.ndarray:
    bounds = config["inference"]["scan_range"]
    return np.linspace(float(bounds[0]), float(bounds[1]), int(config["inference"]["scan_points"]))


def _fit_reconstructed(
    reconstructed: dict[str, torch.Tensor], calibration_events: dict[str, torch.Tensor],
    calibration: dict[str, torch.Tensor], config: dict[str, Any], bins: int | None = None,
) -> tuple[float, float, float, np.ndarray]:
    bin_count = bins or int(config["inference"]["reco_bins"])
    edges = np.linspace(-1.0, 1.0, bin_count + 1)
    valid = reconstructed["valid"].cpu().numpy().astype(bool)
    observed, _ = np.histogram(reconstructed["y"].cpu().numpy()[valid], bins=edges)
    calibration_valid = calibration["valid"].cpu().numpy().astype(bool)
    templates = reweighted_templates(
        calibration_events["x"].cpu().numpy(), calibration["y"].cpu().numpy(), calibration_valid,
        edges, float(config["physics"]["nominal_C"]), _scan(config), float(reconstructed["y"].numel()),
    )
    return fit_poisson(observed, templates, _scan(config))


def _fit_with_templates(
    reconstructed: dict[str, torch.Tensor], templates: np.ndarray, config: dict[str, Any],
) -> tuple[float, float, float, np.ndarray]:
    edges = np.linspace(-1.0, 1.0, int(config["inference"]["reco_bins"]) + 1)
    valid = reconstructed["valid"].cpu().numpy().astype(bool)
    observed, _ = np.histogram(reconstructed["y"].cpu().numpy()[valid], bins=edges)
    return fit_poisson(observed, templates, _scan(config))


def _build_high_stat_templates(
    config: dict[str, Any],
    device: torch.device,
    policies: dict[str, ConditionalFlow],
    target_exposure: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]]]:
    settings = config["inference"]
    total_events = int(settings["calibration_events"])
    chunk_size = int(settings["calibration_chunk_size"])
    if total_events <= 0 or chunk_size <= 0:
        raise ValueError("inference calibration event counts must be positive")
    nominal_C = float(config["physics"]["nominal_C"])
    edges = np.linspace(-1.0, 1.0, int(settings["reco_bins"]) + 1)
    fisher_edges = np.linspace(-1.0, 1.0, int(config["refresh"]["direct_fisher_bins"]) + 1)
    bin_count = edges.size - 1
    fisher_bin_count = fisher_edges.size - 1
    counts = {name: np.zeros(bin_count) for name in policies}
    derivatives = {name: np.zeros(bin_count) for name in policies}
    fisher_counts = {name: np.zeros(fisher_bin_count) for name in policies}
    fisher_derivatives = {name: np.zeros(fisher_bin_count) for name in policies}
    chunks = (total_events + chunk_size - 1) // chunk_size
    progress = tqdm(total=chunks * len(policies), desc="high-stat policy calibration", unit="policy-chunk")
    for chunk in range(chunks):
        count = min(chunk_size, total_events - chunk * chunk_size)
        events = generate_events(
            count, nominal_C, config, device,
            make_generator(device, int(config["seed"]) + 300000 + chunk),
        )
        x = events["x"].cpu().numpy()
        score = x / (1.0 + nominal_C * x)
        for name, policy in policies.items():
            reconstructed = reconstruct_policy(
                policy, events, config,
                make_generator(device, int(config["seed"]) + 310000 + chunk),
            )
            valid = reconstructed["valid"].cpu().numpy().astype(bool)
            y = reconstructed["y"].cpu().numpy()[valid]
            indices = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, bin_count - 1)
            fisher_indices = np.clip(
                np.searchsorted(fisher_edges, y, side="right") - 1, 0, fisher_bin_count - 1,
            )
            counts[name] += np.bincount(indices, minlength=bin_count)
            derivatives[name] += np.bincount(indices, weights=score[valid], minlength=bin_count)
            fisher_counts[name] += np.bincount(fisher_indices, minlength=fisher_bin_count)
            fisher_derivatives[name] += np.bincount(
                fisher_indices, weights=score[valid], minlength=fisher_bin_count,
            )
            progress.update()
    progress.close()
    scale = target_exposure / total_events
    scan = _scan(config)
    templates = {
        name: scale * (counts[name][None, :] + (scan[:, None] - nominal_C) * derivatives[name][None, :])
        for name in policies
    }
    metrics = {
        name: {
            "events": total_events,
            "valid_efficiency": float(counts[name].sum() / total_events),
            "direct_fisher_bins": fisher_bin_count,
            "binned_fisher_per_event": float(np.sum(
                fisher_derivatives[name] ** 2 / np.clip(fisher_counts[name], 1.0e-12, None)
            ) / total_events),
        }
        for name in policies
    }
    return templates, metrics


def validate_reference_fisher(
    reference_flow: ConditionalFlow, score_model: ScoreModel, calibration_events: dict[str, torch.Tensor],
    config: dict[str, Any], device: torch.device, output: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    calibration = make_reference_reconstruction(reference_flow, score_model, calibration_events, config, int(config["seed"]) + 21000)
    settings = config["fisher_validation"]
    event_count = int(settings["events_per_pseudo_experiment"])
    score_fisher_per_event = float(calibration["information"].mean())
    edges = np.linspace(-1.0, 1.0, int(settings["reco_bins"]) + 1)
    binned = binned_fisher_per_event(
        calibration_events["x"].cpu().numpy(), calibration["y"].cpu().numpy(),
        calibration["valid"].cpu().numpy().astype(bool), edges, float(config["physics"]["nominal_C"]),
    )
    estimates = []
    for index in tqdm(range(int(settings["pseudo_experiments"])), desc="pre-DGPO Fisher closure", unit="toy"):
        generator = make_generator(device, int(config["seed"]) + 22000 + index)
        pseudo = generate_events(event_count, float(config["physics"]["nominal_C"]), config, device, generator)
        reconstructed = reconstruct_policy(reference_flow, pseudo, config, generator)
        estimate, _, _, _ = _fit_reconstructed(reconstructed, calibration_events, calibration, config, int(settings["reco_bins"]))
        estimates.append(estimate)
    observed_sigma = float(np.std(estimates, ddof=1))
    sigma_score = float(1.0 / np.sqrt(score_fisher_per_event * event_count))
    sigma_binned = float(1.0 / np.sqrt(binned * event_count))
    values = np.array([sigma_score, sigma_binned, observed_sigma])
    disagreement = float(values.max() / values.min() - 1.0)
    result = {
        "score_fisher_per_event": score_fisher_per_event,
        "binned_fisher_per_event": binned,
        "score_fisher_for_pseudo_experiment": score_fisher_per_event * event_count,
        "binned_fisher_for_pseudo_experiment": binned * event_count,
        "sigma_score": sigma_score,
        "sigma_binned": sigma_binned,
        "sigma_pseudo_experiments": observed_sigma,
        "relative_spread": disagreement,
        "relative_tolerance": float(settings["relative_tolerance"]),
        "passed": disagreement <= float(settings["relative_tolerance"]),
        "pseudo_estimates": estimates,
    }
    with (output / "fisher_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result, calibration


def _save_models(
    policies: dict[str, ConditionalFlow], score_model: ScoreModel, checkpoints: Path
) -> None:
    for name, policy in policies.items():
        torch.save({"method_version": 3, "state_dict": policy.state_dict()}, checkpoints / f"{name}.pt")
    torch.save({"method_version": 3, "state_dict": score_model.state_dict()}, checkpoints / "reference_score.pt")


def _load_models(config: dict[str, Any], device: torch.device, checkpoints: Path) -> tuple[dict[str, ConditionalFlow], ScoreModel]:
    enabled = {name for name, value in config["policies"].items() if value}
    names = comparison_policy_names(config, enabled)
    policies: dict[str, ConditionalFlow] = {}
    for name in names:
        payload = torch.load(checkpoints / f"{name}.pt", map_location=device, weights_only=True)
        if payload.get("method_version") != 3:
            raise RuntimeError("Checkpoint is not from the Z-to-tau-tau flow method; retrain it")
        policy = make_flow(config, device)
        policy.load_state_dict(payload["state_dict"])
        policies[name] = policy.eval()
    score_payload = torch.load(checkpoints / "reference_score.pt", map_location=device, weights_only=True)
    if score_payload.get("method_version") != 3:
        raise RuntimeError("Score checkpoint is not from the Z-to-tau-tau flow method; retrain it")
    settings = config["training"]
    score_model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
    score_model.load_state_dict(score_payload["state_dict"])
    return policies, score_model.eval()


def _write_history(output: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    for name, rows in histories.items():
        with (output / f"training_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def _write_checkpoint_validation(
    output: Path,
    histories: dict[str, list[dict[str, Any]]],
    baseline_row: dict[str, Any],
) -> None:
    combined: list[dict[str, Any]] = []
    for name, rows in histories.items():
        combined.extend(rows)
        with (output / f"checkpoint_validation_{name}.json").open("w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
        with (output / f"checkpoint_validation_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (output / "checkpoint_validation_all.json").open("w", encoding="utf-8") as stream:
        json.dump([baseline_row, *combined], stream, indent=2)

    baseline_fisher = float(baseline_row["validation_fisher"])
    baseline_sigma = float(baseline_row["validation_sigma"])
    summary: list[dict[str, Any]] = []
    for name, rows in histories.items():
        best = max(rows, key=lambda row: float(row["validation_fisher"]))
        final = rows[-1]
        for role, row in (("best", best), ("final", final)):
            summary.append({
                "policy": name,
                "checkpoint_role": role,
                "checkpoint": row["checkpoint"],
                "epoch": row["epoch"],
                "round": row["round"],
                "validation_fisher": row["validation_fisher"],
                "validation_fisher_per_event": row["validation_fisher_per_event"],
                "validation_sigma": row["validation_sigma"],
                "validation_fisher_40_100_relative_spread": row["validation_fisher_40_100_relative_spread"],
                "fisher_gain_vs_baseline": float(row["validation_fisher"]) / baseline_fisher - 1.0,
                "predicted_sigma_reduction": 1.0 - float(row["validation_sigma"]) / baseline_sigma,
                "global_kl_to_baseline": row["global_kl_to_baseline"],
                "valid_fraction": row["valid_fraction"],
                "angular_error": row["angular_error"],
            })
    with (output / "checkpoint_selection_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    with (output / "checkpoint_selection_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _write_refresh_artifacts(
    output: Path,
    checkpoints: Path,
    config: dict[str, Any],
    name: str,
    score_model: ScoreModel,
    rounds: list[dict[str, Any]],
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {"method_version": 3, "diagnostic_only": False, "state_dict": score_model.state_dict()},
        checkpoints / f"{name}_surrogate_score.pt",
    )
    serializable = [{key: value for key, value in row.items() if key not in {"score_grid", "score_curve"}} for row in rounds]
    with (output / f"refresh_rounds_{name}.json").open("w", encoding="utf-8") as stream:
        json.dump(serializable, stream, indent=2)
    np.savez(
        output / f"score_evolution_{name}.npz",
        grid=np.stack([row["score_grid"] for row in rounds]),
        scores=np.stack([row["score_curve"] for row in rounds]),
    )
    score_shifts = [
        float(np.sqrt(np.mean((rounds[index]["score_curve"] - rounds[index - 1]["score_curve"]) ** 2)))
        for index in range(1, len(rounds))
    ]
    invariants = {
        "all_rounds_start_exactly_from_current_reference": all(row["start_max_parameter_difference"] == 0.0 for row in history),
        "local_kl_reference": "current_round_policy",
        "global_kl_reference": "original_baseline",
        "candidate_replacement": "I_round - i_event_round + i_candidate",
        "fisher_rebuilt_each_round": len(rounds) == int(len({row["round"] for row in history})),
        "independent_fisher_sample_each_round": len({row["fisher_sample_seed"] for row in rounds}) == len(rounds),
        "fixed_training_reference_seed_each_round": len({row["training_reference_seed"] for row in rounds}) == len(rounds),
        "loss_beta_local": history[0]["beta_local"],
        "loss_beta_global": history[0]["beta_global"],
        "no_trust_loss_has_no_kl": name != "iterative_refresh_no_trust" or all(
            row["beta_local"] == 0.0 and row["beta_global"] == 0.0 for row in history
        ),
        "score_curve_rms_changes": score_shifts,
        "score_refresh_changed_model": all(value > 0.0 for value in score_shifts),
        "configured_total_dgpo_epochs": int(config["refresh"]["rounds"]) * int(config["refresh"]["dgpo_epochs_per_round"]),
        "actual_refresh_rounds": len(rounds),
        "actual_total_dgpo_epochs": len(rounds) * int(config["refresh"]["dgpo_epochs_per_round"]),
        "stopped_before_maximum_rounds": len(rounds) < int(config["refresh"]["rounds"]),
        "selection_metric": f"independent_{int(config['ablation']['selection_bins'])}_bin_validation_fisher",
    }
    with (output / f"refresh_invariants_{name}.json").open("w", encoding="utf-8") as stream:
        json.dump(invariants, stream, indent=2)


def train_pipeline(
    config: dict[str, Any], device: torch.device, nominal: dict[str, torch.Tensor], output: Path, checkpoints: Path,
) -> tuple[dict[str, ConditionalFlow], ScoreModel, dict[str, list[dict[str, float]]], dict[str, Any], dict[str, torch.Tensor]]:
    train_events, score_events, calibration_events = _split_nominal(nominal, config)
    baseline = train_baseline(train_events, config, device)
    reference_flow = baseline.eval()
    for parameter in reference_flow.parameters():
        parameter.requires_grad_(False)
    score_model, _ = train_reference_score(reference_flow, score_events, config, device)
    fisher_validation, baseline_calibration = validate_reference_fisher(reference_flow, score_model, calibration_events, config, device, output)
    if not fisher_validation["passed"]:
        from .plots import plot_pre_dgpo
        plot_pre_dgpo(
            nominal, calibration_events, score_model,
            baseline_calibration, fisher_validation, config, output,
        )
        raise RuntimeError(f"Pre-DGPO Fisher closure failed: relative spread {fisher_validation['relative_spread']:.3f} exceeds tolerance")
    reference_train = make_reference_reconstruction(reference_flow, score_model, train_events, config, int(config["seed"]) + 23000)
    validation_events = generate_events(
        int(config["ablation"]["validation_events"]), float(config["physics"]["nominal_C"]),
        config, device, make_generator(
            device, int(config["seed"]) + int(config["ablation"]["validation_event_seed_offset"]),
        ),
    )
    baseline_validation = validation_fisher_metrics(
        reference_flow, reference_flow, score_model, validation_events, config,
    )
    baseline_validation.update({
        "policy": "baseline", "checkpoint": "pi_0", "epoch": 0, "round": 0,
    })
    baseline_dir = checkpoints / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"method_version": 3, "policy": "baseline", "state_dict": reference_flow.state_dict()},
        baseline_dir / "best_validation_policy.pt",
    )
    torch.save(
        {"method_version": 3, "policy": "baseline", "state_dict": reference_flow.state_dict()},
        baseline_dir / "final_policy.pt",
    )
    policies: dict[str, ConditionalFlow] = {"baseline": reference_flow}
    histories: dict[str, list[dict[str, float]]] = {}
    validation_histories: dict[str, list[dict[str, Any]]] = {}
    if config["policies"].get("fisher_dgpo_no_trust", False):
        _, policies["fisher_dgpo_no_trust"], histories["fisher_dgpo_no_trust"], validation_histories["fisher_dgpo_no_trust"] = train_dgpo(
            reference_flow, score_model, train_events, reference_train, config, device,
            "fisher_dgpo_no_trust", float(config["dgpo"]["no_trust_kl_coefficient"]), checkpoints,
            validation_events,
        )
    if config["policies"].get("fisher_dgpo_trust", False):
        _, policies["fisher_dgpo_trust"], histories["fisher_dgpo_trust"], validation_histories["fisher_dgpo_trust"] = train_dgpo(
            reference_flow, score_model, train_events, reference_train, config, device,
            "fisher_dgpo_trust", float(config["dgpo"]["trust_kl_coefficient"]), checkpoints,
            validation_events,
        )
    if config["policies"].get("iterative_refresh_trust", False):
        _, policy, active_score, refresh_history, rounds, validation_rows = train_iterative_refresh(
            reference_flow, train_events, config, device, "iterative_refresh_trust", checkpoints,
            validation_events,
        )
        policies["iterative_refresh_trust"] = policy
        histories["iterative_refresh_trust"] = refresh_history
        validation_histories["iterative_refresh_trust"] = validation_rows
        _write_refresh_artifacts(output, checkpoints, config, "iterative_refresh_trust", active_score, rounds, refresh_history)
    if config["policies"].get("iterative_refresh_no_trust", False):
        _, policy, active_score, refresh_history, rounds, validation_rows = train_iterative_refresh(
            reference_flow, train_events, config, device, "iterative_refresh_no_trust", checkpoints,
            validation_events, 0.0, 0.0,
        )
        policies["iterative_refresh_no_trust"] = policy
        histories["iterative_refresh_no_trust"] = refresh_history
        validation_histories["iterative_refresh_no_trust"] = validation_rows
        _write_refresh_artifacts(output, checkpoints, config, "iterative_refresh_no_trust", active_score, rounds, refresh_history)
    if config["policies"].get("fisher_dgpo_trust_bias_control", False):
        raise RuntimeError("Bias-control policy is intentionally disabled until the three primary policies establish score imbalance")
    policies = {
        name: policies[name]
        for name in comparison_policy_names(config, policies)
    }
    _save_models(policies, score_model, checkpoints)
    _write_history(output, histories)
    ordered_validation_histories = {
        name: validation_histories[name]
        for name in config["ablation"]["policy_order"]
        if name != "baseline" and name in validation_histories
    }
    _write_checkpoint_validation(output, ordered_validation_histories, baseline_validation)
    return policies, score_model, histories, fisher_validation, baseline_calibration


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for policy, C_true in sorted({(row["policy"], row["C_true"]) for row in rows}):
        selected = [row for row in rows if row["policy"] == policy and row["C_true"] == C_true]
        estimates = np.array([row["C_hat"] for row in selected])
        pulls = np.array([row["pull"] for row in selected])
        bias = float(estimates.mean() - C_true)
        spread = float(estimates.std(ddof=1))
        coverage = float(np.mean([row["covered_68"] for row in selected]))
        count = len(selected)
        denominator = 1.0 + 1.0 / count
        center = (coverage + 0.5 / count) / denominator
        half = np.sqrt(coverage * (1.0 - coverage) / count + 0.25 / count**2) / denominator
        summaries.append({
            "policy": policy, "C_true": C_true, "mean_C_hat": float(estimates.mean()),
            "mean_C_hat_error": spread / np.sqrt(count), "bias": bias,
            "normalized_bias": bias / spread if spread > 0.0 else None, "std_C_hat": spread,
            "rmse": float(np.sqrt(bias**2 + spread**2)), "mean_reported_sigma": float(np.mean([row["sigma_C_hat"] for row in selected])),
            "pull_mean": float(pulls.mean()), "pull_std": float(pulls.std(ddof=1)),
            "coverage_68": coverage, "coverage_68_low": center - half, "coverage_68_high": center + half,
            "pseudo_experiments": count,
        })
    return summaries


def evaluate_pipeline(
    config: dict[str, Any], device: torch.device, nominal: dict[str, torch.Tensor], policies: dict[str, ConditionalFlow],
    score_model: ScoreModel, histories: dict[str, list[dict[str, float]]], fisher_validation: dict[str, Any], output: Path,
) -> list[dict[str, Any]]:
    _, _, calibration_events = _split_nominal(nominal, config)
    calibrations = {
        name: make_reference_reconstruction(policy, score_model, calibration_events, config, int(config["seed"]) + 50000)
        for name, policy in policies.items()
    }
    rows: list[dict[str, Any]] = []
    experiments = int(config["data"]["pseudo_experiments"])
    events_per = int(config["data"]["events_per_pseudo_experiment"])
    fit_templates, calibration_metrics = _build_high_stat_templates(config, device, policies, float(events_per))
    with (output / "high_stat_calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(calibration_metrics, stream, indent=2)
    total = len(policies) * len(config["physics"]["true_C_values"]) * experiments
    progress = tqdm(total=total, desc="off-nominal closure", unit="toy")
    for name, policy in policies.items():
        for truth_index, value in enumerate(config["physics"]["true_C_values"]):
            C_true = float(value)
            for experiment in range(experiments):
                generator = make_generator(device, int(config["seed"]) + 100000 + 1000 * truth_index + experiment)
                pseudo = generate_events(events_per, C_true, config, device, generator)
                reconstructed = reconstruct_policy(policy, pseudo, config, generator)
                estimate, lower, upper, _ = _fit_with_templates(reconstructed, fit_templates[name], config)
                sigma = 0.5 * (lower + upper)
                rows.append({
                    "policy": name, "C_true": C_true, "pseudo_experiment": experiment,
                    "C_hat": estimate, "sigma_minus": lower, "sigma_plus": upper, "sigma_C_hat": sigma,
                    "pull": (estimate - C_true) / sigma if sigma > 0.0 else np.nan,
                    "covered_68": estimate - lower <= C_true <= estimate + upper,
                    "valid_fraction": float(reconstructed["valid"].float().mean()),
                })
                progress.update()
                progress.set_postfix(policy=name, C_true=f"{C_true:.2f}")
    progress.close()
    summaries = _summarize(rows)
    for path, data in ((output / "pseudo_experiments.csv", rows), (output / "summary.csv", summaries)):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    from .plots import make_all_plots
    make_all_plots(
        nominal, calibration_events, policies, score_model, calibrations,
        histories, fisher_validation, summaries, config, device, output,
    )
    if config["ablation"].get("enabled", False):
        from .refresh_study import make_ablation_study
        make_ablation_study(
            policies, score_model, histories, summaries, config, device, output, output / "checkpoints",
        )
    return summaries


def run(config_path: str | Path, mode: str, device_override: str | None, output_override: str | None) -> None:
    config = load_config(config_path)
    if device_override is not None:
        config["device"] = device_override
    if output_override is not None:
        config["output_dir"] = output_override
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))
    output, checkpoints = _paths(config)
    print(f"Using device: {device}", flush=True)
    if mode in {"diagnose", "closure"}:
        resolved_path = output / "resolved_config.yaml"
        if not resolved_path.exists():
            raise FileNotFoundError(f"Missing {resolved_path}; {mode} requires an existing trained/evaluated run")
        with resolved_path.open(encoding="utf-8") as stream:
            trained_config = yaml.safe_load(stream)
        compared_sections = ("physics", "detector", "flow", "training", "refresh", "ablation", "inference", "policies")
        mismatched = [section for section in compared_sections if trained_config.get(section) != config.get(section)]
        if mismatched:
            raise RuntimeError(f"Diagnostic configuration does not match the frozen run in sections: {mismatched}")
        with (output / f"resolved_{mode}_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        policies, score_model = _load_models(config, device, checkpoints)
        if mode == "diagnose":
            from .diagnosis import run_diagnosis
            run_diagnosis(config, device, policies, score_model, output)
        else:
            from .closure import run_statistical_closure
            run_statistical_closure(config, device, policies, output)
        return
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    nominal = _nominal_events(config, device)
    if mode in {"run", "train"}:
        policies, score_model, histories, fisher_validation, _ = train_pipeline(config, device, nominal, output, checkpoints)
    else:
        policies, score_model = _load_models(config, device, checkpoints)
        histories = {}
        for name in policies:
            path = output / f"training_{name}.csv"
            if path.exists():
                with path.open(encoding="utf-8") as stream:
                    histories[name] = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        _, _, calibration_events = _split_nominal(nominal, config)
        fisher_validation, _ = validate_reference_fisher(policies["baseline"], score_model, calibration_events, config, device, output)
        if not fisher_validation["passed"]:
            raise RuntimeError("Pre-DGPO Fisher closure failed during evaluation")
    if mode in {"run", "evaluate"}:
        summaries = evaluate_pipeline(config, device, nominal, policies, score_model, histories, fisher_validation, output)
        print(f"Results written to {output}", flush=True)
        for row in summaries:
            print(f"{row['policy']:>24s} C_true={row['C_true']:.2f}: mean={row['mean_C_hat']:.4f}, std={row['std_C_hat']:.4f}, bias={row['bias']:+.4f}, coverage={row['coverage_68']:.3f}", flush=True)
