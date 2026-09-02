from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
import yaml

from .core import (
    detector_features,
    load_config,
    make_generator,
    make_policy,
    reconstruct,
    resolve_device,
    sample_truth,
    seed_everything,
)
from .training import train_baseline, train_dgpo, train_score_model
from .diagnostics import make_diagnostics
from .unfolding import (
    fit_poisson_parameter,
    response_matrix,
    reweighted_reco_templates,
)


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    output = Path(config["output_dir"]).expanduser().resolve()
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    return output, checkpoints


def _save_config(config: dict[str, Any], output: Path) -> None:
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def _nominal_sample(config: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = make_generator(device, int(config["seed"]))
    truth = sample_truth(
        int(config["data"]["nominal_events"]),
        float(config["physics"]["nominal_C"]),
        device,
        generator,
    )
    return detector_features(truth, config["physics"], generator), truth


def _split_nominal(
    features: torch.Tensor, truth: torch.Tensor, config: dict[str, Any]
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    count = truth.numel()
    train_end = int(count * float(config["data"]["train_fraction"]))
    score_end = train_end + int(count * float(config["data"]["score_fraction"]))
    if not 0 < train_end < score_end < count:
        raise ValueError("train_fraction and score_fraction must leave a non-empty response split")
    return (
        (features[:train_end], truth[:train_end]),
        (features[train_end:score_end], truth[train_end:score_end]),
        (features[score_end:], truth[score_end:]),
    )


def train_pipeline(config: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    output, checkpoints = _paths(config)
    _save_config(config, output)
    features, truth = _nominal_sample(config, device)
    train_data, score_data, _ = _split_nominal(features, truth, config)

    baseline = train_baseline(*train_data, config, device)
    score_model = train_score_model(baseline, *score_data, config, device)
    policies: dict[str, torch.nn.Module] = {"baseline": baseline}
    policy_scores: dict[str, torch.nn.Module] = {"baseline": score_model}
    if bool(config["policies"].get("fisher_dgpo", False)):
        policies["fisher_dgpo"], policy_scores["fisher_dgpo"] = train_dgpo(
            baseline, score_model, *train_data, *score_data, config, device, False
        )
    if bool(config["policies"].get("bias_controlled_dgpo", False)):
        policies["bias_controlled_dgpo"], policy_scores["bias_controlled_dgpo"] = train_dgpo(
            baseline, score_model, *train_data, *score_data, config, device, True
        )

    for name, policy in policies.items():
        torch.save({"method_version": 2, "state_dict": policy.state_dict()}, checkpoints / f"{name}.pt")
        torch.save(
            {"method_version": 2, "state_dict": policy_scores[name].state_dict()},
            checkpoints / f"{name}_score.pt",
        )
    return policies


def load_policies(config: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    _, checkpoints = _paths(config)
    names = [name for name, enabled in config["policies"].items() if enabled]
    if "baseline" not in names:
        names.insert(0, "baseline")
    policies: dict[str, torch.nn.Module] = {}
    for name in names:
        path = checkpoints / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint {path}; run the train command first")
        payload = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(payload, dict) or payload.get("method_version") != 2:
            raise RuntimeError(f"Checkpoint {path} predates the current-score DGPO method; run the train command again")
        model = make_policy(config).to(device)
        model.load_state_dict(payload["state_dict"])
        policies[name] = model.eval()
    return policies


def _calibration_for_policy(
    policy: torch.nn.Module,
    response_features: torch.Tensor,
    response_truth: torch.Tensor,
    config: dict[str, Any],
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unfolding = config["unfolding"]
    sigma = float(config["model"]["policy_sigma"])
    reco = reconstruct(policy, response_features, sigma, generator).cpu().numpy()
    truth = response_truth.cpu().numpy()
    truth_edges = np.linspace(-1.0, 1.0, int(unfolding["truth_bins"]) + 1)
    reco_edges = np.linspace(-1.0, 1.0, int(unfolding["reco_bins"]) + 1)
    response, prior = response_matrix(truth, reco, truth_edges, reco_edges)
    return truth, reco, truth_edges, reco_edges, response, prior


def _evaluate_pseudo_dataset(
    reco: np.ndarray,
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    config: dict[str, Any],
) -> tuple[float, float, float]:
    nominal_truth, nominal_reco, _, reco_edges, _, _ = calibration
    counts, _ = np.histogram(reco, bins=reco_edges)
    settings = config["unfolding"]
    bounds = config["physics"]["physical_C_range"]
    scan = np.linspace(float(bounds[0]), float(bounds[1]), int(settings["scan_points"]))
    templates = reweighted_reco_templates(
        nominal_truth,
        nominal_reco,
        reco_edges,
        float(config["physics"]["nominal_C"]),
        scan,
        float(counts.sum()),
    )
    estimate, lower_error, upper_error, _ = fit_poisson_parameter(counts, templates, scan)
    return estimate, lower_error, upper_error


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    keys = sorted({(str(row["policy"]), float(row["C_true"])) for row in rows})
    for policy, C_true in keys:
        selected = [row for row in rows if row["policy"] == policy and float(row["C_true"]) == C_true]
        estimates = np.array([float(row["C_hat"]) for row in selected])
        pulls = np.array([float(row["pull"]) for row in selected])
        bias = float(estimates.mean() - C_true)
        standard_deviation = float(estimates.std(ddof=1))
        normalized_bias = bias / standard_deviation if standard_deviation > 0.0 else None
        coverage = float(np.mean([bool(row["covered_68"]) for row in selected]))
        count = estimates.size
        denominator = 1.0 + 1.0 / count
        coverage_center = (coverage + 0.5 / count) / denominator
        coverage_half_width = np.sqrt(coverage * (1.0 - coverage) / count + 0.25 / count**2) / denominator
        summaries.append(
            {
                "inference_method": "poisson_forward_folding",
                "policy": policy,
                "C_true": C_true,
                "mean_C_hat": float(estimates.mean()),
                "mean_C_hat_error": standard_deviation / np.sqrt(estimates.size),
                "bias": bias,
                "normalized_bias": normalized_bias,
                "std_C_hat": standard_deviation,
                "rmse": float(np.sqrt(bias**2 + standard_deviation**2)),
                "mean_reported_sigma": float(np.mean([float(row["sigma_C_hat"]) for row in selected])),
                "pull_mean": float(pulls.mean()),
                "pull_std": float(pulls.std(ddof=1)),
                "coverage_68": coverage,
                "coverage_68_low": coverage_center - coverage_half_width,
                "coverage_68_high": coverage_center + coverage_half_width,
                "pseudo_experiments": estimates.size,
            }
        )
    return summaries


def evaluate_pipeline(
    config: dict[str, Any], device: torch.device, policies: dict[str, torch.nn.Module] | None = None
) -> list[dict[str, Any]]:
    output, _ = _paths(config)
    _save_config(config, output)
    policies = policies or load_policies(config, device)
    features, truth = _nominal_sample(config, device)
    _, _, response_data = _split_nominal(features, truth, config)
    response_features, response_truth = response_data
    sigma = float(config["model"]["policy_sigma"])
    base_seed = int(config["seed"])
    events = int(config["data"]["events_per_pseudo_experiment"])
    experiments = int(config["data"]["pseudo_experiments"])
    pseudo_batch = int(config["data"]["pseudo_batch_size"])
    rows: list[dict[str, Any]] = []
    calibrations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    total = len(policies) * len(config["physics"]["true_C_values"]) * experiments
    progress = tqdm(total=total, desc="closure pseudo-experiments", unit="toy")
    for policy_index, (name, policy) in enumerate(policies.items()):
        calibration_generator = make_generator(device, base_seed + 1000 + policy_index)
        calibration = _calibration_for_policy(
            policy, response_features, response_truth, config, calibration_generator
        )
        calibrations[name] = calibration
        for truth_index, C_true_value in enumerate(config["physics"]["true_C_values"]):
            C_true = float(C_true_value)
            generator = make_generator(device, base_seed + 100000 * (policy_index + 1) + 1000 * truth_index)
            for start in range(0, experiments, pseudo_batch):
                batch_count = min(pseudo_batch, experiments - start)
                batch_truth = sample_truth(batch_count * events, C_true, device, generator)
                batch_features = detector_features(batch_truth, config["physics"], generator)
                batch_reco = reconstruct(policy, batch_features, sigma, generator).reshape(batch_count, events).cpu().numpy()
                for offset, reco in enumerate(batch_reco):
                    estimate, lower_error, upper_error = _evaluate_pseudo_dataset(reco, calibration, config)
                    symmetric_error = 0.5 * (lower_error + upper_error)
                    rows.append(
                        {
                            "inference_method": "poisson_forward_folding",
                            "policy": name,
                            "C_true": C_true,
                            "pseudo_experiment": start + offset,
                            "C_hat": estimate,
                            "sigma_minus": lower_error,
                            "sigma_plus": upper_error,
                            "sigma_C_hat": symmetric_error,
                            "pull": (estimate - C_true) / symmetric_error if symmetric_error > 0.0 else np.nan,
                            "covered_68": estimate - lower_error <= C_true <= estimate + upper_error,
                        }
                    )
                progress.update(batch_count)
                progress.set_postfix(policy=name, C_true=f"{C_true:.2f}")
    progress.close()

    summaries = _summarize(rows)
    _write_rows(output / "pseudo_experiments.csv", rows)
    _write_rows(output / "summary.csv", summaries)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summaries, stream, indent=2)
    make_diagnostics(policies, calibrations, config, device, output)
    make_plots(rows, summaries, config, output)
    return summaries


def _policy_label(name: str) -> str:
    return name.replace("_", " ").title().replace("Dgpo", "DGPO")


def make_plots(
    rows: list[dict[str, Any]], summaries: list[dict[str, Any]], config: dict[str, Any], output: Path
) -> None:
    dpi = int(config["plots"]["dpi"])
    policies = list(dict.fromkeys(str(row["policy"]) for row in rows))
    colors = dict(zip(policies, plt.get_cmap("tab10").colors))
    C_values = np.array(sorted({float(row["C_true"]) for row in rows}))

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for policy in policies:
        selected = sorted((row for row in summaries if row["policy"] == policy), key=lambda row: row["C_true"])
        axis.errorbar(
            [row["C_true"] for row in selected],
            [row["mean_C_hat"] for row in selected],
            yerr=[row["mean_C_hat_error"] for row in selected],
            marker="o",
            capsize=3,
            label=_policy_label(policy),
            color=colors[policy],
        )
    axis.plot(C_values, C_values, "--", color="black", label="Identity")
    axis.set(xlabel=r"$C_{\mathrm{true}}$", ylabel=r"$E[\hat C]$")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "12_bias_linearity.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True)
    for policy in policies:
        selected = sorted((row for row in summaries if row["policy"] == policy), key=lambda row: row["C_true"])
        axes[0].plot(C_values, [row["bias"] for row in selected], marker="o", label=_policy_label(policy), color=colors[policy])
        normalized = [np.nan if row["normalized_bias"] is None else row["normalized_bias"] for row in selected]
        axes[1].plot(C_values, normalized, marker="o", color=colors[policy])
    axes[0].axhline(0.0, color="black", linestyle="--")
    axes[1].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel(r"$E[\hat C]-C_{\mathrm{true}}$")
    axes[1].set(xlabel=r"$C_{\mathrm{true}}$", ylabel="Normalized bias")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "13_bias_vs_Ctrue.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(C_values), figsize=(3.4 * len(C_values), 3.4), sharey=True, squeeze=False)
    pull_range = tuple(float(value) for value in config["plots"]["pull_range"])
    bins = np.linspace(pull_range[0], pull_range[1], int(config["plots"]["pull_bins"]) + 1)
    for axis, C_true in zip(axes[0], C_values):
        for policy in policies:
            pulls = np.array([row["pull"] for row in rows if row["policy"] == policy and row["C_true"] == C_true])
            finite = pulls[np.isfinite(pulls)]
            label = (
                f"{_policy_label(policy)}\n$\\mu={np.mean(finite):.2f}$, $\\sigma={np.std(finite, ddof=1):.2f}$"
                if finite.size > 1
                else f"{_policy_label(policy)}\nPull unavailable"
            )
            visible = finite[(finite >= bins[0]) & (finite <= bins[-1])]
            if visible.size:
                axis.hist(visible, bins=bins, histtype="step", density=True, linewidth=1.5, label=label, color=colors[policy])
            else:
                axis.plot([], [], color=colors[policy], label=label)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set_title(rf"$C_{{\mathrm{{true}}}}={C_true:.2f}$")
        axis.set_xlabel("Pull")
        axis.legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output / "14_pull_distributions.png", dpi=dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for policy in policies:
        selected = sorted((row for row in summaries if row["policy"] == policy), key=lambda row: row["C_true"])
        coverage = np.array([row["coverage_68"] for row in selected])
        lower = coverage - np.array([row["coverage_68_low"] for row in selected])
        upper = np.array([row["coverage_68_high"] for row in selected]) - coverage
        axis.errorbar(C_values, coverage, yerr=np.vstack((lower, upper)), marker="o", capsize=3, label=_policy_label(policy), color=colors[policy])
    axis.axhline(0.68, color="black", linestyle="--", label="68% reference")
    axis.set(xlabel=r"$C_{\mathrm{true}}$", ylabel="Empirical 68% coverage", ylim=(0.0, 1.0))
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "15_coverage_vs_Ctrue.png", dpi=dpi)
    plt.close(fig)

    representatives = [float(value) for value in config["plots"]["representative_C_values"]]
    fig, axes = plt.subplots(1, len(representatives), figsize=(6.0 * len(representatives), 4.0), squeeze=False)
    estimate_bins = int(config["plots"]["estimate_bins"])
    nominal_C = float(config["physics"]["nominal_C"])
    for axis, C_true in zip(axes[0], representatives):
        for policy in policies:
            estimates = [row["C_hat"] for row in rows if row["policy"] == policy and row["C_true"] == C_true]
            axis.hist(estimates, bins=estimate_bins, histtype="step", density=True, linewidth=1.5, label=_policy_label(policy), color=colors[policy])
        axis.axvline(C_true, color="black", linewidth=1.5, label=r"$C_{\mathrm{true}}$")
        axis.axvline(nominal_C, color="black", linestyle="--", linewidth=1.5, label=r"Nominal $C_0$")
        axis.set(xlabel=r"$\hat C$", ylabel="Density", title=rf"$C_{{\mathrm{{true}}}}={C_true:.2f}$")
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "16_C_hat_offnominal.png", dpi=dpi)
    plt.close(fig)


def run(config_path: str | Path, mode: str, device_override: str | None, output_override: str | None) -> None:
    config = load_config(config_path)
    if device_override is not None:
        config["device"] = device_override
    if output_override is not None:
        config["output_dir"] = output_override
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config.get("device", "auto")))
    print(f"Using device: {device}", flush=True)
    policies = train_pipeline(config, device) if mode in {"run", "train"} else None
    if mode in {"run", "evaluate"}:
        summaries = evaluate_pipeline(config, device, policies)
        print(f"Results written to {Path(config['output_dir']).expanduser().resolve()}", flush=True)
        for row in summaries:
            print(
                f"{row['policy']:>24s} C_true={row['C_true']:.2f}: "
                f"ensemble mean={row['mean_C_hat']:.4f}, ensemble std={row['std_C_hat']:.4f}, "
                f"bias={row['bias']:+.4f}, coverage={row['coverage_68']:.3f}",
                flush=True,
            )
