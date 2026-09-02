from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .core import ScoreModel, make_generator, truth_score
from .flow import ConditionalFlow
from .inference import binned_fisher_per_event
from .training import reconstruct_policy, slice_events
from .ztautau import generate_events


def _label(name: str) -> str:
    return {
        "baseline": "Baseline",
        "fisher_dgpo_no_trust": "No trust",
        "fisher_dgpo_trust": "Trust",
    }.get(name, name.replace("_", " ").title())


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _train_diagnostic_score(
    policy_name: str,
    reconstructed: dict[str, torch.Tensor],
    events: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
) -> ScoreModel:
    settings = config["training"]
    model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]),
    )
    target = truth_score(events["x"], float(config["physics"]["nominal_C"]))
    valid_indices = torch.nonzero(reconstructed["valid"], as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        raise RuntimeError(f"Policy {policy_name} produced no valid diagnostic-score training events")
    batch_size = int(settings["batch_size"])
    progress = tqdm(range(int(config["diagnosis"]["score_epochs"])), desc=f"diagnostic score: {_label(policy_name)}", unit="epoch")
    for _ in progress:
        order = valid_indices[torch.randperm(valid_indices.numel(), device=device)]
        total = 0.0
        for start in range(0, order.numel(), batch_size):
            index = order[start : start + batch_size]
            loss = (model(reconstructed["y"][index]) - target[index]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * index.numel()
        progress.set_postfix(mse=f"{total / order.numel():.5f}", valid=f"{valid_indices.numel() / events['x'].numel():.3f}")
    return model.eval()


def _binned_mean(values: np.ndarray, indices: np.ndarray, bins: int, selected: np.ndarray | None = None) -> np.ndarray:
    result = np.full(bins, np.nan)
    mask = np.ones(values.size, dtype=bool) if selected is None else selected
    for index in range(bins):
        in_bin = mask & (indices == index)
        if np.any(in_bin):
            result[index] = float(np.mean(values[in_bin]))
    return result


def _read_summary(output: Path) -> list[dict[str, Any]]:
    path = output / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run evaluate with the frozen policies before diagnose")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append({key: value if key == "policy" else float(value) if value else float("nan") for key, value in row.items()})
    return rows


def _plot_fisher_decomposition(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = [row["policy"] for row in metrics]
    fields = [
        ("valid_efficiency", "Valid efficiency"),
        ("frozen_mean_information_valid", r"$E[s_{ref}^2\mid valid]$"),
        ("frozen_fisher_per_event", r"$I_{frozen}/N$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, (field, title) in zip(axes, fields):
        values = [row[field] for row in metrics]
        axis.bar(range(len(names)), values)
        axis.set(xticks=range(len(names)), xticklabels=[_label(name) for name in names], ylabel=title)
        axis.tick_params(axis="x", rotation=15)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Frozen-score Fisher decomposition per generated event")
    fig.tight_layout()
    fig.savefig(output / "09_fisher_decomposition.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_policy_scores(
    names: list[str], grid: np.ndarray, reference: np.ndarray, curves: dict[str, np.ndarray],
    config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(len(names), 2, figsize=(11.0, 3.4 * len(names)), squeeze=False, sharex=True)
    for row, name in enumerate(names):
        axes[row, 0].plot(grid, reference, color="black", linestyle="--", label=r"Frozen $s_{ref}$")
        axes[row, 0].plot(grid, curves[name], color="tab:blue", label=rf"Diagnostic $s_\pi$: {_label(name)}")
        axes[row, 0].set(ylabel="Score")
        axes[row, 0].legend(frameon=False, fontsize=8)
        axes[row, 1].plot(grid, curves[name] - reference, color="tab:red")
        axes[row, 1].axhline(0.0, color="black", linewidth=0.7)
        axes[row, 1].set(ylabel=r"$s_\pi-s_{ref}$")
    axes[-1, 0].set_xlabel("Reconstructed y")
    axes[-1, 1].set_xlabel("Reconstructed y")
    fig.suptitle("Independent policy-specific reconstructed scores")
    fig.tight_layout()
    fig.savefig(output / "10_policy_specific_scores.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_frozen_vs_specific(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = [row["policy"] for row in metrics]
    positions = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.bar(positions - width / 2, [100.0 * row["frozen_fisher_gain"] for row in metrics], width, label="Frozen-score Fisher gain")
    axis.bar(positions + width / 2, [100.0 * row["specific_fisher_gain"] for row in metrics], width, label="Policy-score Fisher gain")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel="Gain relative to baseline [%]")
    axis.legend(frameon=False)
    axis.set_title("Frozen-score surrogate versus independently estimated score")
    fig.tight_layout()
    fig.savefig(output / "11_frozen_vs_policy_score_fisher.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_policy_fisher_closure(
    metrics: list[dict[str, Any]], bin_counts: list[int], binned: dict[str, list[float]],
    config: dict[str, Any], output: Path,
) -> None:
    names = [row["policy"] for row in metrics]
    positions = np.arange(len(names))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    axes[0].bar(positions - width, [row["frozen_fisher_per_event"] for row in metrics], width, label="Frozen score")
    axes[0].bar(positions, [row["specific_fisher_per_event"] for row in metrics], width, label="Policy score")
    axes[0].bar(positions + width, [binned[name][-1] for name in names], width, label=f"Binned ({bin_counts[-1]} bins)")
    axes[0].set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel="Fisher per generated event")
    axes[0].legend(frameon=False)
    for name in names:
        axes[1].plot(bin_counts, binned[name], marker="o", label=_label(name))
    axes[1].set(xlabel="Reconstructed-y bins", ylabel="Direct binned Fisher per event")
    axes[1].legend(frameon=False)
    fig.suptitle("Policy-specific Fisher closure")
    fig.tight_layout()
    fig.savefig(output / "12_policy_specific_fisher_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_fisher_vs_precision(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = [row["policy"] for row in metrics]
    positions = np.arange(len(names))
    fields = [
        ("frozen_sigma_ratio", "Frozen score"),
        ("specific_sigma_ratio", "Policy score"),
        ("binned_sigma_ratio", "Fine-binned"),
        ("actual_sigma_ratio", "Pseudo-experiments"),
    ]
    width = 0.19
    fig, axis = plt.subplots(figsize=(10.0, 5.2))
    for index, (field, label) in enumerate(fields):
        offset = (index - 1.5) * width
        axis.bar(positions + offset, [row[field] for row in metrics], width, label=label)
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel=r"$\sigma_\pi/\sigma_{baseline}$", title="Fisher prediction versus actual nominal precision")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "13_fisher_vs_actual_precision.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_score_balance(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = [row["policy"] for row in metrics]
    positions = np.arange(len(names))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].bar(positions - width / 2, [row["frozen_b_local"] for row in metrics], width, label="Frozen score")
    axes[0].bar(positions + width / 2, [row["specific_b_local"] for row in metrics], width, label="Policy score")
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel=r"$U/I$")
    axes[1].bar(positions - width / 2, [row["frozen_Z_bias"] for row in metrics], width, label="Frozen score")
    axes[1].bar(positions + width / 2, [row["specific_Z_bias"] for row in metrics], width, label="Policy score")
    axes[1].set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel=r"$|U|/\sqrt{I}$")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Extended-Poisson compensated score balance")
    fig.tight_layout()
    fig.savefig(output / "14_score_balance_diagnostics.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_phase_space(
    names: list[str], centers: np.ndarray, phase: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any], output: Path,
) -> None:
    fields = [
        ("efficiency", r"$\epsilon_{valid}(x)$"),
        ("frozen_information", r"$E[s_{ref}^2 1_{valid}\mid x]$"),
        ("specific_information", r"$E[s_\pi^2 1_{valid}\mid x]$"),
        ("mean_y_valid", r"$E[Y\mid x,valid]$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True)
    for axis, (field, ylabel) in zip(axes.flat, fields):
        for name in names:
            axis.plot(centers, phase[name][field], marker="o", markersize=3, label=_label(name))
        axis.set(ylabel=ylabel)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Truth x (diagnostic only)")
    axes[1, 1].set_xlabel("Truth x (diagnostic only)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where DGPO changes efficiency and information")
    fig.tight_layout()
    fig.savefig(output / "15_phase_space_information_shift.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_kl_phase_space(
    names: list[str], x_centers: np.ndarray, score_centers: np.ndarray,
    phase: dict[str, dict[str, np.ndarray]], kl_status: dict[str, tuple[float, float]],
    config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for name in names:
        axes[0].plot(x_centers, phase[name]["kl_x"], marker="o", markersize=3, label=_label(name))
        axes[1].plot(score_centers, phase[name]["kl_score"], marker="o", markersize=3, label=_label(name))
    positions = np.arange(len(names))
    width = 0.36
    axes[2].bar(positions - width / 2, [kl_status[name][0] for name in names], width, label="Valid")
    axes[2].bar(positions + width / 2, [kl_status[name][1] for name in names], width, label="Invalid")
    axes[0].set(xlabel="Truth x (diagnostic only)", ylabel=r"Conditional $D_{KL}$")
    axes[1].set(xlabel=r"$|s_{ref}(y)|$", ylabel=r"Conditional $D_{KL}$")
    axes[2].set(xticks=positions, xticklabels=[_label(name) for name in names], ylabel=r"Mean $D_{KL}$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False)
    fig.suptitle("Local policy drift hidden by a global KL average")
    fig.tight_layout()
    fig.savefig(output / "16_kl_phase_space.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_offnominal(summary: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = sorted({row["policy"] for row in summary})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8))
    for name in names:
        rows = sorted((row for row in summary if row["policy"] == name), key=lambda row: row["C_true"])
        truth = np.array([row["C_true"] for row in rows])
        axes[0, 0].errorbar(truth, [row["mean_C_hat"] for row in rows], yerr=[row["mean_C_hat_error"] for row in rows], marker="o", capsize=3, label=_label(name))
        axes[0, 1].plot(truth, [row["bias"] for row in rows], marker="o", label=_label(name))
        axes[1, 0].plot(truth, [row["pull_mean"] for row in rows], marker="o", label=_label(name))
        axes[1, 1].plot(truth, [row["coverage_68"] for row in rows], marker="o", label=_label(name))
    values = [float(value) for value in config["physics"]["true_C_values"]]
    axes[0, 0].plot((min(values), max(values)), (min(values), max(values)), "k--", label="Identity")
    axes[0, 1].axhline(0.0, color="black", linestyle="--")
    axes[1, 0].axhline(0.0, color="black", linestyle="--")
    axes[1, 1].axhline(0.68, color="black", linestyle="--")
    axes[0, 0].set(xlabel=r"$C_{true}$", ylabel=r"$E[\hat C]$", title="Mean estimate")
    axes[0, 1].set(xlabel=r"$C_{true}$", ylabel="Bias", title="Bias")
    axes[1, 0].set(xlabel=r"$C_{true}$", ylabel="Pull mean", title="Pull mean")
    axes[1, 1].set(xlabel=r"$C_{true}$", ylabel="68% coverage", title="Coverage", ylim=(0, 1))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Detailed off-nominal closure with unchanged frozen policies")
    fig.tight_layout()
    fig.savefig(output / "17_offnominal_bias_detailed.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_dashboard(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    columns = [
        "Frozen I gain", "Policy I gain", "Binned I gain", "PE gain", "Valid eff.",
        "Mean frozen s2|valid", "KL", "Max |bias|", "Mean RMSE", "Mean coverage",
    ]
    cells = [[
        f"{100 * row['frozen_fisher_gain']:+.1f}%", f"{100 * row['specific_fisher_gain']:+.1f}%",
        f"{100 * row['binned_fisher_gain']:+.1f}%", f"{100 * row['actual_precision_gain']:+.1f}%",
        f"{row['valid_efficiency']:.3f}", f"{row['frozen_mean_information_valid']:.4g}",
        f"{row['global_kl']:.4g}", f"{row['max_abs_bias']:.4f}", f"{row['mean_rmse']:.4f}",
        f"{row['mean_coverage']:.3f}",
    ] for row in metrics]
    fig, axis = plt.subplots(figsize=(16.0, 4.5))
    axis.axis("off")
    table = axis.table(cellText=cells, rowLabels=[_label(row["policy"]) for row in metrics], colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    axis.set_title("Reward-hacking diagnosis: efficiency, real information, and stale-surrogate gain", pad=20)
    fig.tight_layout()
    fig.savefig(output / "18_reward_hacking_diagnosis.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _write_diagnosis(
    metrics: list[dict[str, Any]], score_curve_rms: dict[str, float],
    binned: dict[str, list[float]], bin_counts: list[int], summary: list[dict[str, Any]],
    phase: dict[str, dict[str, np.ndarray]], config: dict[str, Any], output: Path,
) -> None:
    baseline = next(row for row in metrics if row["policy"] == "baseline")
    optimized = [row for row in metrics if row["policy"] != "baseline"]
    lines = ["# Frozen-score Fisher-DGPO diagnosis", ""]
    if min(row["pseudo_experiments"] for row in summary) < 50:
        lines.extend(["> Warning: this artifact uses fewer than 50 pseudo-experiments per point. Numerical conclusions are software-smoke diagnostics, not a physics result.", ""])

    lines.extend(["## 1. Fisher decomposition", ""])
    for row in optimized:
        delta = row["frozen_fisher_per_event"] - baseline["frozen_fisher_per_event"]
        efficiency_term = (row["valid_efficiency"] - baseline["valid_efficiency"]) * baseline["frozen_mean_information_valid"]
        information_term = row["valid_efficiency"] * (row["frozen_mean_information_valid"] - baseline["frozen_mean_information_valid"])
        if abs(delta) > 1.0e-12:
            shares = f"efficiency {100 * efficiency_term / delta:+.1f}%, per-valid information {100 * information_term / delta:+.1f}%"
        else:
            shares = "the total change is numerically negligible"
        lines.append(f"- **{_label(row['policy'])}:** efficiency {row['valid_efficiency']:.4f} versus {baseline['valid_efficiency']:.4f}; exact additive decomposition: {shares}.")

    lines.extend(["", "## 2. Fisher after policy-specific score estimation", ""])
    for row in optimized:
        lines.append(f"- **{_label(row['policy'])}:** frozen-score gain {100 * row['frozen_fisher_gain']:+.2f}%; policy-score gain {100 * row['specific_fisher_gain']:+.2f}%; score-curve RMS shift {score_curve_rms[row['policy']]:.4g}.")

    lines.extend(["", "## 3. Policy score versus direct binned Fisher", ""])
    score_tolerance = float(config["diagnosis"]["score_fisher_relative_tolerance"])
    score_closed: dict[str, bool] = {}
    for row in metrics:
        fine = binned[row["policy"]][-1]
        difference = abs(row["specific_fisher_per_event"] - fine) / max(fine, 1.0e-12)
        score_closed[row["policy"]] = difference <= score_tolerance
        status = "PASS" if score_closed[row["policy"]] else "FAIL"
        lines.append(f"- **{_label(row['policy'])}:** policy-score Fisher {row['specific_fisher_per_event']:.6g}; {bin_counts[-1]}-bin Fisher {fine:.6g}; relative difference {100 * difference:.1f}% -- **{status}** at the configured {100 * score_tolerance:.0f}% tolerance.")

    prediction_fields = [
        ("frozen_sigma_ratio", "frozen score"),
        ("specific_sigma_ratio", "policy-specific score"),
        ("binned_sigma_ratio", "fine-binned Fisher"),
    ]
    errors = {
        label: float(np.mean([abs(np.log(max(row[field], 1.0e-12) / max(row["actual_sigma_ratio"], 1.0e-12))) for row in optimized]))
        for field, label in prediction_fields
    }
    best_prediction = min(errors, key=errors.get)
    lines.extend(["", "## 4. Prediction of pseudo-experiment precision", "", f"The closest relative prediction across optimized policies is **{best_prediction}** (mean absolute log-ratio error {errors[best_prediction]:.3f}).", ""])

    stale_threshold = float(config["diagnosis"]["stale_fisher_gain_gap_threshold"])
    stale = []
    inconclusive = []
    for row in optimized:
        excess = row["frozen_fisher_gain"] - row["specific_fisher_gain"]
        valid_diagnosis = score_closed[row["policy"]]
        stale.append(valid_diagnosis and excess > stale_threshold)
        inconclusive.append(not valid_diagnosis)
    if any(stale):
        stale_conclusion = f"At least one score-validated optimized policy loses more than {100 * stale_threshold:.1f} percentage points of Fisher gain after policy-specific score estimation, which is direct evidence of frozen-score surrogate exploitation."
    elif any(inconclusive):
        stale_conclusion = "Frozen-score exploitation is inconclusive because at least one policy-specific score regression fails the direct binned-Fisher closure gate. Fix the diagnostic estimator before changing the method."
    else:
        stale_conclusion = f"Neither optimized policy loses more than {100 * stale_threshold:.1f} percentage points of Fisher gain after validated policy-specific score estimation; this test does not establish frozen-score exploitation."
    lines.extend(["## 5. Frozen-score exploitation", "", stale_conclusion, ""])

    concentration_threshold = float(config["diagnosis"]["kl_concentration_ratio_threshold"])
    concentration = []
    for row in optimized:
        local_max = float(np.nanmax(np.abs(phase[row["policy"]]["kl_x"])))
        ratio = local_max / max(abs(row["global_kl"]), 1.0e-12)
        concentration.append(ratio > concentration_threshold)
        lines.append(f"- {_label(row['policy'])}: global KL {row['global_kl']:.4g}, maximum absolute truth-bin KL {local_max:.4g}, concentration ratio {ratio:.2f}.")
    lines.extend(["", "## 6. Local versus global KL", "", "Global KL hides concentrated phase-space drift." if any(concentration) else f"No greater-than-{concentration_threshold:g}-fold KL concentration is established.", ""])

    lines.extend(["## 7. Extended-Poisson score balance versus fitted bias", ""])
    lines.extend(["The full score is U_full = sum_i s(y_i) - Lambda'(C0). The intensity-score Fisher already contains rate and shape information; no separate rate-Fisher term is added.", ""])
    nominal_C = float(config["physics"]["nominal_C"])
    for row in metrics:
        nominal = min((item for item in summary if item["policy"] == row["policy"]), key=lambda item: abs(item["C_true"] - nominal_C))
        lines.append(f"- **{_label(row['policy'])}:** frozen b_local {row['frozen_b_local']:+.4f}, policy b_local {row['specific_b_local']:+.4f}, nominal fitted bias {nominal['bias']:+.4f}, maximum off-nominal |bias| {row['max_abs_bias']:.4f}.")
    largest_frozen_b = max(optimized, key=lambda row: abs(row["frozen_b_local"]))["policy"]
    largest_frozen_z = max(optimized, key=lambda row: row["frozen_Z_bias"])["policy"]
    largest_observed_bias = max(optimized, key=lambda row: row["max_abs_bias"])["policy"]
    balance_predicts = largest_frozen_b == largest_frozen_z == largest_observed_bias
    if balance_predicts:
        balance_conclusion = f"Yes: among the optimized policies, both frozen balance measures rank **{_label(largest_observed_bias)}** as the policy with the larger observed off-nominal bias."
    else:
        balance_conclusion = f"No: among the optimized policies, frozen |b_local| ranks **{_label(largest_frozen_b)}**, frozen Z_bias ranks **{_label(largest_frozen_z)}**, while the larger observed off-nominal bias belongs to **{_label(largest_observed_bias)}**."
    lines.extend(["", balance_conclusion, ""])

    lines.extend(["## 8. Statistical closure requirement", "", "No training-method change is justified until the direct-versus-reweighted high-statistics Asimov closure passes.", ""])
    (output / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def run_diagnosis(
    config: dict[str, Any], device: torch.device, policies: dict[str, ConditionalFlow],
    reference_score: ScoreModel, output: Path,
) -> None:
    names = [name for name in ("baseline", "fisher_dgpo_no_trust", "fisher_dgpo_trust") if name in policies]
    if len(names) != 3:
        raise RuntimeError("Diagnosis requires baseline, fisher_dgpo_no_trust, and fisher_dgpo_trust checkpoints")
    settings = config["diagnosis"]
    count = int(settings["nominal_events"])
    fraction = float(settings["score_train_fraction"])
    split = int(count * fraction)
    if not 0 < split < count:
        raise ValueError("diagnosis.score_train_fraction must leave non-empty train and evaluation samples")
    nominal_C = float(config["physics"]["nominal_C"])
    all_events = generate_events(count, nominal_C, config, device, make_generator(device, int(config["seed"]) + 80000))
    score_events = slice_events(all_events, slice(0, split))
    evaluation_events = slice_events(all_events, slice(split, count))

    score_reconstruction = {
        name: reconstruct_policy(policies[name], score_events, config, make_generator(device, int(config["seed"]) + 81000))
        for name in names
    }
    score_models: dict[str, ScoreModel] = {}
    diagnostic_score_dir = output / "diagnostic_scores"
    diagnostic_score_dir.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        torch.manual_seed(int(config["seed"]) + 81100 + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]) + 81100 + index)
        score_models[name] = _train_diagnostic_score(name, score_reconstruction[name], score_events, config, device)
        torch.save({"method_version": 3, "diagnostic_only": True, "state_dict": score_models[name].state_dict()}, diagnostic_score_dir / f"{name}.pt")

    reconstruction = {
        name: reconstruct_policy(policies[name], evaluation_events, config, make_generator(device, int(config["seed"]) + 82000))
        for name in names
    }
    x = _numpy(evaluation_events["x"])
    truth_x_bins = int(settings["truth_x_bins"])
    x_edges = np.linspace(-1.0, 1.0, truth_x_bins + 1)
    x_centers = 0.5 * (x_edges[1:] + x_edges[:-1])
    x_indices = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, truth_x_bins - 1)
    bin_counts = [int(value) for value in settings["fisher_reco_bins"]]
    summary = _read_summary(output)
    nominal_summary = {
        name: min((row for row in summary if row["policy"] == name), key=lambda row: abs(row["C_true"] - nominal_C))
        for name in names
    }
    baseline_actual_sigma = nominal_summary["baseline"]["std_C_hat"]
    if baseline_actual_sigma <= 0.0:
        raise RuntimeError("Baseline nominal pseudo-experiment width must be positive for the precision comparison")

    grid_tensor = torch.linspace(-1.0, 1.0, int(settings["score_grid_points"]), device=device)
    grid = _numpy(grid_tensor)
    with torch.no_grad():
        reference_curve = _numpy(reference_score(grid_tensor))
        policy_curves = {name: _numpy(score_models[name](grid_tensor)) for name in names}
    score_curve_rms: dict[str, float] = {}

    data: dict[str, dict[str, Any]] = {}
    binned: dict[str, list[float]] = {}
    phase: dict[str, dict[str, np.ndarray]] = {}
    kl_status: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for name in names:
            result = reconstruction[name]
            valid_tensor = result["valid"]
            raw_frozen_tensor = reference_score(result["y"])
            frozen_tensor = raw_frozen_tensor * valid_tensor
            specific_tensor = score_models[name](result["y"]) * valid_tensor
            kl_tensor = result["log_prob"] - policies["baseline"].log_prob(result["action"], evaluation_events["context"])
            valid = _numpy(valid_tensor).astype(bool)
            y = _numpy(result["y"])
            raw_frozen = _numpy(raw_frozen_tensor)
            frozen = _numpy(frozen_tensor)
            specific = _numpy(specific_tensor)
            kl = _numpy(kl_tensor)
            frozen_information = frozen**2
            specific_information = specific**2
            score_curve_rms[name] = float(np.sqrt(np.mean((specific[valid] - frozen[valid]) ** 2)))
            frozen_I = float(frozen_information.sum())
            specific_I = float(specific_information.sum())
            frozen_score_sum = float(frozen.sum())
            specific_score_sum = float(specific.sum())
            yield_derivative = float(np.sum(valid * x / (1.0 + nominal_C * x)))
            frozen_U = frozen_score_sum - yield_derivative
            specific_U = specific_score_sum - yield_derivative
            data[name] = {
                "valid": valid, "y": y, "raw_frozen": raw_frozen,
                "frozen": frozen, "specific": specific, "kl": kl,
                "efficiency": float(valid.mean()),
                "frozen_mean_valid": float(frozen_information[valid].mean()),
                "specific_mean_valid": float(specific_information[valid].mean()),
                "frozen_fisher": frozen_I / x.size,
                "specific_fisher": specific_I / x.size,
                "frozen_score_sum": frozen_score_sum,
                "specific_score_sum": specific_score_sum,
                "yield_derivative": yield_derivative,
                "frozen_U_full": frozen_U, "specific_U_full": specific_U,
                "frozen_b": frozen_U / frozen_I,
                "specific_b": specific_U / specific_I,
                "frozen_Z": abs(frozen_U) / np.sqrt(frozen_I),
                "specific_Z": abs(specific_U) / np.sqrt(specific_I),
                "global_kl": float(kl.mean()),
            }
            binned[name] = [
                binned_fisher_per_event(x, y, valid, np.linspace(-1.0, 1.0, bins + 1), nominal_C)
                for bins in bin_counts
            ]
            phase[name] = {
                "efficiency": _binned_mean(valid.astype(float), x_indices, truth_x_bins),
                "frozen_information": _binned_mean(frozen_information, x_indices, truth_x_bins),
                "specific_information": _binned_mean(specific_information, x_indices, truth_x_bins),
                "mean_y_valid": _binned_mean(y, x_indices, truth_x_bins, valid),
                "kl_x": _binned_mean(kl, x_indices, truth_x_bins),
            }
            kl_status[name] = (
                float(kl[valid].mean()) if np.any(valid) else float("nan"),
                float(kl[~valid].mean()) if np.any(~valid) else float("nan"),
            )

    score_bins = int(settings["score_magnitude_bins"])
    maximum_raw_score = max(float(np.max(np.abs(data[name]["raw_frozen"]))) for name in names)
    score_edges = np.linspace(0.0, max(maximum_raw_score, 1.0e-6), score_bins + 1)
    score_centers = 0.5 * (score_edges[1:] + score_edges[:-1])
    for name in names:
        magnitude = np.abs(data[name]["raw_frozen"])
        score_indices = np.clip(np.searchsorted(score_edges, magnitude, side="right") - 1, 0, score_bins - 1)
        phase[name]["kl_score"] = _binned_mean(data[name]["kl"], score_indices, score_bins)

    baseline_frozen = data["baseline"]["frozen_fisher"]
    baseline_specific = data["baseline"]["specific_fisher"]
    baseline_binned = binned["baseline"][-1]
    metrics: list[dict[str, Any]] = []
    for name in names:
        policy_summary = [row for row in summary if row["policy"] == name]
        frozen_gain = data[name]["frozen_fisher"] / baseline_frozen - 1.0
        specific_gain = data[name]["specific_fisher"] / baseline_specific - 1.0
        binned_gain = binned[name][-1] / baseline_binned - 1.0
        actual_sigma_ratio = nominal_summary[name]["std_C_hat"] / baseline_actual_sigma
        metrics.append({
            "policy": name,
            "evaluation_events": x.size,
            "valid_efficiency": data[name]["efficiency"],
            "frozen_mean_information_valid": data[name]["frozen_mean_valid"],
            "specific_mean_information_valid": data[name]["specific_mean_valid"],
            "frozen_fisher_per_event": data[name]["frozen_fisher"],
            "specific_fisher_per_event": data[name]["specific_fisher"],
            "fine_binned_fisher_per_event": binned[name][-1],
            "frozen_fisher_gain": frozen_gain,
            "specific_fisher_gain": specific_gain,
            "binned_fisher_gain": binned_gain,
            "frozen_sigma_ratio": np.sqrt(baseline_frozen / data[name]["frozen_fisher"]),
            "specific_sigma_ratio": np.sqrt(baseline_specific / data[name]["specific_fisher"]),
            "binned_sigma_ratio": np.sqrt(baseline_binned / binned[name][-1]),
            "actual_sigma_ratio": actual_sigma_ratio,
            "actual_precision_gain": 1.0 / actual_sigma_ratio - 1.0,
            "frozen_score_sum": data[name]["frozen_score_sum"],
            "specific_score_sum": data[name]["specific_score_sum"],
            "yield_derivative": data[name]["yield_derivative"],
            "frozen_U_full": data[name]["frozen_U_full"],
            "specific_U_full": data[name]["specific_U_full"],
            "frozen_b_local": data[name]["frozen_b"],
            "specific_b_local": data[name]["specific_b"],
            "frozen_Z_bias": data[name]["frozen_Z"],
            "specific_Z_bias": data[name]["specific_Z"],
            "global_kl": data[name]["global_kl"],
            "score_curve_rms_shift": score_curve_rms[name],
            "max_abs_bias": float(max(abs(row["bias"]) for row in policy_summary)),
            "mean_rmse": float(np.mean([row["rmse"] for row in policy_summary])),
            "mean_coverage": float(np.mean([row["coverage_68"] for row in policy_summary])),
        })

    with (output / "diagnosis_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "diagnosis_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({"policies": metrics, "fisher_bins": bin_counts, "binned_fisher": binned}, stream, indent=2)
    detail_fields = [
        "policy", "C_true", "mean_C_hat", "bias", "std_C_hat", "rmse",
        "pull_mean", "pull_std", "coverage_68", "pseudo_experiments",
    ]
    with (output / "17_offnominal_detailed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in detail_fields} for row in summary)

    _plot_fisher_decomposition(metrics, config, output)
    _plot_policy_scores(names, grid, reference_curve, policy_curves, config, output)
    _plot_frozen_vs_specific(metrics, config, output)
    _plot_policy_fisher_closure(metrics, bin_counts, binned, config, output)
    _plot_fisher_vs_precision(metrics, config, output)
    _plot_score_balance(metrics, config, output)
    _plot_phase_space(names, x_centers, phase, config, output)
    _plot_kl_phase_space(names, x_centers, score_centers, phase, kl_status, config, output)
    _plot_offnominal(summary, config, output)
    _plot_dashboard(metrics, config, output)
    _write_diagnosis(metrics, score_curve_rms, binned, bin_counts, summary, phase, config, output)
    print(f"Diagnostic report written to {output / 'diagnosis.md'}", flush=True)
