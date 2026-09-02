from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import ScoreModel, make_generator
from .flow import ConditionalFlow
from .training import make_flow, rebuild_policy_score, reconstruct_policy
from .ztautau import generate_events


def _label(name: str) -> str:
    return {
        "baseline": "Baseline",
        "fisher_dgpo_no_trust": "Frozen no-trust",
        "fisher_dgpo_trust": "Frozen trust",
        "iterative_refresh_no_trust": "Refresh no-trust",
        "iterative_refresh_trust": "Refresh trust",
    }[name]


def _load_flow(path: Path, config: dict[str, Any], device: torch.device) -> ConditionalFlow:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = make_flow(config, device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _load_score(path: Path, config: dict[str, Any], device: torch.device) -> ScoreModel:
    payload = torch.load(path, map_location=device, weights_only=True)
    settings = config["training"]
    model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _fisher_by_bins(
    events: dict[str, torch.Tensor], reconstructed: dict[str, torch.Tensor],
    nominal_C: float, bin_counts: list[int],
) -> list[float]:
    x = events["x"].detach().cpu().numpy()
    y = reconstructed["y"].detach().cpu().numpy()
    valid = reconstructed["valid"].detach().cpu().numpy().astype(bool)
    score = x / (1.0 + nominal_C * x)
    values = []
    for bins in bin_counts:
        edges = np.linspace(-1.0, 1.0, bins + 1)
        indices = np.clip(np.searchsorted(edges, y[valid], side="right") - 1, 0, bins - 1)
        counts = np.bincount(indices, minlength=bins)
        derivatives = np.bincount(indices, weights=score[valid], minlength=bins)
        values.append(float(np.sum(derivatives**2 / np.clip(counts, 1.0e-12, None)) / x.size))
    return values


def _response(x: np.ndarray, y: np.ndarray, valid: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(y[valid], x[valid], bins=(edges, edges))
    return counts / np.clip(counts.sum(axis=0, keepdims=True), 1.0, None)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)


def _roundwise_diagnostics(
    names: list[str], policies: dict[str, ConditionalFlow], config: dict[str, Any],
    device: torch.device, checkpoints: Path, output: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]], dict[str, ScoreModel], dict[str, ScoreModel]]:
    settings = config["ablation"]
    refresh = config["refresh"]
    nominal_C = float(config["physics"]["nominal_C"])
    rounds = int(refresh["rounds"])
    bin_counts = [int(value) for value in settings["fisher_bin_counts"]]
    grid = torch.linspace(-1.0, 1.0, int(refresh["score_grid_points"]), device=device)
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    curves_by_policy: dict[str, list[np.ndarray]] = {}
    final_active: dict[str, ScoreModel] = {}
    final_diagnostic: dict[str, ScoreModel] = {}

    for name in names:
        rows: list[dict[str, Any]] = []
        curves: list[np.ndarray] = []
        round_dir = checkpoints / name
        for round_index in range(rounds):
            reference = _load_flow(round_dir / f"round_{round_index:02d}_reference_policy.pt", config, device)
            updated = _load_flow(round_dir / f"round_{round_index:02d}_updated_policy.pt", config, device)
            active_score = _load_score(round_dir / f"round_{round_index:02d}_score.pt", config, device)
            score_events = generate_events(
                int(settings["diagnostic_score_events"]), nominal_C, config, device,
                make_generator(device, int(config["seed"]) + 500000 + 10000 * round_index),
            )
            diagnostic_score, _, losses = rebuild_policy_score(
                updated, score_events, config, device,
                int(config["seed"]) + 510000 + 10000 * round_index,
                int(refresh["score_epochs"]), f"round diagnostic: {_label(name)} {round_index + 1}",
            )
            torch.save(
                {"method_version": 3, "diagnostic_only": True, "round": round_index, "state_dict": diagnostic_score.state_dict()},
                output / f"diagnostic_score_{name}_round_{round_index:02d}.pt",
            )
            evaluation_events = generate_events(
                int(settings["evaluation_events"]), nominal_C, config, device,
                make_generator(device, int(config["seed"]) + 520000 + 10000 * round_index),
            )
            reconstruction_seed = int(config["seed"]) + 530000 + 10000 * round_index
            before = reconstruct_policy(reference, evaluation_events, config, make_generator(device, reconstruction_seed))
            after = reconstruct_policy(updated, evaluation_events, config, make_generator(device, reconstruction_seed))
            with torch.no_grad():
                before_score = active_score(before["y"]) * before["valid"]
                surrogate_score = active_score(after["y"]) * after["valid"]
                policy_score = diagnostic_score(after["y"]) * after["valid"]
                action = after["action"]
                local_kl = (
                    updated.log_prob(action, evaluation_events["context"])
                    - reference.log_prob(action, evaluation_events["context"])
                ).mean()
                global_kl = (
                    updated.log_prob(action, evaluation_events["context"])
                    - policies["baseline"].log_prob(action, evaluation_events["context"])
                ).mean()
                cosine = (after["k_a"] * evaluation_events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
                valid = after["valid"]
                mean_information_valid = policy_score[valid].square().mean()
            before_fisher = float(before_score.square().mean())
            surrogate_fisher = float(surrogate_score.square().mean())
            policy_fisher = float(policy_score.square().mean())
            binned = _fisher_by_bins(evaluation_events, after, nominal_C, bin_counts)
            rows.append({
                "policy": name,
                "round": round_index + 1,
                "total_dgpo_epoch": (round_index + 1) * int(refresh["dgpo_epochs_per_round"]),
                "I_before_update_per_event": before_fisher,
                "I_surrogate_after_update_per_event": surrogate_fisher,
                "I_policy_after_update_per_event": policy_fisher,
                "I_binned_after_update_per_event": binned[-1],
                "stale_gap": surrogate_fisher / policy_fisher - 1.0,
                "binned_policy_gap": binned[-1] / policy_fisher - 1.0,
                "local_kl_to_round_reference": float(local_kl),
                "global_kl_to_baseline": float(global_kl),
                "valid_efficiency": float(valid.float().mean()),
                "mean_policy_score_squared_valid": float(mean_information_valid),
                "angular_error": float(torch.acos(cosine).mean()),
                "diagnostic_score_loss": losses[-1],
                "binned_fisher_by_bin_count": binned,
            })
            with torch.no_grad():
                curves.append(active_score(grid).detach().cpu().numpy())
            final_active[name] = active_score
            final_diagnostic[name] = diagnostic_score
        with torch.no_grad():
            curves.append(final_diagnostic[name](grid).detach().cpu().numpy())
        rows_by_policy[name] = rows
        curves_by_policy[name] = curves
        _write_rows(output / f"roundwise_{name}", rows)
    np.save(output / "score_grid.npy", grid.detach().cpu().numpy())
    return rows_by_policy, curves_by_policy, final_active, final_diagnostic


def _final_diagnostics(
    names: list[str], iterative_names: list[str], policies: dict[str, ConditionalFlow],
    reference_score: ScoreModel, final_active: dict[str, ScoreModel], final_diagnostic: dict[str, ScoreModel],
    summaries: list[dict[str, Any]], config: dict[str, Any], device: torch.device, output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config["ablation"]
    refresh = config["refresh"]
    nominal_C = float(config["physics"]["nominal_C"])
    bin_counts = [int(value) for value in settings["fisher_bin_counts"]]
    score_events = generate_events(
        int(settings["diagnostic_score_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + 600000),
    )
    policy_scores = dict(final_diagnostic)
    for name in names:
        if name in iterative_names:
            continue
        policy_scores[name], _, _ = rebuild_policy_score(
            policies[name], score_events, config, device, int(config["seed"]) + 610000,
            int(refresh["score_epochs"]), f"final diagnostic: {_label(name)}",
        )
        torch.save(
            {"method_version": 3, "diagnostic_only": True, "state_dict": policy_scores[name].state_dict()},
            output / f"diagnostic_score_final_{name}.pt",
        )
    active_scores = {name: final_active[name] if name in iterative_names else reference_score for name in names}
    evaluation_events = generate_events(
        int(settings["evaluation_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + 620000),
    )
    reconstructions = {
        name: reconstruct_policy(
            policies[name], evaluation_events, config,
            make_generator(device, int(config["seed"]) + 630000),
        )
        for name in names
    }
    raw: dict[str, dict[str, Any]] = {}
    for name in names:
        reconstructed = reconstructions[name]
        valid = reconstructed["valid"]
        with torch.no_grad():
            s0 = reference_score(reconstructed["y"]) * valid
            active = active_scores[name](reconstructed["y"]) * valid
            policy_score = policy_scores[name](reconstructed["y"]) * valid
            action = reconstructed["action"]
            global_kl = (
                policies[name].log_prob(action, evaluation_events["context"])
                - policies["baseline"].log_prob(action, evaluation_events["context"])
            ).mean()
            cosine = (reconstructed["k_a"] * evaluation_events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
        raw[name] = {
            "I_s0": float(s0.square().mean()),
            "I_active": float(active.square().mean()),
            "I_policy": float(policy_score.square().mean()),
            "I_binned_by_bin_count": _fisher_by_bins(evaluation_events, reconstructed, nominal_C, bin_counts),
            "global_kl": float(global_kl),
            "valid_efficiency": float(valid.float().mean()),
            "angular_error": float(torch.acos(cosine).mean()),
        }
    baseline = raw["baseline"]
    nominal_rows = {
        name: min((row for row in summaries if row["policy"] == name), key=lambda row: abs(row["C_true"] - nominal_C))
        for name in names
    }
    baseline_sigma = float(nominal_rows["baseline"]["std_C_hat"])
    metrics = []
    for name in names:
        values = raw[name]
        policy_rows = [row for row in summaries if row["policy"] == name]
        direct = values["I_binned_by_bin_count"][-1]
        metrics.append({
            "policy": name,
            "I_s0_per_event": values["I_s0"],
            "I_active_per_event": values["I_active"],
            "I_policy_per_event": values["I_policy"],
            "I_binned_per_event": direct,
            "I_binned_by_bin_count": values["I_binned_by_bin_count"],
            "s0_fisher_gain": values["I_s0"] / baseline["I_s0"] - 1.0,
            "active_surrogate_fisher_gain": values["I_active"] / baseline["I_active"] - 1.0,
            "policy_score_fisher_gain": values["I_policy"] / baseline["I_policy"] - 1.0,
            "direct_fisher_gain": direct / baseline["I_binned_by_bin_count"][-1] - 1.0,
            "active_policy_stale_gap": values["I_active"] / values["I_policy"] - 1.0,
            "direct_policy_gap": direct / values["I_policy"] - 1.0,
            "nominal_std_C_hat": float(nominal_rows[name]["std_C_hat"]),
            "std_ratio_to_baseline": float(nominal_rows[name]["std_C_hat"]) / baseline_sigma,
            "Cnn_precision_gain": baseline_sigma / float(nominal_rows[name]["std_C_hat"]) - 1.0,
            "max_abs_bias": float(max(abs(row["bias"]) for row in policy_rows)),
            "global_kl": values["global_kl"],
            "valid_efficiency": values["valid_efficiency"],
            "angular_error": values["angular_error"],
        })
    _write_rows(output / "final_ablation_metrics", metrics)
    return metrics, {"events": evaluation_events, "reconstructions": reconstructions}


def _write_report(
    metrics: list[dict[str, Any]], round_rows: dict[str, list[dict[str, Any]]],
    summaries: list[dict[str, Any]], output: Path,
) -> None:
    by_name = {row["policy"]: row for row in metrics}
    frozen_no = by_name["fisher_dgpo_no_trust"]
    frozen_trust = by_name["fisher_dgpo_trust"]
    refresh_no = by_name["iterative_refresh_no_trust"]
    refresh_trust = by_name["iterative_refresh_trust"]
    final_round_no = round_rows["iterative_refresh_no_trust"][-1]
    final_round_trust = round_rows["iterative_refresh_trust"][-1]
    q1 = abs(final_round_no["stale_gap"]) < abs(frozen_no["active_policy_stale_gap"])
    q2_frozen = abs(frozen_trust["active_policy_stale_gap"]) < abs(frozen_no["active_policy_stale_gap"])
    q2_refresh = abs(final_round_trust["stale_gap"]) < abs(final_round_no["stale_gap"])
    q4 = refresh_no["global_kl"] > refresh_trust["global_kl"]
    pseudo_count = min(int(row["pseudo_experiments"]) for row in summaries)
    lines = ["# Clean 2x2 score-refresh and KL-trust ablation", ""]
    if pseudo_count < 50:
        lines.extend([
            "> Warning: this is a software smoke result with fewer than 50 pseudo-experiments per point. Do not use its numerical ordering as a scientific conclusion.",
            "",
        ])
    lines.extend([
        "All optimized policies use the same configured total DGPO epoch budget. No bias, truth-reconstruction, parameter-distance, policy-to-baseline clipping, or global-anchor term is used.",
        "",
        "## Q1: Does refresh reduce the no-trust surrogate gap?",
        "",
        f"{'Yes' if q1 else 'No'} for this run. The absolute final active-surrogate gap is {abs(frozen_no['active_policy_stale_gap']):.4g} for frozen no-trust and {abs(final_round_no['stale_gap']):.4g} for refresh no-trust.",
        "",
        "## Q2: Does trust reduce surrogate exploitation?",
        "",
        f"Frozen comparison: {'yes' if q2_frozen else 'no'}; |gap| changes from {abs(frozen_no['active_policy_stale_gap']):.4g} to {abs(frozen_trust['active_policy_stale_gap']):.4g}.",
        f"Refresh comparison: {'yes' if q2_refresh else 'no'}; final-round |gap| changes from {abs(final_round_no['stale_gap']):.4g} to {abs(final_round_trust['stale_gap']):.4g}.",
        "",
        "## Q3: Does refresh improve actual information or only surrogate honesty?",
        "",
    ])
    for row in metrics:
        lines.append(
            f"- **{_label(row['policy'])}:** policy-score gain {100 * row['policy_score_fisher_gain']:+.2f}%, "
            f"direct-Fisher gain {100 * row['direct_fisher_gain']:+.2f}%, actual nominal Cnn precision gain {100 * row['Cnn_precision_gain']:+.2f}%."
        )
    lines.extend([
        "",
        "The measurement claim must follow the pseudo-experiment precision and high-statistics direct Fisher, not the active training surrogate.",
        "",
        "## Q4: Does refresh no-trust accumulate larger global drift?",
        "",
        f"{'Yes' if q4 else 'No'} for this run. Final global KL is {refresh_no['global_kl']:.4g} without trust and {refresh_trust['global_kl']:.4g} with local trust. Both are diagnostics; no KL enters the refresh-no-trust loss.",
        "",
    ])
    (output / "ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_ablation_fisher(
    metrics: list[dict[str, Any]], bin_counts: list[int], config: dict[str, Any], output: Path,
) -> None:
    positions = np.arange(len(metrics))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    fields = (
        ("policy_score_fisher_gain", "Policy-specific Fisher"),
        ("direct_fisher_gain", "Direct binned Fisher"),
        ("Cnn_precision_gain", r"Actual $C_{nn}$ precision"),
    )
    for index, (field, label) in enumerate(fields):
        axes[0].bar(positions + (index - 1) * width, [100.0 * row[field] for row in metrics], width, label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[0].set(xticks=positions, xticklabels=[_label(row["policy"]) for row in metrics], ylabel="Gain over baseline [%]")
    axes[0].tick_params(axis="x", rotation=16)
    axes[0].legend(frameon=False, fontsize=8)
    for row in metrics:
        axes[1].plot(bin_counts, row["I_binned_by_bin_count"], marker="o", label=_label(row["policy"]))
    axes[1].set(xlabel="Reconstructed-y bins", ylabel="Direct Fisher / event", title="Binned-Fisher convergence")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "01_ablation_fisher.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_roundwise_gaps(rows: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for axis, (name, policy_rows) in zip(axes, rows.items()):
        rounds = [row["round"] for row in policy_rows]
        axis.plot(rounds, [row["stale_gap"] for row in policy_rows], marker="o", label=r"$I_{surrogate}/I_{policy}-1$")
        axis.plot(rounds, [row["binned_policy_gap"] for row in policy_rows], marker="s", label=r"$I_{bin}/I_{policy}-1$")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set(xlabel="Refresh round", title=_label(name))
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Relative Fisher mismatch")
    fig.tight_layout()
    fig.savefig(output / "02_roundwise_stale_gap.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_roundwise_information(rows: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path) -> None:
    fields = (
        ("I_before_update_per_event", "Before update"),
        ("I_surrogate_after_update_per_event", "Active surrogate after"),
        ("I_policy_after_update_per_event", "Policy score after"),
        ("I_binned_after_update_per_event", "Direct binned after"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for axis, (name, policy_rows) in zip(axes, rows.items()):
        for field, label in fields:
            axis.plot([row["round"] for row in policy_rows], [row[field] for row in policy_rows], marker="o", label=label)
        axis.set(xlabel="Refresh round", title=_label(name))
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Fisher / event")
    fig.tight_layout()
    fig.savefig(output / "03_roundwise_information.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_policy_drift(
    names: list[str], histories: dict[str, list[dict[str, float]]], config: dict[str, Any], output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 5.4))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(names)))
    for color, name in zip(colors, names):
        rows = histories[name]
        epochs = [row["epoch"] for row in rows]
        if name.startswith("iterative_refresh"):
            axis.plot(epochs, [row["global_kl_monitor"] for row in rows], color=color, marker="o", label=f"{_label(name)}: global")
            axis.plot(epochs, [row["kl_to_reference"] for row in rows], color=color, linestyle="--", label=f"{_label(name)}: local")
        else:
            axis.plot(epochs, [row["kl_to_reference"] for row in rows], color=color, label=_label(name))
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xlabel="Total DGPO epoch", ylabel="Sampled KL diagnostic", title="Global drift and current-round local drift")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "04_policy_drift.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_score_evolution(
    curves: dict[str, list[np.ndarray]], grid: np.ndarray, config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)
    for axis, (name, policy_curves) in zip(axes, curves.items()):
        for index, curve in enumerate(policy_curves[:-1]):
            axis.plot(grid, curve, label=rf"$s_{index}(y)$")
        axis.plot(grid, policy_curves[-1], color="black", linestyle="--", linewidth=2.0, label=r"$s_{final}^{diag}(y)$")
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set(xlabel="Reconstructed y", title=_label(name))
        axis.legend(frameon=False, fontsize=7, ncol=2)
    axes[0].set_ylabel("Reconstructed score")
    fig.tight_layout()
    fig.savefig(output / "05_score_evolution.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_responses(
    names: list[str], final_sample: dict[str, Any], config: dict[str, Any], output: Path,
) -> None:
    x = final_sample["events"]["x"].detach().cpu().numpy()
    matrices = {}
    for name in names:
        reconstructed = final_sample["reconstructions"][name]
        matrices[name] = _response(
            x, reconstructed["y"].detach().cpu().numpy(),
            reconstructed["valid"].detach().cpu().numpy().astype(bool), int(config["plots"]["response_bins"]),
        )
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2), squeeze=False)
    for index, name in enumerate(names):
        image = axes[0, index].imshow(
            matrices[name], origin="lower", extent=(-1, 1, -1, 1), aspect="auto",
            cmap="magma", vmin=0.0, vmax=maximum,
        )
        axes[0, index].set(title=_label(name), xlabel="Truth x", ylabel="Reconstructed y")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Probability per y bin", shrink=0.82)
    fig.suptitle(r"Column-normalized response $R(y\mid x)$; diagonalness is not the objective")
    fig.subplots_adjust(left=0.05, right=0.94, bottom=0.14, top=0.82, wspace=0.28)
    fig.savefig(output / "06_response_comparison.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_precision(
    names: list[str], summaries: list[dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    baseline = {row["C_true"]: row for row in summaries if row["policy"] == "baseline"}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)
    for name in names:
        rows = sorted((row for row in summaries if row["policy"] == name), key=lambda row: row["C_true"])
        truth = [row["C_true"] for row in rows]
        axes[0].plot(truth, [row["std_C_hat"] for row in rows], marker="o", label=_label(name))
        axes[1].plot(truth, [row["std_C_hat"] / baseline[row["C_true"]]["std_C_hat"] for row in rows], marker="o", label=_label(name))
    axes[0].set(ylabel=r"Std$(\hat C_{nn})$")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set(xlabel=r"$C_{true}$", ylabel="Std ratio to baseline")
    fig.tight_layout()
    fig.savefig(output / "07_Cnn_precision.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_dashboard(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    columns = [
        "Global KL", "Valid eff.", "Active I gain", "Policy I gain", "Direct I gain",
        "Cnn precision gain", "Max |bias|", "Axis error [rad]",
    ]
    cells = [[
        f"{row['global_kl']:.4g}", f"{row['valid_efficiency']:.3f}",
        f"{100 * row['active_surrogate_fisher_gain']:+.1f}%", f"{100 * row['policy_score_fisher_gain']:+.1f}%",
        f"{100 * row['direct_fisher_gain']:+.1f}%", f"{100 * row['Cnn_precision_gain']:+.1f}%",
        f"{row['max_abs_bias']:.4f}", f"{row['angular_error']:.4f}",
    ] for row in metrics]
    fig, axis = plt.subplots(figsize=(14.0, 4.4))
    axis.axis("off")
    table = axis.table(
        cellText=cells, rowLabels=[_label(row["policy"]) for row in metrics],
        colLabels=columns, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    axis.set_title("Controlled 2x2 score-refresh and KL-trust ablation", pad=20)
    fig.tight_layout()
    fig.savefig(output / "08_final_ablation_dashboard.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def make_ablation_study(
    policies: dict[str, ConditionalFlow], reference_score: ScoreModel,
    histories: dict[str, list[dict[str, float]]], summaries: list[dict[str, Any]],
    config: dict[str, Any], device: torch.device, root_output: Path, checkpoints: Path,
) -> list[dict[str, Any]]:
    names = [str(name) for name in config["ablation"]["policy_order"]]
    iterative_names = [name for name in names if name.startswith("iterative_refresh")]
    optimized_names = [name for name in names if name != "baseline"]
    output = root_output / str(config["ablation"]["output_subdir"])
    output.mkdir(parents=True, exist_ok=True)
    round_rows, curves, final_active, final_diagnostic = _roundwise_diagnostics(
        iterative_names, policies, config, device, checkpoints, output,
    )
    metrics, final_sample = _final_diagnostics(
        names, iterative_names, policies, reference_score, final_active, final_diagnostic,
        summaries, config, device, output,
    )
    bin_counts = [int(value) for value in config["ablation"]["fisher_bin_counts"]]
    grid = np.load(output / "score_grid.npy")
    _plot_ablation_fisher(metrics, bin_counts, config, output)
    _plot_roundwise_gaps(round_rows, config, output)
    _plot_roundwise_information(round_rows, config, output)
    _plot_policy_drift(optimized_names, histories, config, output)
    _plot_score_evolution(curves, grid, config, output)
    _plot_responses(names, final_sample, config, output)
    _plot_precision(names, summaries, config, output)
    _plot_dashboard(metrics, config, output)
    _write_report(metrics, round_rows, summaries, output)
    return metrics
