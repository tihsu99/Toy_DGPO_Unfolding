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
from .inference import reweighted_binned_fisher_per_event
from .training import make_flow, rebuild_policy_score, reconstruct_policy
from .ztautau import generate_events


def _label(name: str) -> str:
    return {
        "baseline": "Baseline",
        "fisher_dgpo_no_trust": "Frozen score, no trust",
        "fisher_dgpo_trust": "Frozen score, trust",
        "iterative_refresh_no_trust": "Iterative refresh, no trust",
        "iterative_refresh_trust": "Iterative refresh, trust",
    }[name]


def _style(name: str) -> dict[str, Any]:
    colors = {
        "fisher_dgpo_no_trust": "tab:orange",
        "fisher_dgpo_trust": "tab:blue",
        "iterative_refresh_no_trust": "tab:red",
        "iterative_refresh_trust": "tab:green",
    }
    no_trust = name.endswith("no_trust")
    return {
        "color": colors[name],
        "linestyle": "--" if no_trust else "-",
        "marker": "o" if no_trust else "s",
        "markerfacecolor": "none" if no_trust else colors[name],
    }


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
    device: torch.device, checkpoints: Path, output: Path, reference_score: ScoreModel,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[np.ndarray]], dict[str, ScoreModel]]:
    settings = config["ablation"]
    refresh = config["refresh"]
    nominal_C = float(config["physics"]["nominal_C"])
    bin_counts = [int(value) for value in settings["fisher_bin_counts"]]
    selection_index = bin_counts.index(int(settings["selection_bins"]))
    grid = torch.linspace(-1.0, 1.0, int(refresh["score_grid_points"]), device=device)
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    curves_by_policy: dict[str, list[np.ndarray]] = {}
    final_active: dict[str, ScoreModel] = {}
    final_diagnostic: dict[str, ScoreModel] = {}
    with (output.parent / "checkpoint_selection_summary.json").open(encoding="utf-8") as stream:
        selection_summary = json.load(stream)

    for name in names:
        rows: list[dict[str, Any]] = []
        curves: list[np.ndarray] = []
        round_dir = checkpoints / name
        rounds = len(list(round_dir.glob("round_*_updated_policy.pt")))
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
                int(settings["validation_events"]), nominal_C, config, device,
                make_generator(
                    device, int(config["seed"]) + int(settings["validation_event_seed_offset"]),
                ),
            )
            reconstruction_seed = int(config["seed"]) + int(settings["validation_reconstruction_seed_offset"])
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
                "I_binned_after_update_per_event": binned[selection_index],
                "stale_gap": surrogate_fisher / policy_fisher - 1.0,
                "score_to_direct_gap": policy_fisher / binned[selection_index] - 1.0,
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
        best = next(
            row for row in selection_summary
            if row["policy"] == name and row["checkpoint_role"] == "best"
        )
        best_round = int(best["round"])
        final_active[name] = reference_score if best_round == 0 else _load_score(
            round_dir / f"round_{best_round - 1:02d}_score.pt", config, device,
        )
    np.save(output / "score_grid.npy", grid.detach().cpu().numpy())
    return rows_by_policy, curves_by_policy, final_active


def _final_diagnostics(
    names: list[str], policies: dict[str, ConditionalFlow],
    reference_score: ScoreModel, final_active: dict[str, ScoreModel],
    summaries: list[dict[str, Any]], config: dict[str, Any], device: torch.device, output: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config["ablation"]
    refresh = config["refresh"]
    nominal_C = float(config["physics"]["nominal_C"])
    bin_counts = [int(value) for value in settings["fisher_bin_counts"]]
    selection_index = bin_counts.index(int(settings["selection_bins"]))
    score_events = generate_events(
        int(settings["diagnostic_score_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + 600000),
    )
    policy_scores: dict[str, ScoreModel] = {}
    for name in names:
        policy_scores[name], _, _ = rebuild_policy_score(
            policies[name], score_events, config, device, int(config["seed"]) + 610000,
            int(refresh["score_epochs"]), f"final diagnostic: {_label(name)}",
        )
        torch.save(
            {"method_version": 3, "diagnostic_only": True, "state_dict": policy_scores[name].state_dict()},
            output / f"diagnostic_score_final_{name}.pt",
        )
    active_scores = {
        name: final_active[name] if name.startswith("iterative_refresh") else reference_score
        for name in names
    }
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
    with (output.parent / "checkpoint_validation_all.json").open(encoding="utf-8") as stream:
        validation_rows = json.load(stream)
    selected_validation = {
        name: max(
            (row for row in validation_rows if row["policy"] == name),
            key=lambda row: float(row["validation_fisher"]),
        )
        for name in names
    }
    baseline_validation_fisher = float(selected_validation["baseline"]["validation_fisher"])
    metrics = []
    for name in names:
        values = raw[name]
        policy_rows = [row for row in summaries if row["policy"] == name]
        direct = values["I_binned_by_bin_count"][selection_index]
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
            "direct_fisher_gain": direct / baseline["I_binned_by_bin_count"][selection_index] - 1.0,
            "best_validation_fisher_gain": (
                float(selected_validation[name]["validation_fisher"]) / baseline_validation_fisher - 1.0
            ),
            "active_policy_stale_gap": values["I_active"] / values["I_policy"] - 1.0,
            "direct_policy_gap": direct / values["I_policy"] - 1.0,
            "nominal_std_C_hat": float(nominal_rows[name]["std_C_hat"]),
            "std_ratio_to_baseline": float(nominal_rows[name]["std_C_hat"]) / baseline_sigma,
            "Cnn_precision_gain": baseline_sigma / float(nominal_rows[name]["std_C_hat"]) - 1.0,
            "validation_predicted_std_ratio": float(np.sqrt(
                baseline_validation_fisher / float(selected_validation[name]["validation_fisher"])
            )),
            "predicted_precision_gain": float(np.sqrt(
                float(selected_validation[name]["validation_fisher"]) / baseline_validation_fisher
            ) - 1.0),
            "observed_std_ratio": float(nominal_rows[name]["std_C_hat"]) / baseline_sigma,
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
        "All optimized policies use the same configured maximum DGPO epoch budget; iterative policies may stop earlier by the validation-Fisher patience rule. No bias, truth-reconstruction, parameter-distance, policy-to-baseline clipping, or global-anchor term is used.",
        "",
        "## Metric roles and statistical separation",
        "",
        "- Active surrogate Fisher and replacement reward are training quantities only.",
        "- Independently re-estimated policy-score Fisher is a refresh-boundary diagnostic only.",
        "- Fixed high-statistics direct binned validation Fisher selects the checkpoint.",
        "- Independent Poisson pseudo-experiment Std(C_hat) is the final scientific validation and is evaluated only after selection.",
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
            f"direct-Fisher gain {100 * row['direct_fisher_gain']:+.2f}%, actual nominal Cnn precision gain {100 * row['Cnn_precision_gain']:+.2f}%; "
            f"validation-predicted Std ratio {row['validation_predicted_std_ratio']:.4f}, observed nominal Std ratio {row['observed_std_ratio']:.4f}."
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
        name = row["policy"]
        style = {"color": "black", "linestyle": ":", "marker": "x"} if name == "baseline" else _style(name)
        axes[1].plot(bin_counts, row["I_binned_by_bin_count"], label=_label(name), **style)
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
        axis.plot(rounds, [row["score_to_direct_gap"] for row in policy_rows], marker="s", label=r"$I_{refreshed}/I_{bin}^{val}-1$")
        axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        axis.set(xlabel="Refresh round", title=_label(name))
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Relative Fisher mismatch")
    fig.tight_layout()
    fig.savefig(output / "02_roundwise_stale_gap.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_roundwise_information(rows: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
    for axis, (name, policy_rows) in zip(axes, rows.items()):
        first = True
        for row in policy_rows:
            round_index = row["round"]
            values = [
                row["I_surrogate_after_update_per_event"],
                row["I_policy_after_update_per_event"],
                row["I_binned_after_update_per_event"],
            ]
            axis.plot([round_index] * 3, values, color="0.65", linewidth=1.0, zorder=1)
            axis.scatter(round_index, values[0], marker="o", color="tab:orange", zorder=3,
                         label="Old surrogate on updated policy" if first else None)
            axis.scatter(round_index, values[1], marker="s", color="tab:green", zorder=3,
                         label="Refreshed policy score" if first else None)
            axis.scatter(round_index, values[2], marker="D", color="black", zorder=3,
                         label="Direct validation Fisher" if first else None)
            first = False
        axis.set(xlabel="Refresh boundary", title=_label(name))
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Fisher / event")
    fig.tight_layout()
    fig.savefig(output / "03_roundwise_information.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_roundwise_refresh_closure(
    rows: dict[str, list[dict[str, Any]]], config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(13.0, 4.2 * len(rows)), squeeze=False)
    for row_index, (name, policy_rows) in enumerate(rows.items()):
        information_axis, gap_axis = axes[row_index]
        sequence_x: list[float] = []
        sequence_y: list[float] = []
        for item in policy_rows:
            boundary = float(item["round"])
            offsets = (-0.22, 0.0, 0.22)
            values = (
                item["I_surrogate_after_update_per_event"],
                item["I_policy_after_update_per_event"],
                item["I_binned_after_update_per_event"],
            )
            sequence_x.extend(boundary + offset for offset in offsets)
            sequence_y.extend(values)
        information_axis.plot(sequence_x, sequence_y, color="0.7", linewidth=1.0, zorder=1)
        for offset, key, marker, color, label in (
            (-0.22, "I_surrogate_after_update_per_event", "o", "tab:orange", "Old surrogate"),
            (0.0, "I_policy_after_update_per_event", "s", "tab:green", "Refreshed score"),
            (0.22, "I_binned_after_update_per_event", "D", "black", "Direct validation"),
        ):
            information_axis.scatter(
                [float(item["round"]) + offset for item in policy_rows],
                [item[key] for item in policy_rows], marker=marker, color=color, label=label, zorder=3,
            )
        boundaries = [item["round"] for item in policy_rows]
        gap_axis.plot(
            boundaries, [item["stale_gap"] for item in policy_rows], marker="o",
            label=r"$I_{old}/I_{refreshed}-1$",
        )
        gap_axis.plot(
            boundaries, [item["score_to_direct_gap"] for item in policy_rows], marker="s",
            label=r"$I_{refreshed}/I_{bin}^{val}-1$",
        )
        gap_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        information_axis.set(
            xlabel="Refresh boundary", ylabel="Fisher / event",
            title=f"{_label(name)}: boundary sawtooth",
        )
        gap_axis.set(
            xlabel="Refresh boundary", ylabel="Relative mismatch",
            title=f"{_label(name)}: closure gaps",
        )
        information_axis.legend(frameon=False, fontsize=8)
        gap_axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "roundwise_refresh_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_validation_early_stopping(
    names: list[str], config: dict[str, Any], root_output: Path, output: Path,
) -> None:
    with (root_output / "checkpoint_validation_all.json").open(encoding="utf-8") as stream:
        validation_rows = json.load(stream)
    with (root_output / "checkpoint_selection_summary.json").open(encoding="utf-8") as stream:
        selection_rows = json.load(stream)
    optimized = [name for name in names if name != "baseline"]
    baseline = next(row for row in validation_rows if row["policy"] == "baseline")
    best = {
        row["policy"]: row for row in selection_rows if row["checkpoint_role"] == "best"
    }
    final = {
        row["policy"]: row for row in selection_rows if row["checkpoint_role"] == "final"
    }
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    for name in optimized:
        rows = [row for row in validation_rows if row["policy"] == name]
        style = _style(name)
        epochs = [row["epoch"] for row in rows]
        fisher = [row["validation_fisher_per_event"] for row in rows]
        sigma = [row["validation_sigma"] for row in rows]
        axes[0].plot(epochs, fisher, label=_label(name), **style)
        axes[1].plot(epochs, sigma, label=_label(name), **style)
        for axis, key in zip(axes, ("validation_fisher_per_event", "validation_sigma")):
            axis.scatter(best[name]["epoch"], best[name][key], marker="*", s=140,
                         color=style["color"], edgecolor="black", zorder=5)
            axis.scatter(final[name]["epoch"], final[name][key], marker="X", s=75,
                         color=style["color"], edgecolor="black", zorder=5)
            if name.startswith("iterative_refresh") and int(final[name]["round"]) < int(config["refresh"]["max_refresh_rounds"]):
                axis.scatter(final[name]["epoch"], final[name][key], marker="v", s=170,
                             facecolor="none", edgecolor=style["color"], linewidth=1.5, zorder=6)
    axes[0].axhline(baseline["validation_fisher_per_event"], color="black", linestyle=":", label="Baseline")
    axes[1].axhline(baseline["validation_sigma"], color="black", linestyle=":", label="Baseline")
    axes[0].set(xlabel="Total DGPO epoch", ylabel="Direct validation Fisher / event")
    axes[1].set(xlabel="Total DGPO epoch", ylabel=r"Predicted $\sigma_C$")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].text(0.02, 0.98, "star: best   X: final   open triangle: patience stop",
                 transform=axes[1].transAxes, va="top", fontsize=8)
    fig.suptitle("Checkpoint selection uses only fixed independent 80-bin validation Fisher")
    fig.tight_layout()
    fig.savefig(output / "validation_early_stopping.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _fisher_vs_pe_by_C(
    names: list[str], policies: dict[str, ConditionalFlow], summaries: list[dict[str, Any]],
    config: dict[str, Any], device: torch.device, output: Path,
) -> list[dict[str, Any]]:
    settings = config["ablation"]
    nominal_C = float(config["physics"]["nominal_C"])
    target_values = np.asarray(config["physics"]["true_C_values"], dtype=float)
    events = generate_events(
        int(settings["validation_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + int(settings["validation_event_seed_offset"])),
    )
    reconstruction_seed = int(config["seed"]) + int(settings["validation_reconstruction_seed_offset"])
    edges = np.linspace(-1.0, 1.0, int(settings["selection_bins"]) + 1)
    exposure = float(config["data"]["events_per_pseudo_experiment"])
    summary_map = {(row["policy"], float(row["C_true"])): row for row in summaries}
    fisher_by_policy: dict[str, np.ndarray] = {}
    for name in names:
        reconstructed = reconstruct_policy(
            policies[name], events, config, make_generator(device, reconstruction_seed),
        )
        fisher_by_policy[name] = reweighted_binned_fisher_per_event(
            events["x"].detach().cpu().numpy(),
            reconstructed["y"].detach().cpu().numpy(),
            reconstructed["valid"].detach().cpu().numpy().astype(bool),
            edges, nominal_C, target_values,
        )
    rows: list[dict[str, Any]] = []
    for name in names:
        for value_index, target_C in enumerate(target_values):
            per_event = float(fisher_by_policy[name][value_index])
            sigma_fisher = float(1.0 / np.sqrt(per_event * exposure))
            pe_std = float(summary_map[(name, float(target_C))]["std_C_hat"])
            baseline_fisher = float(fisher_by_policy["baseline"][value_index])
            baseline_std = float(summary_map[("baseline", float(target_C))]["std_C_hat"])
            predicted_std_ratio = float(np.sqrt(baseline_fisher / per_event))
            observed_std_ratio = pe_std / baseline_std
            rows.append({
                "policy": name,
                "C_true": float(target_C),
                "validation_events": int(settings["validation_events"]),
                "selection_bins": int(settings["selection_bins"]),
                "pseudo_experiment_exposure": int(exposure),
                "I_bin_per_event": per_event,
                "I_bin_total": per_event * exposure,
                "sigma_fisher": sigma_fisher,
                "PE_std": pe_std,
                "closure_ratio_PE_std_over_sigma_fisher": pe_std / sigma_fisher,
                "predicted_std_ratio_to_baseline": predicted_std_ratio,
                "observed_std_ratio_to_baseline": observed_std_ratio,
                "predicted_precision_gain": 1.0 / predicted_std_ratio - 1.0,
                "observed_PE_precision_gain": 1.0 / observed_std_ratio - 1.0,
            })
    _write_rows(output / "fisher_vs_PE_by_C", rows)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2))
    for name in names:
        selected = [row for row in rows if row["policy"] == name]
        style = {"color": "black", "linestyle": ":", "marker": "x"} if name == "baseline" else _style(name)
        values = [row["C_true"] for row in selected]
        axes[0].plot(values, [row["sigma_fisher"] for row in selected], label=_label(name), **style)
        pe_style = {**style, "linestyle": "none", "marker": "+"}
        axes[0].plot(values, [row["PE_std"] for row in selected], **pe_style)
        axes[1].plot(values, [row["closure_ratio_PE_std_over_sigma_fisher"] for row in selected], label=_label(name), **style)
        axes[2].plot(values, [row["predicted_std_ratio_to_baseline"] for row in selected], label=_label(name), **style)
        observed_style = {**style, "linestyle": "none", "marker": "+"}
        axes[2].plot(values, [row["observed_std_ratio_to_baseline"] for row in selected], **observed_style)
    axes[0].set(xlabel=r"$C_{true}$", ylabel=r"$\sigma(C)$", title="Lines: Fisher prediction; +: PE spread")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set(xlabel=r"$C_{true}$", ylabel=r"PE Std / $\sigma_{Fisher}$", title="Fisher-to-PE closure")
    axes[2].axhline(1.0, color="black", linestyle="--")
    axes[2].set(xlabel=r"$C_{true}$", ylabel="Std ratio to baseline", title="Lines: predicted; +: observed")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "fisher_vs_PE_by_C.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)
    return rows


def _plot_policy_drift(
    names: list[str], histories: dict[str, list[dict[str, float]]], config: dict[str, Any], output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.0, 5.4))
    for name in names:
        rows = histories[name]
        epochs = [row["epoch"] for row in rows]
        style = _style(name)
        if name.startswith("iterative_refresh"):
            axis.plot(epochs, [row["global_kl_monitor"] for row in rows], label=f"{_label(name)}: global", **style)
            local_style = {**style, "linestyle": ":", "marker": None}
            axis.plot(epochs, [row["kl_to_reference"] for row in rows], label=f"{_label(name)}: local", **local_style)
        else:
            axis.plot(epochs, [row["kl_to_reference"] for row in rows], label=_label(name), **style)
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
        style = {"color": "black", "linestyle": ":", "marker": "x"} if name == "baseline" else _style(name)
        axes[0].plot(truth, [row["std_C_hat"] for row in rows], label=_label(name), **style)
        axes[1].plot(truth, [row["std_C_hat"] / baseline[row["C_true"]]["std_C_hat"] for row in rows], label=_label(name), **style)
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


def _plot_final_2x2_ablation(
    metrics: list[dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    by_name = {row["policy"]: row for row in metrics}
    order = [str(name) for name in config["ablation"]["summary_policy_order"]]
    columns = [
        "Best val. I gain", "Final policy-score I gain", "Final direct I gain",
        "Predicted precision gain", "Observed PE precision gain", "Global KL",
        "Valid efficiency", "Tau-axis error [deg]",
    ]
    cells = []
    for name in order:
        row = by_name[name]
        cells.append([
            f"{100 * row['best_validation_fisher_gain']:+.2f}%",
            f"{100 * row['policy_score_fisher_gain']:+.2f}%",
            f"{100 * row['direct_fisher_gain']:+.2f}%",
            f"{100 * row['predicted_precision_gain']:+.2f}%",
            f"{100 * row['Cnn_precision_gain']:+.2f}%",
            f"{row['global_kl']:.4g}",
            f"{row['valid_efficiency']:.3f}",
            f"{np.degrees(row['angular_error']):.3f}",
        ])
    fig, axis = plt.subplots(figsize=(16.8, 4.8))
    axis.axis("off")
    table = axis.table(
        cellText=cells, rowLabels=[_label(name) for name in order],
        colLabels=columns, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1.0, 1.8)
    candidate = _label(str(config["ablation"]["primary_candidate"]))
    axis.set_title(f"Final 2x2 ablation; {candidate} is the primary candidate", pad=20)
    fig.tight_layout()
    fig.savefig(output / "final_2x2_ablation.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_training_dashboard(
    names: list[str], histories: dict[str, list[dict[str, float]]],
    config: dict[str, Any], root_output: Path, output: Path,
) -> None:
    with (root_output / "checkpoint_validation_all.json").open(encoding="utf-8") as stream:
        validation_rows = json.load(stream)
    baseline = next(row for row in validation_rows if row["policy"] == "baseline")
    optimized = [name for name in names if name != "baseline"]
    panels = (
        ("active", "Active/local surrogate Fisher / event"),
        ("direct", "Independent direct validation Fisher / event"),
        ("sigma", r"Validation $\sigma_C=1/\sqrt{I_{bin}^{val}}$"),
        ("reward", "Replacement reward"),
        ("local_kl", "Local KL diagnostic"),
        ("global_kl", r"Global KL to $\pi_0$"),
        ("invalid_fraction", "Invalid fraction"),
        ("angular_error", "Tau-axis error [rad]"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.2))
    for axis, (panel, title) in zip(axes.flat, panels):
        for name in optimized:
            rows = histories[name]
            style = _style(name)
            if panel in {"direct", "sigma"}:
                selected = [row for row in validation_rows if row["policy"] == name]
                key = "validation_fisher_per_event" if panel == "direct" else "validation_sigma"
                marker_style = {**style, "linestyle": "none"}
                axis.plot(
                    [row["epoch"] for row in selected], [row[key] for row in selected],
                    markersize=5, label=_label(name), **marker_style,
                )
                continue
            if panel == "active":
                key = "fisher_per_event"
            elif panel == "reward":
                key = "reward"
            elif panel == "local_kl":
                key = "kl_to_reference"
            elif panel == "global_kl":
                key = "global_kl_monitor" if name.startswith("iterative_refresh") else "kl_to_reference"
            else:
                key = panel
            axis.plot(
                [row["epoch"] for row in rows], [row[key] for row in rows],
                markersize=2.5, markevery=max(1, len(rows) // 12), label=_label(name), **style,
            )
        if panel == "active":
            axis.axhline(baseline["active_fisher_per_event"], color="black", linestyle=":", label="Baseline")
        elif panel == "direct":
            axis.axhline(baseline["validation_fisher_per_event"], color="black", linestyle=":", label="Baseline")
        elif panel == "sigma":
            axis.axhline(baseline["validation_sigma"], color="black", linestyle=":", label="Baseline")
        elif panel == "invalid_fraction":
            axis.axhline(1.0 - baseline["valid_fraction"], color="black", linestyle=":", label="Baseline")
        elif panel == "angular_error":
            axis.axhline(baseline["angular_error"], color="black", linestyle=":", label="Baseline")
        axis.set(xlabel="Total DGPO epoch", ylabel=title)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Complete 2x2 training diagnostics; validation markers are independent checkpoints")
    fig.tight_layout()
    fig.savefig(output / "00_training_diagnostics.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_checkpoint_summary(config: dict[str, Any], root_output: Path, output: Path) -> None:
    with (root_output / "checkpoint_selection_summary.json").open(encoding="utf-8") as stream:
        rows = json.load(stream)
    columns = [
        "Role", "Epoch / round", r"$I_{bin}^{val}$", r"$\sigma_{C,val}$", "I gain",
        "Sigma reduction", "Global KL", "Valid", "Axis error",
    ]
    cells = []
    row_labels = []
    for row in rows:
        location = f"{row['epoch']} / {row['round'] if row['round'] is not None else '-'}"
        cells.append([
            row["checkpoint_role"], location, f"{row['validation_fisher']:.5g}",
            f"{row['validation_sigma']:.5g}", f"{100 * row['fisher_gain_vs_baseline']:+.2f}%",
            f"{100 * row['predicted_sigma_reduction']:+.2f}%", f"{row['global_kl_to_baseline']:.4g}",
            f"{row['valid_fraction']:.3f}", f"{row['angular_error']:.4f}",
        ])
        row_labels.append(_label(row["policy"]))
    fig, axis = plt.subplots(figsize=(16.5, 5.8))
    axis.axis("off")
    table = axis.table(cellText=cells, rowLabels=row_labels, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    axis.set_title("Best validation checkpoint versus final optimization checkpoint", pad=18)
    fig.tight_layout()
    fig.savefig(output / "09_checkpoint_selection_summary.png", dpi=int(config["plots"]["dpi"]))
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
    round_rows, curves, final_active = _roundwise_diagnostics(
        iterative_names, policies, config, device, checkpoints, output, reference_score,
    )
    metrics, final_sample = _final_diagnostics(
        names, policies, reference_score, final_active,
        summaries, config, device, output,
    )
    bin_counts = [int(value) for value in config["ablation"]["fisher_bin_counts"]]
    grid = np.load(output / "score_grid.npy")
    _plot_training_dashboard(names, histories, config, root_output, output)
    _plot_ablation_fisher(metrics, bin_counts, config, output)
    _plot_roundwise_gaps(round_rows, config, output)
    _plot_roundwise_information(round_rows, config, output)
    _plot_roundwise_refresh_closure(round_rows, config, output)
    _plot_validation_early_stopping(names, config, root_output, output)
    _plot_policy_drift(optimized_names, histories, config, output)
    _plot_score_evolution(curves, grid, config, output)
    _plot_responses(names, final_sample, config, output)
    _plot_precision(names, summaries, config, output)
    _fisher_vs_pe_by_C(names, policies, summaries, config, device, output)
    _plot_dashboard(metrics, config, output)
    _plot_final_2x2_ablation(metrics, config, output)
    _plot_checkpoint_summary(config, root_output, output)
    _write_report(metrics, round_rows, summaries, output)
    return metrics
