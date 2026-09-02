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
from .training import rebuild_policy_score, reconstruct_policy
from .ztautau import generate_events


def _label(name: str) -> str:
    return {
        "baseline": "Baseline",
        "fisher_dgpo_trust": "Old frozen-score trust",
        "iterative_refresh_trust": "Iterative refresh trust",
        "iterative_refresh_no_trust": "Iterative refresh, no trust",
    }.get(name, name.replace("_", " ").title())


def _load_surrogate_scores(
    names: list[str], reference_score: ScoreModel, config: dict[str, Any], device: torch.device, checkpoints: Path,
) -> dict[str, ScoreModel]:
    models = {name: reference_score for name in names if not name.startswith("iterative_refresh")}
    settings = config["training"]
    for name in names:
        if not name.startswith("iterative_refresh"):
            continue
        payload = torch.load(checkpoints / f"{name}_surrogate_score.pt", map_location=device, weights_only=True)
        model = ScoreModel(int(settings["score_hidden_width"]), int(settings["score_hidden_layers"])).to(device)
        model.load_state_dict(payload["state_dict"])
        models[name] = model.eval()
    return models


def _response(x: np.ndarray, y: np.ndarray, valid: np.ndarray, bins: int) -> np.ndarray:
    edges = np.linspace(-1.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(y[valid], x[valid], bins=(edges, edges))
    return counts / np.clip(counts.sum(axis=0, keepdims=True), 1.0, None)


def _plot_training(history: list[dict[str, float]], config: dict[str, Any], output: Path) -> None:
    fields = [
        ("round_fisher", r"Current policy-score Fisher $I_r$"),
        ("round_predicted_sigma", r"$1/\sqrt{I_r}$"),
        ("local_kl", r"Local $D_{KL}(\pi_\phi||\pi_r)$"),
        ("score_loss", "Score training MSE"),
        ("round_valid_fraction", "Valid fraction"),
        ("angular_error", "Tau-axis error [rad]"),
    ]
    epochs = np.array([row["epoch"] for row in history])
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
    for axis, (field, ylabel) in zip(axes.flat, fields):
        axis.plot(epochs, [row[field] for row in history], marker="o", markersize=3)
        for boundary in range(1, int(config["refresh"]["rounds"])):
            axis.axvline(boundary * int(config["refresh"]["dgpo_epochs_per_round"]) + 0.5, color="gray", linestyle="--", linewidth=0.8)
        axis.set(xlabel="Cumulative local DGPO epoch", ylabel=ylabel)
        axis.grid(alpha=0.2)
    fig.suptitle("Iterative local-refresh training")
    fig.tight_layout()
    fig.savefig(output / "01_refresh_training.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_scores(
    evolution_path: Path, final_model: ScoreModel, config: dict[str, Any], device: torch.device, output: Path,
) -> None:
    evolution = np.load(evolution_path)
    grid = evolution["grid"][0]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for index, curve in enumerate(evolution["scores"]):
        axis.plot(grid, curve, label=rf"$s_{index}(y)$")
    tensor_grid = torch.as_tensor(grid, dtype=torch.float32, device=device)
    with torch.no_grad():
        final_curve = final_model(tensor_grid).cpu().numpy()
    axis.plot(grid, final_curve, color="black", linestyle="--", linewidth=2.0, label=r"Independent $s_{final}(y)$")
    axis.axhline(0.0, color="gray", linewidth=0.7)
    axis.set(xlabel="Reconstructed y", ylabel="Reconstructed score", title="Score evolution across refresh rounds")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "02_score_evolution.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_fisher(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    positions = np.arange(len(metrics))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9.0, 5.2))
    for offset, (field, label) in zip(
        (-width, 0.0, width),
        (("surrogate_gain", "Final-round surrogate"), ("policy_score_gain", "Independent policy score"), ("direct_gain", "Direct fine-binned")),
    ):
        axis.bar(positions + offset, [100.0 * row[field] for row in metrics], width, label=label)
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(xticks=positions, xticklabels=[_label(row["policy"]) for row in metrics], ylabel="Fisher gain over baseline [%]")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(frameon=False)
    axis.set_title("Training-surrogate closure against independent information estimates")
    fig.tight_layout()
    fig.savefig(output / "03_fisher_surrogate_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_precision(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    positions = np.arange(len(metrics))
    fields = [
        ("surrogate_sigma_ratio", "Surrogate"),
        ("policy_score_sigma_ratio", "Independent score"),
        ("direct_sigma_ratio", "Direct binned"),
        ("actual_sigma_ratio", "Pseudo-experiments"),
    ]
    width = 0.19
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, (field, label) in enumerate(fields):
        axis.bar(positions + (index - 1.5) * width, [row[field] for row in metrics], width, label=label)
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set(xticks=positions, xticklabels=[_label(row["policy"]) for row in metrics], ylabel=r"$\sigma_C/\sigma_{C,baseline}$")
    axis.tick_params(axis="x", rotation=12)
    axis.legend(frameon=False, ncol=2)
    axis.set_title("Fisher predictions versus fitted nominal precision")
    fig.tight_layout()
    fig.savefig(output / "04_fisher_vs_actual_precision.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_responses(
    names: list[str], calibration_events: dict[str, torch.Tensor], calibrations: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any], output: Path,
) -> None:
    x = calibration_events["x"].cpu().numpy()
    matrices = {
        name: _response(
            x, calibrations[name]["y"].cpu().numpy(), calibrations[name]["valid"].cpu().numpy().astype(bool),
            int(config["plots"]["response_bins"]),
        )
        for name in names
    }
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.5), squeeze=False)
    for index, name in enumerate(names):
        image = axes[0, index].imshow(
            matrices[name], origin="lower", extent=(-1, 1, -1, 1), aspect="auto",
            cmap="magma", vmin=0.0, vmax=maximum,
        )
        axes[0, index].set(title=_label(name), xlabel="Truth x", ylabel="Reconstructed y")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Probability per y bin", shrink=0.85)
    fig.suptitle(r"Column-normalized response $R(y\mid x)$; diagonalness is not the objective")
    fig.subplots_adjust(left=0.06, right=0.92, bottom=0.13, top=0.84, wspace=0.25)
    fig.savefig(output / "05_response_comparison.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_offnominal(summary: list[dict[str, Any]], names: list[str], config: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for name in names:
        rows = sorted((row for row in summary if row["policy"] == name), key=lambda row: row["C_true"])
        truth = np.array([row["C_true"] for row in rows])
        axes[0, 0].errorbar(truth, [row["mean_C_hat"] for row in rows], yerr=[row["mean_C_hat_error"] for row in rows], marker="o", capsize=3, label=_label(name))
        axes[0, 1].plot(truth, [row["bias"] for row in rows], marker="o", label=_label(name))
        axes[1, 0].plot(truth, [row["std_C_hat"] for row in rows], marker="o", label=_label(name))
        axes[1, 1].plot(truth, [row["coverage_68"] for row in rows], marker="o", label=_label(name))
    values = [float(value) for value in config["physics"]["true_C_values"]]
    axes[0, 0].plot((min(values), max(values)), (min(values), max(values)), "k--")
    axes[0, 1].axhline(0.0, color="black", linestyle="--")
    axes[1, 1].axhline(0.68, color="black", linestyle="--")
    axes[0, 0].set(xlabel=r"$C_{true}$", ylabel=r"$E[\hat C]$")
    axes[0, 1].set(xlabel=r"$C_{true}$", ylabel="Bias")
    axes[1, 0].set(xlabel=r"$C_{true}$", ylabel=r"Std$(\hat C)$")
    axes[1, 1].set(xlabel=r"$C_{true}$", ylabel="68% coverage", ylim=(0, 1))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("High-stat calibration off-nominal closure")
    fig.tight_layout()
    fig.savefig(output / "06_offnominal_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_summary(metrics: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    columns = ["Direct I gain", "Actual precision gain", "Max |bias|", "Mean RMSE", "Mean coverage", "KL to baseline", "Valid eff.", "Axis error [rad]"]
    cells = [[
        f"{100 * row['direct_gain']:+.1f}%",
        f"{100 * (1.0 / row['actual_sigma_ratio'] - 1.0):+.1f}%",
        f"{row['max_abs_bias']:.4f}", f"{row['mean_rmse']:.4f}", f"{row['mean_coverage']:.3f}",
        f"{row['kl_to_baseline']:.4g}", f"{row['valid_efficiency']:.3f}", f"{row['angular_error']:.3f}",
    ] for row in metrics]
    fig, axis = plt.subplots(figsize=(14.0, 4.3))
    axis.axis("off")
    table = axis.table(cellText=cells, rowLabels=[_label(row["policy"]) for row in metrics], colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    axis.set_title("Final C_nn measurement summary", pad=20)
    fig.tight_layout()
    fig.savefig(output / "07_final_summary.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def make_refresh_study(
    policies: dict[str, ConditionalFlow],
    reference_score: ScoreModel,
    histories: dict[str, list[dict[str, float]]],
    calibration_events: dict[str, torch.Tensor],
    calibrations: dict[str, dict[str, torch.Tensor]],
    calibration_metrics: dict[str, dict[str, float]],
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    root_output: Path,
    checkpoints: Path,
) -> list[dict[str, Any]]:
    names = [name for name in ("baseline", "fisher_dgpo_trust", "iterative_refresh_trust", "iterative_refresh_no_trust") if name in policies]
    if "iterative_refresh_trust" not in names:
        return []
    output = root_output / str(config["refresh"]["output_subdir"])
    output.mkdir(parents=True, exist_ok=True)
    surrogate_models = _load_surrogate_scores(names, reference_score, config, device, checkpoints)
    nominal_C = float(config["physics"]["nominal_C"])
    score_events = generate_events(
        int(config["refresh"]["score_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + 400000),
    )
    independent_scores: dict[str, ScoreModel] = {}
    for index, name in enumerate(names):
        model, _, _ = rebuild_policy_score(
            policies[name], score_events, config, device,
            int(config["seed"]) + 410000 + 1000 * index,
            int(config["refresh"]["score_epochs"]), f"independent score: {_label(name)}",
        )
        independent_scores[name] = model
        torch.save({"method_version": 3, "diagnostic_only": True, "state_dict": model.state_dict()}, output / f"diagnostic_score_{name}.pt")

    fisher_events = generate_events(
        int(config["refresh"]["fisher_events"]), nominal_C, config, device,
        make_generator(device, int(config["seed"]) + 420000),
    )
    per_policy: dict[str, dict[str, float]] = {}
    for index, name in enumerate(names):
        reconstructed = reconstruct_policy(
            policies[name], fisher_events, config,
            make_generator(device, int(config["seed"]) + 430000),
        )
        valid = reconstructed["valid"]
        with torch.no_grad():
            surrogate = surrogate_models[name](reconstructed["y"]) * valid
            independent = independent_scores[name](reconstructed["y"]) * valid
            kl = policies[name].log_prob(reconstructed["action"], fisher_events["context"]) - policies["baseline"].log_prob(reconstructed["action"], fisher_events["context"])
            cosine = (reconstructed["k_a"] * fisher_events["k_true"]).sum(dim=-1).clamp(-1.0, 1.0)
        per_policy[name] = {
            "surrogate_fisher": float(surrogate.square().mean()),
            "policy_score_fisher": float(independent.square().mean()),
            "direct_fisher": float(calibration_metrics[name]["binned_fisher_per_event"]),
            "valid_efficiency": float(valid.float().mean()),
            "kl_to_baseline": float(kl.mean()),
            "angular_error": float(torch.acos(cosine).mean()),
        }

    baseline = per_policy["baseline"]
    nominal_summary = {
        name: min((row for row in summaries if row["policy"] == name), key=lambda row: abs(row["C_true"] - nominal_C))
        for name in names
    }
    baseline_sigma = nominal_summary["baseline"]["std_C_hat"]
    metrics: list[dict[str, Any]] = []
    for name in names:
        rows = [row for row in summaries if row["policy"] == name]
        values = per_policy[name]
        metrics.append({
            "policy": name,
            **values,
            "surrogate_gain": values["surrogate_fisher"] / baseline["surrogate_fisher"] - 1.0,
            "policy_score_gain": values["policy_score_fisher"] / baseline["policy_score_fisher"] - 1.0,
            "direct_gain": values["direct_fisher"] / baseline["direct_fisher"] - 1.0,
            "surrogate_sigma_ratio": np.sqrt(baseline["surrogate_fisher"] / values["surrogate_fisher"]),
            "policy_score_sigma_ratio": np.sqrt(baseline["policy_score_fisher"] / values["policy_score_fisher"]),
            "direct_sigma_ratio": np.sqrt(baseline["direct_fisher"] / values["direct_fisher"]),
            "actual_sigma_ratio": nominal_summary[name]["std_C_hat"] / baseline_sigma,
            "max_abs_bias": float(max(abs(row["bias"]) for row in rows)),
            "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
            "mean_coverage": float(np.mean([row["coverage_68"] for row in rows])),
        })

    with (output / "refresh_study_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "refresh_study_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    history = histories["iterative_refresh_trust"]
    _plot_training(history, config, output)
    _plot_scores(root_output / "score_evolution_iterative_refresh_trust.npz", independent_scores["iterative_refresh_trust"], config, device, output)
    _plot_fisher(metrics, config, output)
    _plot_precision(metrics, config, output)
    _plot_responses(names, calibration_events, calibrations, config, output)
    _plot_offnominal(summaries, names, config, output)
    _plot_summary(metrics, config, output)
    return metrics
