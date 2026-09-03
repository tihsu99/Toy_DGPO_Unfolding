from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize

from .core import ScoreModel, make_generator, truth_score
from .flow import ConditionalFlow
from .training import slice_events
from .ztautau import candidate_reconstruction, generate_events


def _label(name: str) -> str:
    labels = {
        "baseline": "Baseline",
        "fisher_dgpo_no_trust": "Frozen score, no trust",
        "fisher_dgpo_trust": "Frozen score, trust",
        "iterative_refresh_trust": "Iterative refresh, trust",
        "iterative_refresh_no_trust": "Iterative refresh, no trust",
        "fisher_dgpo_trust_bias_control": "Fisher DGPO, trust + balance",
    }
    return labels.get(name, name.replace("_", " ").title())


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _angle(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    cosine = (first * second).sum(dim=-1).clamp(-1.0, 1.0)
    return _numpy(torch.rad2deg(torch.acos(cosine)))


def _response(x: np.ndarray, y: np.ndarray, valid: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-1.0, 1.0, bins + 1)
    counts, _, _ = np.histogram2d(y[valid], x[valid], bins=(edges, edges))
    response = counts / np.clip(counts.sum(axis=0, keepdims=True), 1.0, None)
    return response, edges


def _plot_truth_physics(nominal: dict[str, torch.Tensor], config: dict[str, Any], output: Path) -> None:
    dpi = int(config["plots"]["dpi"])
    count = min(int(config["plots"]["diagnostic_events"]), nominal["x"].numel())
    c_a = _numpy(nominal["c_a"][:count])
    c_b = _numpy(nominal["c_b"][:count])
    x_nominal = _numpy(nominal["x"][:count])
    nominal_C = float(config["physics"]["nominal_C"])
    values = [float(value) for value in config["physics"]["true_C_values"]]
    x_grid = np.concatenate((np.linspace(-0.999, -1.0e-3, 600), np.linspace(1.0e-3, 0.999, 600)))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    image = axes[0, 0].hexbin(c_a, c_b, gridsize=45, extent=(-1, 1, -1, 1), mincnt=1, cmap="viridis")
    axes[0, 0].set(xlabel=r"$c_A$", ylabel=r"$c_B$", title=rf"Spin angles at $C_0={nominal_C:.2f}$")
    fig.colorbar(image, ax=axes[0, 0], label="Events")

    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(values)))
    device = nominal["x"].device
    generated_x: dict[float, np.ndarray] = {}
    for index, (C, color) in enumerate(zip(values, colors)):
        generated = generate_events(count, C, config, device, make_generator(device, int(config["seed"]) + 60000 + index))
        generated_x[C] = _numpy(generated["x"])
        axes[0, 1].hist(generated_x[C], bins=55, range=(-1, 1), density=True, histtype="step", color=color, label=rf"$C={C:.2f}$")
        density = 0.5 * (1.0 + C * x_grid) * (-np.log(np.abs(x_grid)))
        axes[0, 1].plot(x_grid, density, color=color, linewidth=0.9, alpha=0.8)
    axes[0, 1].set(xlabel=r"$x=c_Ac_B$", ylabel="Density", title="Generated and analytic truth density", ylim=(0, 3.2))
    axes[0, 1].legend(frameon=False, fontsize=8, ncol=2)

    score = x_grid / (1.0 + nominal_C * x_grid)
    step = 1.0e-5
    log_plus = np.log1p((nominal_C + step) * x_grid)
    log_minus = np.log1p((nominal_C - step) * x_grid)
    finite_difference = (log_plus - log_minus) / (2.0 * step)
    axes[1, 0].plot(x_grid, score, label=r"$t(x)=x/(1+C_0x)$")
    axes[1, 0].plot(x_grid, finite_difference, "--", label="Finite-difference check")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.6)
    axes[1, 0].set(xlabel=r"$x$", ylabel="Truth score", title="Nominal truth score")
    axes[1, 0].legend(frameon=False)

    validation_edges = np.linspace(-1.0, 1.0, int(config["plots"]["reweight_validation_bins"]) + 1)
    nominal_denominator = 1.0 + nominal_C * x_nominal
    reweight_rms = {}
    for C, color in zip(values, colors):
        direct, _ = np.histogram(generated_x[C], bins=validation_edges)
        weighted, _ = np.histogram(x_nominal, bins=validation_edges, weights=(1.0 + C * x_nominal) / nominal_denominator)
        direct = direct / direct.sum()
        weighted = weighted / weighted.sum()
        populated = direct * count > 10
        ratio = weighted[populated] / direct[populated]
        centers = 0.5 * (validation_edges[1:] + validation_edges[:-1])
        axes[1, 1].plot(centers[populated], ratio, marker=".", linewidth=0.8, color=color, label=rf"$C={C:.2f}$")
        reweight_rms[str(C)] = float(np.sqrt(np.mean((ratio - 1.0) ** 2)))
    axes[1, 1].axhline(1.0, color="black", linestyle="--")
    axes[1, 1].set(xlabel=r"$x$", ylabel="Nominal-reweighted / direct", title="Exact nominal reweighting check")
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "00_truth_physics.png", dpi=dpi)
    plt.close(fig)
    validation = {
        "truth_score_finite_difference_max_abs_error": float(np.max(np.abs(score - finite_difference))),
        "reweighted_to_direct_histogram_ratio_rms": reweight_rms,
        "four_momentum_closure_max_GeV": float(nominal["closure"].max()),
        "four_momentum_closure_median_GeV": float(nominal["closure"].median()),
    }
    with (output / "physics_validation.json").open("w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)


def _plot_invisible_kinematics(nominal: dict[str, torch.Tensor], config: dict[str, Any], output: Path) -> None:
    dpi = int(config["plots"]["dpi"])
    count = min(int(config["plots"]["diagnostic_events"]), nominal["x"].numel())
    visible_direction_a = torch.nn.functional.normalize(nominal["visible_obs_a"][:count, 1:], dim=-1)
    visible_direction_b = torch.nn.functional.normalize(nominal["visible_obs_b"][:count, 1:], dim=-1)
    tau_direction_a = nominal["k_true"][:count]
    tau_direction_b = -tau_direction_a
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    axes[0, 0].hist(np.concatenate((_numpy(nominal["r_a"][:count]), _numpy(nominal["r_b"][:count]))), bins=50, density=True, color="tab:blue", alpha=0.75)
    r_grid = np.linspace(0, 1, 300)
    axes[0, 0].plot(r_grid, 6.0 * r_grid * (1.0 - r_grid), color="black", linestyle="--", label=r"Beta$(2,2)$")
    axes[0, 0].set(xlabel="Visible energy fraction r", ylabel="Density", title="Tau-rest-frame visible fraction")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].hist(np.concatenate((_numpy(nominal["invisible_mass_a"][:count]), _numpy(nominal["invisible_mass_b"][:count]))), bins=50, color="tab:orange", alpha=0.75)
    axes[0, 1].set(xlabel=r"Invisible-system mass [GeV]", ylabel="Events", title="Invisible invariant mass")

    angular = np.concatenate((_angle(visible_direction_a, tau_direction_a), _angle(visible_direction_b, tau_direction_b)))
    axes[1, 0].hist(angular, bins=55, color="tab:green", alpha=0.75)
    axes[1, 0].set(xlabel="Visible--true tau angle [deg]", ylabel="Events", title="Invisible reconstruction ambiguity")

    closure = np.clip(_numpy(nominal["closure"][:count]), 1.0e-12, None)
    axes[1, 1].hist(np.log10(closure), bins=55, color="tab:red", alpha=0.75)
    axes[1, 1].set(xlabel=r"$\log_{10}$ closure residual [GeV]", ylabel="Events", title="Four-momentum closure")
    fig.tight_layout()
    fig.savefig(output / "01_invisible_kinematics.png", dpi=dpi)
    plt.close(fig)


def _plot_observed_z_by_C(config: dict[str, Any], device: torch.device, output: Path) -> None:
    count = int(config["plots"]["diagnostic_events"])
    values = [float(value) for value in config["physics"]["true_C_values"]]
    samples = []
    for index, C in enumerate(values):
        events = generate_events(count, C, config, device, make_generator(device, int(config["seed"]) + 65000 + index))
        direction_a = torch.nn.functional.normalize(events["visible_obs_a"][:, 1:], dim=-1)
        direction_b = torch.nn.functional.normalize(events["visible_obs_b"][:, 1:], dim=-1)
        opening = torch.rad2deg(torch.acos((direction_a * direction_b).sum(dim=-1).clamp(-1.0, 1.0)))
        samples.append((events["visible_obs_a"][:, 0], events["visible_obs_b"][:, 0], opening, events["seed"][:, 2]))
    titles = ("Observed visible A energy [GeV]", "Observed visible B energy [GeV]", "Observed opening angle [deg]", "Observed seed z component")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    for axis, component, title in zip(axes.flat, range(4), titles):
        for C, sample in zip(values, samples):
            axis.hist(_numpy(sample[component]), bins=45, density=True, histtype="step", label=rf"$C={C:.2f}$")
        axis.set(xlabel=title, ylabel="Density")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(r"Observed input $z=(p_{A,obs},p_{B,obs})$ across physics hypotheses")
    fig.tight_layout()
    fig.savefig(output / "01b_observed_z_by_C.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_baseline_reconstruction(
    calibration_events: dict[str, torch.Tensor], calibration: dict[str, torch.Tensor],
    config: dict[str, Any], output: Path,
) -> None:
    dpi = int(config["plots"]["dpi"])
    valid = _numpy(calibration["valid"]).astype(bool)
    x = _numpy(calibration_events["x"])
    y = _numpy(calibration["y"])
    bins = int(config["plots"]["response_bins"])
    response, edges = _response(x, y, valid, bins)
    angular = _angle(calibration["k_a"][calibration["valid"]], calibration_events["k_true"][calibration["valid"]])
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    axes[0].hist(angular, bins=50, color="tab:blue", alpha=0.8)
    axes[0].set(xlabel="Tau-axis angular error [deg]", ylabel="Events", title="Baseline axis reconstruction")
    image = axes[1].imshow(response, origin="lower", aspect="auto", extent=(-1, 1, -1, 1), cmap="magma", vmin=0.0)
    axes[1].set(xlabel="Truth x", ylabel="Reconstructed y", title=r"$R_{\rm ref}(y\mid x)$")
    fig.colorbar(image, ax=axes[1], label="Probability per y bin")
    truth_slices = [(-1.0, -0.5), (-0.2, 0.2), (0.5, 1.0)]
    for low, high in truth_slices:
        selected = valid & (x >= low) & (x < high)
        if np.any(selected):
            axes[2].hist(y[selected], bins=edges, density=True, histtype="step", linewidth=1.6, label=rf"${low:.1f}\leq x<{high:.1f}$")
    axes[2].set(xlabel="Reconstructed y", ylabel="Conditional density", title="Conditional response slices")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "02_baseline_reconstruction.png", dpi=dpi)
    plt.close(fig)


def _plot_score_fisher(
    calibration_events: dict[str, torch.Tensor], score_model: ScoreModel,
    calibration: dict[str, torch.Tensor], fisher: dict[str, Any], config: dict[str, Any], output: Path,
) -> None:
    dpi = int(config["plots"]["dpi"])
    valid = _numpy(calibration["valid"]).astype(bool)
    y = _numpy(calibration["y"])
    target = _numpy(truth_score(calibration_events["x"], float(config["physics"]["nominal_C"])))
    edges = np.linspace(-1.0, 1.0, int(config["fisher_validation"]["reco_bins"]) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    indices = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, centers.size - 1)
    empirical = np.full(centers.shape, np.nan)
    empirical_error = np.full(centers.shape, np.nan)
    for index in range(centers.size):
        selected = valid & (indices == index)
        if selected.sum() > 2:
            empirical[index] = target[selected].mean()
            empirical_error[index] = target[selected].std(ddof=1) / np.sqrt(selected.sum())
    grid = torch.linspace(-1.0, 1.0, 400, device=calibration_events["x"].device)
    learned = _numpy(score_model(grid))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    axes[0].errorbar(centers, empirical, yerr=empirical_error, fmt="o", markersize=3, label=r"Empirical $E[t(X)\mid Y]$")
    axes[0].plot(_numpy(grid), learned, color="tab:orange", label=r"Frozen $s_{\rm ref}(y)$")
    axes[0].axhline(0.0, color="black", linewidth=0.6)
    axes[0].set(xlabel="Reconstructed y", ylabel="Score", title="Reference-score regression")
    axes[0].legend(frameon=False, fontsize=8)

    sigmas = [fisher["sigma_score"], fisher["sigma_binned"], fisher["sigma_pseudo_experiments"]]
    names = [r"$1/\sqrt{I_{\rm score}}$", r"$1/\sqrt{I_{\rm bin}}$", "Pseudo-experiment std"]
    axes[1].bar(names, sigmas, color=("tab:blue", "tab:orange", "tab:green"))
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(ylabel=r"Nominal $\sigma_C$", title=f"Fisher closure: {'PASS' if fisher['passed'] else 'FAIL'}")
    axes[1].text(0.03, 0.95, f"relative spread = {fisher['relative_spread']:.3f}\ntolerance = {fisher['relative_tolerance']:.3f}", transform=axes[1].transAxes, va="top")

    estimates = np.asarray(fisher["pseudo_estimates"])
    axes[2].hist(estimates, bins=max(5, min(30, estimates.size // 3)), color="tab:green", alpha=0.75)
    axes[2].axvline(float(config["physics"]["nominal_C"]), color="black", linestyle="--", label=r"$C_0$")
    axes[2].set(xlabel=r"Nominal pseudo-experiment $\hat C$", ylabel="Toys", title="Pre-DGPO likelihood closure")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "03_score_fisher_closure.png", dpi=dpi)
    plt.close(fig)


def plot_pre_dgpo(
    nominal: dict[str, torch.Tensor], calibration_events: dict[str, torch.Tensor],
    score_model: ScoreModel, calibration: dict[str, torch.Tensor], fisher: dict[str, Any],
    config: dict[str, Any], output: Path,
) -> None:
    _plot_truth_physics(nominal, config, output)
    _plot_invisible_kinematics(nominal, config, output)
    _plot_observed_z_by_C(config, nominal["x"].device, output)
    _plot_baseline_reconstruction(calibration_events, calibration, config, output)
    _plot_score_fisher(calibration_events, score_model, calibration, fisher, config, output)


@torch.no_grad()
def _plot_candidate_display(
    events: dict[str, torch.Tensor], policies: dict[str, ConditionalFlow], score_model: ScoreModel,
    calibrations: dict[str, dict[str, torch.Tensor]], config: dict[str, Any], device: torch.device, output: Path,
) -> None:
    name = "fisher_dgpo_trust" if "fisher_dgpo_trust" in policies else list(policies)[-1]
    count = min(int(config["plots"]["candidate_display_events"]), events["x"].numel())
    selected = slice_events(events, slice(0, count))
    group_size = int(config["dgpo"]["group_size"])
    generator = make_generator(device, int(config["seed"]) + 71000)
    actions, _ = policies[name].sample(selected["context"], group_size, generator)
    candidates = candidate_reconstruction(selected, actions, config)
    score = score_model(candidates["y"]) * candidates["valid"]
    information = score.square()
    baseline = slice_events(calibrations["baseline"], slice(0, count))
    reference_information = calibrations["baseline"]["information"].sum()
    reward = 0.5 * torch.log((reference_information - baseline["information"][:, None] + information) / reference_information)
    reward_np = _numpy(reward)
    normalization = Normalize(vmin=float(reward_np.min()), vmax=float(reward_np.max()) + 1.0e-12)
    figure = plt.figure(figsize=(5.3 * count, 5.4))
    for event_index in range(count):
        axis = figure.add_subplot(1, count, event_index + 1, projection="3d")
        origins = np.zeros((3, 3))
        directions = np.stack((_numpy(torch.nn.functional.normalize(selected["visible_obs_a"][event_index, 1:], dim=-1)), _numpy(torch.nn.functional.normalize(selected["visible_obs_b"][event_index, 1:], dim=-1)), _numpy(selected["seed"][event_index])))
        colors = ("tab:blue", "tab:orange", "black")
        labels = ("Observed visible A", "Observed visible B", "Seed")
        for origin, direction, color, label in zip(origins, directions, colors, labels):
            axis.quiver(*origin, *direction, color=color, linewidth=2.0, label=label)
        truth = _numpy(selected["k_true"][event_index])
        axis.quiver(0, 0, 0, *truth, color="magenta", linewidth=2.0, linestyle="--", label="True tau (validation)")
        candidate_directions = _numpy(candidates["k_a"][event_index])
        for candidate_index, direction in enumerate(candidate_directions):
            color = plt.get_cmap("coolwarm")(normalization(reward_np[event_index, candidate_index]))
            axis.quiver(0, 0, 0, *direction, color=color, linewidth=1.2)
        best = int(np.argmax(reward_np[event_index]))
        details = "\n".join(
            f"{index}: y={float(candidates['y'][event_index, index].detach()):+.2f}, s2={float(information[event_index, index].detach()):.3f}, R={float(reward[event_index, index].detach()):+.2e}"
            for index in range(group_size)
        )
        axis.text2D(0.01, -0.28, details, transform=axis.transAxes, fontsize=6, family="monospace")
        axis.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="z", title=f"Event {event_index}; best candidate {best}")
        if event_index == 0:
            axis.legend(frameon=False, fontsize=6, loc="upper left")
    scalar = plt.cm.ScalarMappable(norm=normalization, cmap="coolwarm")
    color_axis = figure.add_axes((0.93, 0.32, 0.012, 0.38))
    figure.colorbar(scalar, cax=color_axis, label="Replacement reward")
    figure.suptitle(f"Candidate geometry: {_label(name)}", y=0.99)
    figure.subplots_adjust(left=0.03, right=0.90, bottom=0.30, top=0.90, wspace=0.25)
    figure.savefig(output / "04_candidate_event_display.png", dpi=int(config["plots"]["dpi"]), bbox_inches="tight")
    plt.close(figure)


def _plot_training(histories: dict[str, list[dict[str, float]]], config: dict[str, Any], output: Path) -> None:
    metrics = [
        ("fisher", "Fisher"), ("predicted_sigma", r"Predicted $\sigma_C$"),
        ("reward", "Replacement reward"), ("kl_to_reference", r"$D_{KL}(q_\phi||q_{ref})$"),
        ("invalid_fraction", "Invalid fraction"), ("angular_error", "Tau-axis error [rad]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.5))
    for axis, (key, title) in zip(axes.flat, metrics):
        for name, rows in histories.items():
            if rows:
                axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], marker="o", markersize=3, label=_label(name))
        axis.set(xlabel="Epoch", ylabel=title)
        axis.grid(alpha=0.2)
    if histories:
        axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("DGPO training diagnostics (axis error is validation only)")
    fig.tight_layout()
    fig.savefig(output / "05_training.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_responses(
    events: dict[str, torch.Tensor], calibrations: dict[str, dict[str, torch.Tensor]], config: dict[str, Any], output: Path,
) -> None:
    names = list(calibrations)
    x = _numpy(events["x"])
    bins = int(config["plots"]["response_bins"])
    matrices = {}
    for name, calibration in calibrations.items():
        matrices[name], _ = _response(x, _numpy(calibration["y"]), _numpy(calibration["valid"]).astype(bool), bins)
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    difference_max = max(float(np.abs(matrix - matrices["baseline"]).max()) for matrix in matrices.values())
    fig, axes = plt.subplots(2, len(names), figsize=(5.0 * len(names), 8.2), squeeze=False)
    for index, name in enumerate(names):
        image = axes[0, index].imshow(matrices[name], origin="lower", aspect="auto", extent=(-1, 1, -1, 1), cmap="magma", vmin=0.0, vmax=maximum)
        axes[0, index].set(title=_label(name), xlabel="Truth x", ylabel="Reconstructed y")
        difference = matrices[name] - matrices["baseline"]
        diff_image = axes[1, index].imshow(difference, origin="lower", aspect="auto", extent=(-1, 1, -1, 1), cmap="coolwarm", vmin=-difference_max, vmax=difference_max)
        axes[1, index].set(title="Difference from baseline", xlabel="Truth x", ylabel="Reconstructed y")
    color_axis = fig.add_axes((0.93, 0.56, 0.012, 0.32))
    difference_axis = fig.add_axes((0.93, 0.12, 0.012, 0.32))
    fig.colorbar(image, cax=color_axis, label="Probability per y bin")
    fig.colorbar(diff_image, cax=difference_axis, label="Response difference")
    fig.suptitle("Matched response comparison (same nominal events and latent random stream)")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.07, top=0.92, wspace=0.25, hspace=0.28)
    fig.savefig(output / "06_response_before_after.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_closure(summaries: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    names = sorted({row["policy"] for row in summaries})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    for name in names:
        selected = sorted((row for row in summaries if row["policy"] == name), key=lambda row: row["C_true"])
        truth = np.array([row["C_true"] for row in selected])
        axes[0, 0].errorbar(truth, [row["mean_C_hat"] for row in selected], yerr=[row["mean_C_hat_error"] for row in selected], marker="o", capsize=3, label=_label(name))
        axes[0, 1].plot(truth, [row["bias"] for row in selected], marker="o", label=_label(name))
        axes[1, 0].plot(truth, [row["pull_mean"] for row in selected], marker="o", label=_label(name))
        coverage = np.array([row["coverage_68"] for row in selected])
        lower = coverage - np.array([row["coverage_68_low"] for row in selected])
        upper = np.array([row["coverage_68_high"] for row in selected]) - coverage
        axes[1, 1].errorbar(truth, coverage, yerr=np.stack((lower, upper)), marker="o", capsize=3, label=_label(name))
    bounds = [float(value) for value in config["physics"]["true_C_values"]]
    axes[0, 0].plot((min(bounds), max(bounds)), (min(bounds), max(bounds)), "k--", label="Identity")
    axes[0, 0].set(xlabel=r"$C_{true}$", ylabel=r"$E[\hat C]$", title="Linearity")
    axes[0, 1].axhline(0.0, color="black", linestyle="--")
    axes[0, 1].set(xlabel=r"$C_{true}$", ylabel=r"$E[\hat C]-C_{true}$", title="Bias")
    axes[1, 0].axhline(0.0, color="black", linestyle="--")
    axes[1, 0].set(xlabel=r"$C_{true}$", ylabel="Pull mean", title="Normalized bias")
    axes[1, 1].axhline(0.68, color="black", linestyle="--", label="68% reference")
    axes[1, 1].set(xlabel=r"$C_{true}$", ylabel="Empirical coverage", title="68% interval coverage", ylim=(0, 1))
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(r"Off-nominal Poisson forward-folding closure; training only at $C_0=0.60$")
    fig.tight_layout()
    fig.savefig(output / "07_offnominal_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_dashboard(
    events: dict[str, torch.Tensor], calibrations: dict[str, dict[str, torch.Tensor]],
    histories: dict[str, list[dict[str, float]]], summaries: list[dict[str, Any]], config: dict[str, Any], output: Path,
) -> None:
    names = list(calibrations)
    baseline_fisher = float(calibrations["baseline"]["information"].mean())
    rows = []
    for name in names:
        selected = [row for row in summaries if row["policy"] == name]
        valid = calibrations[name]["valid"]
        angular = _angle(calibrations[name]["k_a"][valid], events["k_true"][valid])
        final_history = histories.get(name, [])[-1] if histories.get(name) else {}
        fisher = float(calibrations[name]["information"].mean())
        rows.append({
            "policy": name,
            "fisher_improvement": fisher / baseline_fisher - 1.0,
            "mean_precision": float(np.mean([row["std_C_hat"] for row in selected])),
            "max_abs_bias": float(np.max(np.abs([row["bias"] for row in selected]))),
            "mean_rmse": float(np.mean([row["rmse"] for row in selected])),
            "mean_pull_width": float(np.mean([row["pull_std"] for row in selected])),
            "mean_coverage": float(np.mean([row["coverage_68"] for row in selected])),
            "kl_to_reference": float(final_history.get("kl_to_reference", 0.0)),
            "invalid_fraction": 1.0 - float(valid.float().mean()),
            "mean_axis_error_deg": float(np.mean(angular)),
        })
    baseline_precision = next(row["mean_precision"] for row in rows if row["policy"] == "baseline")
    baseline_axis_error = next(row["mean_axis_error_deg"] for row in rows if row["policy"] == "baseline")
    for row in rows:
        row["measurement_precision_improved"] = row["mean_precision"] < baseline_precision
        row["axis_error_improved"] = row["mean_axis_error_deg"] < baseline_axis_error
    with (output / "policy_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ["Fisher gain", "PE std", "Max |bias|", "RMSE", "Pull width", "Coverage", "KL", "Invalid", "Axis err [deg]"]
    table_values = [[
        f"{100.0 * row['fisher_improvement']:+.1f}%", f"{row['mean_precision']:.4f}", f"{row['max_abs_bias']:.4f}",
        f"{row['mean_rmse']:.4f}", f"{row['mean_pull_width']:.3f}", f"{row['mean_coverage']:.3f}",
        f"{row['kl_to_reference']:.3g}", f"{row['invalid_fraction']:.3f}", f"{row['mean_axis_error_deg']:.2f}",
    ] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(15.0, 7.0), gridspec_kw={"height_ratios": [1.0, 1.2]})
    axes[0].axis("off")
    table = axes[0].table(cellText=table_values, rowLabels=[_label(row["policy"]) for row in rows], colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    positions = np.arange(len(rows))
    precision_gain = [100.0 * (baseline_precision / row["mean_precision"] - 1.0) for row in rows]
    axis_error = [row["mean_axis_error_deg"] for row in rows]
    width = 0.36
    axes[1].bar(positions - width / 2, precision_gain, width, label="Measurement precision gain [%]")
    second = axes[1].twinx()
    second.bar(positions + width / 2, axis_error, width, color="tab:orange", label="Tau-axis error [deg]")
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set(xticks=positions, xticklabels=[_label(row["policy"]) for row in rows], ylabel="Precision gain [%]")
    second.set_ylabel("Mean tau-axis error [deg]")
    axes[1].legend(frameon=False, loc="upper left")
    second.legend(frameon=False, loc="upper right")
    fig.suptitle("Measurement precision can improve without improving ordinary reconstruction error")
    fig.tight_layout()
    fig.savefig(output / "08_final_dashboard.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def make_all_plots(
    nominal: dict[str, torch.Tensor], calibration_events: dict[str, torch.Tensor],
    policies: dict[str, ConditionalFlow], score_model: ScoreModel,
    calibrations: dict[str, dict[str, torch.Tensor]], histories: dict[str, list[dict[str, float]]],
    fisher: dict[str, Any], summaries: list[dict[str, Any]],
    config: dict[str, Any], device: torch.device, output: Path,
) -> None:
    plot_pre_dgpo(
        nominal, calibration_events, score_model, calibrations["baseline"], fisher, config, output,
    )
    _plot_candidate_display(calibration_events, policies, score_model, calibrations, config, device, output)
    _plot_training(histories, config, output)
    _plot_responses(calibration_events, calibrations, config, output)
    _plot_closure(summaries, config, output)
    _plot_dashboard(calibration_events, calibrations, histories, summaries, config, output)
