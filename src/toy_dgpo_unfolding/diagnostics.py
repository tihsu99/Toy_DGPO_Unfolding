from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .core import detector_features, make_generator, reconstruct, sample_truth
from .unfolding import fit_poisson_parameter, iterative_bayes, reweighted_reco_templates


Calibration = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _policy_label(name: str) -> str:
    return name.replace("_", " ").title().replace("Dgpo", "DGPO")


def _diagnostic_samples(
    config: dict[str, Any], device: torch.device
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    count = int(config["diagnostics"]["events_per_C"])
    samples: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for index, value in enumerate(config["physics"]["true_C_values"]):
        C = float(value)
        generator = make_generator(device, int(config["seed"]) + 700000 + index)
        truth = sample_truth(count, C, device, generator)
        samples[C] = (truth, detector_features(truth, config["physics"], generator))
    return samples


def _plot_truth_and_detector(
    samples: dict[float, tuple[torch.Tensor, torch.Tensor]], config: dict[str, Any], output: Path
) -> None:
    dpi = int(config["plots"]["dpi"])
    truth_bins = int(config["diagnostics"]["truth_hist_bins"])
    x_grid = np.linspace(-1.0, 1.0, 400)
    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 0.8, len(samples)))
    for color, (C, (truth, _)) in zip(colors, samples.items()):
        values = truth.cpu().numpy()
        axis.hist(values, bins=truth_bins, range=(-1.0, 1.0), density=True, histtype="step", color=color, label=rf"Generated $C={C:.2f}$")
        axis.plot(x_grid, 0.5 * (1.0 + C * x_grid), color=color, linestyle="--", linewidth=1)
    axis.set(xlabel="Generated truth x", ylabel="Density", title="Truth generator validation")
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "00_truth_generation.png", dpi=dpi)
    plt.close(fig)

    feature_labels = (r"$z_1=x+b+\sigma_1\epsilon$", r"$z_2=x^2/2+\sigma_2\epsilon$", r"$z_3=\sin(\pi x)+\sigma_2\epsilon$")
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for feature_index, (axis, label) in enumerate(zip(axes, feature_labels)):
        for C, (_, features) in samples.items():
            axis.hist(features[:, feature_index].cpu().numpy(), bins=truth_bins, density=True, histtype="step", label=rf"$C={C:.2f}$")
        axis.set(xlabel=label, ylabel="Density" if feature_index == 0 else None)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Detector-feature distributions for genuinely off-nominal samples")
    fig.tight_layout()
    fig.savefig(output / "02_z_distributions.png", dpi=dpi)
    plt.close(fig)


def _plot_policy_response(
    calibrations: dict[str, Calibration], config: dict[str, Any], output: Path
) -> list[dict[str, Any]]:
    dpi = int(config["plots"]["dpi"])
    policies = list(calibrations)
    bins = int(config["diagnostics"]["reco_vs_truth_bins"])
    fig, axes = plt.subplots(1, len(policies), figsize=(5.0 * len(policies), 4.4), squeeze=False)
    for axis, name in zip(axes[0], policies):
        truth, reco, _, _, _, _ = calibrations[name]
        image = axis.hist2d(truth, reco, bins=bins, range=((-1.0, 1.0), (-1.0, 1.0)), cmap="viridis")
        axis.plot((-1.0, 1.0), (-1.0, 1.0), "--", color="white", linewidth=1)
        truth_edges = np.linspace(-1.0, 1.0, int(config["training"]["conditional_calibration_bins"]) + 1)
        truth_bin = np.searchsorted(truth_edges, truth, side="right") - 1
        centers = []
        means = []
        for index in range(truth_edges.size - 1):
            selected = truth_bin == index
            if np.any(selected):
                centers.append(float(np.mean(truth[selected])))
                means.append(float(np.mean(reco[selected])))
        axis.plot(centers, means, color="tab:red", marker="o", markersize=3, linewidth=1, label=r"$E[y\mid x\ \mathrm{bin}]$")
        axis.set(xlabel="Truth x", ylabel="Reconstructed y", title=_policy_label(name))
        axis.legend(frameon=False, fontsize=8)
        fig.colorbar(image[3], ax=axis, label="Nominal MC events")
    fig.tight_layout()
    fig.savefig(output / "01_reco_vs_truth.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(policies), figsize=(5.0 * len(policies), 4.4), squeeze=False)
    singular_values: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    rcond = float(config["diagnostics"]["response_rank_rcond"])
    nominal_C = float(config["physics"]["nominal_C"])
    for axis, name in zip(axes[0], policies):
        truth, reco, truth_edges, reco_edges, response, _ = calibrations[name]
        singular_values[name] = np.linalg.svd(response, compute_uv=False)
        image = axis.imshow(response, origin="lower", aspect="auto", extent=(truth_edges[0], truth_edges[-1], reco_edges[0], reco_edges[-1]), cmap="magma")
        axis.set(xlabel="Truth x", ylabel="Reconstructed y", title=_policy_label(name))
        fig.colorbar(image, ax=axis, label=r"$P(y\ \mathrm{bin}\mid x\ \mathrm{bin})$")
        scan = np.array([nominal_C - 0.01, nominal_C, nominal_C + 0.01])
        probabilities = reweighted_reco_templates(truth, reco, reco_edges, nominal_C, scan, 1.0)
        derivative = (probabilities[2] - probabilities[0]) / 0.02
        fisher = float(np.sum(derivative**2 / np.clip(probabilities[1], 1.0e-12, None)))
        values = singular_values[name]
        effective_rank = int(np.sum(values > rcond * values[0]))
        rows.append(
            {
                "policy": name,
                "reco_truth_correlation": float(np.corrcoef(truth, reco)[0, 1]),
                "reconstruction_mse": float(np.mean((reco - truth) ** 2)),
                "response_effective_rank": effective_rank,
                "response_largest_singular_value": float(values[0]),
                "response_smallest_singular_value": float(values[-1]),
                "reconstructed_fisher_per_event": fisher,
            }
        )
    fig.tight_layout()
    fig.savefig(output / "03_response_matrices.png", dpi=dpi)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 5.0))
    for name, values in singular_values.items():
        axis.semilogy(np.arange(1, values.size + 1), values / values[0], marker="o", label=_policy_label(name))
    axis.axhline(rcond, color="black", linestyle="--", label="Effective-rank threshold")
    axis.set(xlabel="Singular-value index", ylabel=r"$s_i/s_{\max}$", title="Response-matrix singular spectrum")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "04_response_singular_values.png", dpi=dpi)
    plt.close(fig)

    with (output / "policy_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_unfolding_and_likelihood(
    policies: dict[str, torch.nn.Module],
    calibrations: dict[str, Calibration],
    samples: dict[float, tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
    output: Path,
) -> None:
    dpi = int(config["plots"]["dpi"])
    representatives = [float(value) for value in config["plots"]["representative_C_values"]]
    sigma = float(config["model"]["policy_sigma"])
    iterations = int(config["unfolding"]["iterations"])
    policy_names = list(policies)
    fig, axes = plt.subplots(len(policy_names), len(representatives), figsize=(6.0 * len(representatives), 3.8 * len(policy_names)), squeeze=False)
    likelihood_curves: dict[float, dict[str, tuple[np.ndarray, float]]] = {C: {} for C in representatives}
    bounds = config["physics"]["physical_C_range"]
    scan = np.linspace(float(bounds[0]), float(bounds[1]), int(config["unfolding"]["scan_points"]))
    nominal_C = float(config["physics"]["nominal_C"])
    for policy_index, name in enumerate(policy_names):
        nominal_truth, nominal_reco, truth_edges, reco_edges, response, prior = calibrations[name]
        centers = 0.5 * (truth_edges[1:] + truth_edges[:-1])
        for truth_index, C_true in enumerate(representatives):
            truth, features = samples[C_true]
            generator = make_generator(device, int(config["seed"]) + 900000 + 100 * policy_index + truth_index)
            reco = reconstruct(policies[name], features, sigma, generator).cpu().numpy()
            observed, _ = np.histogram(reco, bins=reco_edges)
            unfolded = iterative_bayes(observed.astype(float), response, prior, iterations)
            generated_truth, _ = np.histogram(truth.cpu().numpy(), bins=truth_edges)
            nominal_prior = prior * generated_truth.sum() / prior.sum()
            axis = axes[policy_index, truth_index]
            axis.step(centers, generated_truth, where="mid", color="black", label="Generated truth")
            axis.step(centers, unfolded, where="mid", color="tab:red", label=f"Unfolded ({iterations} iterations)")
            axis.step(centers, nominal_prior, where="mid", color="tab:gray", linestyle="--", label=rf"Prior $C_0={nominal_C:.2f}$")
            axis.set(title=rf"{_policy_label(name)}, $C_{{\mathrm{{true}}}}={C_true:.2f}$", xlabel="Truth x", ylabel="Events / bin")
            axis.legend(frameon=False, fontsize=8)

            templates = reweighted_reco_templates(nominal_truth, nominal_reco, reco_edges, nominal_C, scan, float(observed.sum()))
            estimate, _, _, delta = fit_poisson_parameter(observed, templates, scan)
            likelihood_curves[C_true][name] = (delta, estimate)
    fig.tight_layout()
    fig.savefig(output / "05_unfolding_diagnostics.png", dpi=dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(representatives), figsize=(6.0 * len(representatives), 4.2), squeeze=False)
    for axis, C_true in zip(axes[0], representatives):
        for name, (delta, estimate) in likelihood_curves[C_true].items():
            axis.plot(scan, delta, label=rf"{_policy_label(name)}: $\hat C={estimate:.3f}$")
        axis.axhline(1.0, color="black", linestyle=":", label=r"$\Delta(-2\log L)=1$")
        axis.axvline(C_true, color="black", linestyle="--", label=r"$C_{\mathrm{true}}$")
        axis.set(xlabel="C", ylabel=r"$\Delta(-2\log L)$", ylim=(0.0, 10.0), title=rf"$C_{{\mathrm{{true}}}}={C_true:.2f}$")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "06_forward_likelihood_scans.png", dpi=dpi)
    plt.close(fig)


def make_diagnostics(
    policies: dict[str, torch.nn.Module],
    calibrations: dict[str, Calibration],
    config: dict[str, Any],
    device: torch.device,
    output: Path,
) -> None:
    samples = _diagnostic_samples(config, device)
    _plot_truth_and_detector(samples, config, output)
    _plot_policy_response(calibrations, config, output)
    _plot_unfolding_and_likelihood(policies, calibrations, samples, config, device, output)
