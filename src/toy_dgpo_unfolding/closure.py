from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .core import make_generator
from .flow import ConditionalFlow
from .inference import fit_poisson, poisson_score_components, reweighted_templates
from .training import reconstruct_policy, slice_events
from .ztautau import generate_events


def _label(name: str) -> str:
    return {
        "baseline": "Baseline",
        "fisher_dgpo_no_trust": "No trust",
        "fisher_dgpo_trust": "Trust",
        "iterative_refresh_trust": "Iterative refresh trust",
        "iterative_refresh_no_trust": "Iterative refresh no trust",
    }.get(name, name.replace("_", " ").title())


def _histogram(reconstructed: dict[str, torch.Tensor], edges: np.ndarray) -> np.ndarray:
    valid = reconstructed["valid"].detach().cpu().numpy().astype(bool)
    y = reconstructed["y"].detach().cpu().numpy()
    return np.histogram(y[valid], bins=edges)[0].astype(float)


def _scan(config: dict[str, Any]) -> np.ndarray:
    lower, upper = config["inference"]["scan_range"]
    return np.linspace(float(lower), float(upper), int(config["inference"]["scan_points"]))


def _build_high_statistics_predictions(
    config: dict[str, Any],
    device: torch.device,
    policies: dict[str, ConditionalFlow],
    names: list[str],
    edges: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[float, dict[str, np.ndarray]]]:
    settings = config["closure"]
    total_events = int(settings["events"])
    chunk_size = int(settings["chunk_size"])
    nominal_C = float(config["physics"]["nominal_C"])
    true_values = [float(value) for value in config["physics"]["true_C_values"]]
    bins = edges.size - 1
    nominal_counts = {name: np.zeros(bins) for name in names}
    nominal_derivatives = {name: np.zeros(bins) for name in names}
    nominal_second_moments = {name: np.zeros(bins) for name in names}
    direct = {value: {name: np.zeros(bins) for name in names} for value in true_values}
    chunks = (total_events + chunk_size - 1) // chunk_size
    progress = tqdm(total=chunks * (1 + len(true_values)) * len(names), desc="statistical closure", unit="policy-chunk")

    for chunk in range(chunks):
        count = min(chunk_size, total_events - chunk * chunk_size)
        events = generate_events(
            count, nominal_C, config, device,
            make_generator(device, int(config["seed"]) + 700000 + chunk),
        )
        x = events["x"].detach().cpu().numpy()
        truth_score = x / (1.0 + nominal_C * x)
        for name in names:
            reconstructed = reconstruct_policy(
                policies[name], events, config,
                make_generator(device, int(config["seed"]) + 710000 + chunk),
            )
            valid = reconstructed["valid"].detach().cpu().numpy().astype(bool)
            y = reconstructed["y"].detach().cpu().numpy()
            indices = np.clip(np.searchsorted(edges, y[valid], side="right") - 1, 0, bins - 1)
            nominal_counts[name] += np.bincount(indices, minlength=bins)
            nominal_derivatives[name] += np.bincount(indices, weights=truth_score[valid], minlength=bins)
            nominal_second_moments[name] += np.bincount(indices, weights=truth_score[valid] ** 2, minlength=bins)
            progress.update()

    for value_index, value in enumerate(true_values):
        for chunk in range(chunks):
            count = min(chunk_size, total_events - chunk * chunk_size)
            events = generate_events(
                count, value, config, device,
                make_generator(device, int(config["seed"]) + 720000 + 10000 * value_index + chunk),
            )
            for name in names:
                reconstructed = reconstruct_policy(
                    policies[name], events, config,
                    make_generator(device, int(config["seed"]) + 730000 + 10000 * value_index + chunk),
                )
                direct[value][name] += _histogram(reconstructed, edges)
                progress.update()
    progress.close()
    return nominal_counts, nominal_derivatives, nominal_second_moments, direct


def _current_calibration_templates(
    config: dict[str, Any],
    device: torch.device,
    policies: dict[str, ConditionalFlow],
    names: list[str],
    edges: np.ndarray,
    scan: np.ndarray,
    target_exposure: float,
) -> dict[str, np.ndarray]:
    count = int(config["data"]["nominal_events"])
    nominal_C = float(config["physics"]["nominal_C"])
    events = generate_events(count, nominal_C, config, device, make_generator(device, int(config["seed"])))
    train_end = int(count * float(config["data"]["train_fraction"]))
    score_end = train_end + int(count * float(config["data"]["score_fraction"]))
    calibration_events = slice_events(events, slice(score_end, count))
    x = calibration_events["x"].detach().cpu().numpy()
    templates: dict[str, np.ndarray] = {}
    for name in tqdm(names, desc="current calibration templates", unit="policy"):
        reconstructed = reconstruct_policy(
            policies[name], calibration_events, config,
            make_generator(device, int(config["seed"]) + 50000),
        )
        templates[name] = reweighted_templates(
            x,
            reconstructed["y"].detach().cpu().numpy(),
            reconstructed["valid"].detach().cpu().numpy().astype(bool),
            edges,
            nominal_C,
            scan,
            target_exposure,
        )
    return templates


def _plot_extended_score(rows: list[dict[str, Any]], config: dict[str, Any], output: Path) -> None:
    positions = np.arange(len(rows))
    labels = [_label(row["policy"]) for row in rows]
    fields = [
        ("score_sum", r"$\sum_i s_\pi(y_i)$"),
        ("yield_derivative", r"$\Lambda'_\pi(C_0)$"),
        ("full_score", r"$U_{full}$"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for axis, (field, title) in zip(axes, fields):
        values = [row[field] for row in rows]
        axis.bar(positions, values)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(xticks=positions, xticklabels=labels, ylabel=title)
        axis.tick_params(axis="x", rotation=15)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.3g}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    fig.suptitle("Extended-Poisson score closure on an independent nominal sample")
    fig.tight_layout()
    fig.savefig(output / "19_extended_score_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_templates(
    names: list[str], true_values: list[float], centers: np.ndarray,
    direct: dict[float, dict[str, np.ndarray]], reweighted: dict[float, dict[str, np.ndarray]],
    config: dict[str, Any], output: Path,
) -> None:
    fig, axes = plt.subplots(2 * len(names), len(true_values), figsize=(4.0 * len(true_values), 3.2 * len(names)), sharex=True)
    for policy_index, name in enumerate(names):
        for value_index, value in enumerate(true_values):
            main = axes[2 * policy_index, value_index]
            ratio_axis = axes[2 * policy_index + 1, value_index]
            direct_values = direct[value][name]
            rw_values = reweighted[value][name]
            main.step(centers, direct_values, where="mid", label="Direct", color="black")
            main.step(centers, rw_values, where="mid", label="Nominal reweighted", color="tab:orange")
            ratio = np.divide(direct_values, rw_values, out=np.full_like(direct_values, np.nan), where=rw_values > 0.0)
            ratio_axis.axhline(1.0, color="black", linewidth=0.7)
            ratio_axis.plot(centers, ratio, marker=".", linewidth=0.8)
            ratio_axis.set_ylim(0.8, 1.2)
            main.set_title(rf"{_label(name)}, $C={value:.2f}$")
            main.set_ylabel("Expected selected yield / bin")
            ratio_axis.set(xlabel="Reconstructed y", ylabel="Direct / rw")
            if policy_index == 0 and value_index == 0:
                main.legend(frameon=False, fontsize=8)
    fig.suptitle("Absolute direct versus nominal-reweighted templates")
    fig.tight_layout()
    fig.savefig(output / "20_direct_vs_reweighted_templates.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_acceptance(rows: list[dict[str, Any]], names: list[str], config: dict[str, Any], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for name in names:
        selected = sorted((row for row in rows if row["policy"] == name), key=lambda row: row["C_true"])
        truth = [row["C_true"] for row in selected]
        axis.plot(truth, [row["direct_efficiency"] for row in selected], marker="o", label=f"{_label(name)} direct")
        axis.plot(truth, [row["reweighted_efficiency"] for row in selected], marker="x", linestyle="--", label=f"{_label(name)} reweighted")
    axis.set(xlabel=r"$C$", ylabel="Absolute valid efficiency", title="Acceptance reweighting closure")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "21_acceptance_reweighting_closure.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_asimov(rows: list[dict[str, Any]], names: list[str], config: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    values = sorted({row["C_true"] for row in rows})
    axes[0].plot(values, values, "k--", label="Identity")
    axes[1].axhline(0.0, color="black", linestyle="--")
    for name in names:
        selected = sorted((row for row in rows if row["policy"] == name), key=lambda row: row["C_true"])
        truth = [row["C_true"] for row in selected]
        axes[0].plot(truth, [row["asimov_C_hat"] for row in selected], marker="o", label=f"{_label(name)} high-stat")
        axes[0].plot(truth, [row["current_calibration_C_hat"] for row in selected], marker="x", linestyle=":", label=f"{_label(name)} current cal.")
        axes[1].plot(truth, [row["asimov_bias"] for row in selected], marker="o", label=f"{_label(name)} high-stat")
        axes[1].plot(truth, [row["current_calibration_bias"] for row in selected], marker="x", linestyle=":", label=f"{_label(name)} current cal.")
    axes[0].set(xlabel=r"$C_{true}$", ylabel=r"$\hat C_{Asimov}$", title="Asimov linearity")
    axes[1].set(xlabel=r"$C_{true}$", ylabel=r"$\hat C_{Asimov}-C_{true}$", title="Asimov residual")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "22_asimov_linearity.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _plot_likelihood_examples(
    names: list[str], scan: np.ndarray, likelihoods: dict[float, dict[str, np.ndarray]],
    config: dict[str, Any], output: Path,
) -> None:
    examples = [value for value in (0.4, 0.8) if value in likelihoods]
    fig, axes = plt.subplots(len(examples), len(names), figsize=(4.5 * len(names), 3.6 * len(examples)), squeeze=False, sharex=True)
    for row, value in enumerate(examples):
        for column, name in enumerate(names):
            axis = axes[row, column]
            axis.plot(scan, likelihoods[value][name], color="tab:blue")
            axis.axvline(value, color="black", linestyle="--", label="Truth")
            axis.axhline(1.0, color="gray", linestyle=":")
            axis.set(title=f"{_label(name)}, C={value:.2f}", xlabel="C", ylabel=r"$-2\Delta\log L$")
            if row == 0 and column == 0:
                axis.legend(frameon=False)
    fig.suptitle("High-statistics direct Asimov data fitted with nominal-reweighted templates")
    fig.tight_layout()
    fig.savefig(output / "23_asimov_likelihood_examples.png", dpi=int(config["plots"]["dpi"]))
    plt.close(fig)


def _read_summary(output: Path) -> dict[tuple[str, float], dict[str, float]]:
    path = output / "summary.csv"
    if not path.exists():
        return {}
    result: dict[tuple[str, float], dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {key: float(value) for key, value in raw.items() if key != "policy" and value}
            result[(raw["policy"], row["C_true"])] = row
    return result


def _update_report(
    output: Path,
    score_rows: list[dict[str, Any]],
    closure_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    path = output / "diagnosis.md"
    previous = path.read_text(encoding="utf-8") if path.exists() else "# Frozen-policy statistical diagnosis\n"
    for heading in (
        "## 7. Score balance versus fitted bias",
        "## 7. Extended-Poisson score balance versus fitted bias",
        "## 7. Corrected extended-Poisson score closure",
    ):
        if heading in previous:
            previous = previous.split(heading, 1)[0].rstrip() + "\n"
    settings = config["closure"]
    score_pass = all(abs(row["full_score_per_event"]) <= float(settings["score_per_event_tolerance"]) for row in score_rows)
    yield_pass = all(abs(row["yield_relative_difference"]) <= float(settings["yield_relative_tolerance"]) for row in closure_rows)
    shape_pass = all(row["shape_total_variation"] <= float(settings["shape_total_variation_tolerance"]) for row in closure_rows)
    asimov_pass = all(abs(row["asimov_bias"]) <= float(settings["asimov_bias_tolerance"]) for row in closure_rows)
    asimov_compatible = all(abs(row["asimov_MC_pull"]) <= float(settings["asimov_sigma_tolerance"]) for row in closure_rows)

    lines = [previous.rstrip(), "", "## 7. Corrected extended-Poisson score closure", ""]
    lines.append("The previous uncorrected U/I diagnostic was invalid because it omitted the extended-Poisson compensator -Lambda'(C). The corrected binned intensity score uses U_full = sum_i s_pi(y_i) - Lambda'(C); its Fisher sum(mu'^2/mu) already contains rate and shape information, so no extra rate-Fisher term is added.")
    lines.append("")
    for row in score_rows:
        lines.append(f"- **{_label(row['policy'])}:** score sum {row['score_sum']:+.6g}, Lambda' {row['yield_derivative']:+.6g}, U_full/N {row['full_score_per_event']:+.3g}, Z_bias {row['Z_bias']:.3f}.")
    lines.extend(["", f"**Does E[U_full] approximately vanish? {'Yes' if score_pass else 'No'}** at the configured per-generated-event tolerance.", ""])

    lines.extend(["## 8. Direct versus nominal-reweighted closure", ""])
    lines.append(f"- Absolute selected yield closure: **{'PASS' if yield_pass else 'FAIL'}**.")
    lines.append(f"- Selected reconstructed-y shape closure: **{'PASS' if shape_pass else 'FAIL'}**.")
    lines.append(f"- High-statistics Asimov absolute-linearity target: **{'PASS' if asimov_pass else 'FAIL'}**.")
    lines.append(f"- Direct-versus-reweighted Asimov MC compatibility: **{'PASS' if asimov_compatible else 'FAIL'}**.")
    lines.append("")
    for row in closure_rows:
        lines.append(f"- **{_label(row['policy'])}, C={row['C_true']:.2f}:** yield difference {100 * row['yield_relative_difference']:+.3f}%, shape TV {row['shape_total_variation']:.4f}, high-stat Asimov bias {row['asimov_bias']:+.4f} ({row['asimov_MC_pull']:+.2f} MC sigma), current-calibration Asimov bias {row['current_calibration_bias']:+.4f}.")

    lines.extend(["", "## 9. Statistical interpretation", ""])
    if not yield_pass:
        conclusion = "The reweighting identity fails in the absolute efficiency component. Inspect policy validity and its C dependence before any training change."
    elif not shape_pass:
        conclusion = "The absolute yield closes but the selected-y response shape does not; the policy response construction or bin assignment breaks the reweighting identity."
    elif not asimov_pass and asimov_compatible:
        conclusion = "The direct and reweighted predictions are statistically compatible, but this run does not meet the requested absolute Asimov precision. Increase closure.events before assigning an implementation failure."
    elif not asimov_pass:
        conclusion = "Direct and reweighted histograms pass the configured component tests, but the Asimov fit fails; the scan or likelihood fit is the remaining broken component."
    else:
        current_failures = [row for row in closure_rows if abs(row["current_calibration_bias"]) > float(settings["asimov_bias_tolerance"])]
        if current_failures:
            conclusion = "The exact high-statistics reweighting identity passes. The remaining displacement appears when the fixed finite production calibration template is used, identifying calibration-template MC statistics rather than DGPO physics bias as the component to debug."
        else:
            conclusion = "The exact reweighting identity and the current calibration Asimov test pass; no template implementation component breaks closure. Remaining pseudo-experiment offsets must be judged against their standard error and physical scan-boundary effects."
    lines.append(conclusion)

    pseudo_rows = [row for row in closure_rows if np.isfinite(row["pseudo_bias_z"])]
    if asimov_pass and pseudo_rows:
        lines.extend(["", "Only after the high-statistics Asimov pass, the existing pseudo-experiment mean biases are compared with their standard errors:", ""])
        for row in pseudo_rows:
            lines.append(f"- **{_label(row['policy'])}, C={row['C_true']:.2f}:** bias {row['pseudo_bias']:+.4f} +/- {row['pseudo_bias_error']:.4f} ({row['pseudo_bias_z']:+.2f} sigma).")
    lines.extend(["", "No DGPO, reward, KL/trust, flow, or score-model change is proposed or performed by this closure.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_statistical_closure(
    config: dict[str, Any],
    device: torch.device,
    policies: dict[str, ConditionalFlow],
    output: Path,
) -> None:
    names = [name for name in (
        "baseline", "fisher_dgpo_trust", "iterative_refresh_trust", "iterative_refresh_no_trust",
        "fisher_dgpo_no_trust",
    ) if name in policies]
    if "baseline" not in names or len(names) < 2:
        raise RuntimeError("Statistical closure requires the baseline and at least one optimized frozen policy")
    settings = config["closure"]
    total_events = int(settings["events"])
    if total_events <= 0 or int(settings["chunk_size"]) <= 0:
        raise ValueError("closure.events and closure.chunk_size must be positive")
    nominal_C = float(config["physics"]["nominal_C"])
    true_values = [float(value) for value in config["physics"]["true_C_values"]]
    edges = np.linspace(-1.0, 1.0, int(config["inference"]["reco_bins"]) + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    scan = _scan(config)
    counts, derivatives, second_moments, direct = _build_high_statistics_predictions(
        config, device, policies, names, edges,
    )
    reweighted_scan = {
        name: counts[name][None, :] + (scan[:, None] - nominal_C) * derivatives[name][None, :]
        for name in names
    }
    reweighted = {
        value: {name: counts[name] + (value - nominal_C) * derivatives[name] for name in names}
        for value in true_values
    }
    current_templates = _current_calibration_templates(
        config, device, policies, names, edges, scan, float(total_events),
    )

    score_rows: list[dict[str, Any]] = []
    nominal_direct = direct[min(true_values, key=lambda value: abs(value - nominal_C))]
    for name in names:
        score_sum, yield_derivative, full_score, fisher = poisson_score_components(
            nominal_direct[name], counts[name], derivatives[name],
        )
        _, _, asimov_full_score, _ = poisson_score_components(counts[name], counts[name], derivatives[name])
        score_rows.append({
            "policy": name,
            "events": total_events,
            "score_sum": score_sum,
            "yield_derivative": yield_derivative,
            "full_score": full_score,
            "full_score_per_event": full_score / total_events,
            "Fisher": fisher,
            "Z_bias": abs(full_score) / np.sqrt(fisher),
            "asimov_full_score": asimov_full_score,
        })

    summary = _read_summary(output)
    closure_rows: list[dict[str, Any]] = []
    likelihoods: dict[float, dict[str, np.ndarray]] = {value: {} for value in true_values}
    for value in true_values:
        delta_C = value - nominal_C
        for name in names:
            direct_values = direct[value][name]
            rw_values = reweighted[value][name]
            direct_yield = float(direct_values.sum())
            rw_yield = float(rw_values.sum())
            direct_shape = direct_values / direct_yield
            rw_shape = rw_values / rw_yield
            weighted_variance = counts[name] + 2.0 * delta_C * derivatives[name] + delta_C**2 * second_moments[name]
            residual_variance = np.clip(direct_values + weighted_variance, 1.0e-12, None)
            local_fisher = float(np.sum(derivatives[name] ** 2 / np.clip(rw_values, 1.0e-12, None)))
            asimov_MC_sigma = np.sqrt(2.0 / local_fisher)
            estimate, _, _, likelihood = fit_poisson(direct_values, reweighted_scan[name], scan)
            current_estimate, _, _, _ = fit_poisson(direct_values, current_templates[name], scan)
            likelihoods[value][name] = likelihood
            old = summary.get((name, value), {})
            old_error = float(old.get("mean_C_hat_error", np.nan))
            old_bias = float(old.get("bias", np.nan))
            closure_rows.append({
                "policy": name,
                "C_true": value,
                "direct_yield": direct_yield,
                "reweighted_yield": rw_yield,
                "yield_relative_difference": (direct_yield - rw_yield) / rw_yield,
                "direct_efficiency": direct_yield / total_events,
                "reweighted_efficiency": rw_yield / total_events,
                "shape_total_variation": 0.5 * float(np.sum(np.abs(direct_shape - rw_shape))),
                "maximum_bin_pull": float(np.max(np.abs(direct_values - rw_values) / np.sqrt(residual_variance))),
                "asimov_C_hat": estimate,
                "asimov_bias": estimate - value,
                "asimov_MC_sigma": asimov_MC_sigma,
                "asimov_MC_pull": (estimate - value) / asimov_MC_sigma,
                "current_calibration_C_hat": current_estimate,
                "current_calibration_bias": current_estimate - value,
                "pseudo_bias": old_bias,
                "pseudo_bias_error": old_error,
                "pseudo_bias_z": old_bias / old_error if old_error > 0.0 else np.nan,
            })

    with (output / "extended_score_closure.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    with (output / "template_closure_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(closure_rows[0]))
        writer.writeheader()
        writer.writerows(closure_rows)
    with (output / "statistical_closure.json").open("w", encoding="utf-8") as stream:
        json.dump({"extended_score": score_rows, "template_closure": closure_rows}, stream, indent=2)

    _plot_extended_score(score_rows, config, output)
    _plot_templates(names, true_values, centers, direct, reweighted, config, output)
    _plot_acceptance(closure_rows, names, config, output)
    _plot_asimov(closure_rows, names, config, output)
    _plot_likelihood_examples(names, scan, likelihoods, config, output)
    _update_report(output, score_rows, closure_rows, config)
    print(f"Statistical closure written to {output / 'diagnosis.md'}", flush=True)
