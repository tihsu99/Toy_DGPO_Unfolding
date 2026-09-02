from __future__ import annotations

import numpy as np


def response_matrix(truth: np.ndarray, reco: np.ndarray, truth_edges: np.ndarray, reco_edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _, _ = np.histogram2d(reco, truth, bins=(reco_edges, truth_edges))
    truth_counts = counts.sum(axis=0)
    if np.any(truth_counts == 0):
        raise ValueError("At least one truth bin has no nominal calibration events")
    return counts / truth_counts[None, :], truth_counts


def iterative_bayes(reco_counts: np.ndarray, response: np.ndarray, prior: np.ndarray, iterations: int) -> np.ndarray:
    current = np.asarray(prior, dtype=float).copy()
    current *= reco_counts.sum() / current.sum()
    efficiency = response.sum(axis=0)
    for _ in range(iterations):
        expected_reco = response @ current
        posterior = response * current[None, :] / np.clip(expected_reco[:, None], 1.0e-12, None)
        current = (posterior.T @ reco_counts) / np.clip(efficiency, 1.0e-12, None)
    return current


def _reweighted_templates(
    nominal_truth: np.ndarray,
    nominal_observable: np.ndarray,
    observable_edges: np.ndarray,
    nominal_C: float,
    scan: np.ndarray,
    target_total: float,
) -> np.ndarray:
    bin_index = np.searchsorted(observable_edges, nominal_observable, side="right") - 1
    bin_index = np.clip(bin_index, 0, observable_edges.size - 2)
    templates = []
    denominator = 1.0 + nominal_C * nominal_truth
    for C in scan:
        weights = (1.0 + C * nominal_truth) / denominator
        template = np.bincount(bin_index, weights=weights, minlength=observable_edges.size - 1).astype(float)
        templates.append(template * target_total / template.sum())
    return np.stack(templates)


def reweighted_truth_templates(
    nominal_truth: np.ndarray,
    truth_edges: np.ndarray,
    nominal_C: float,
    scan: np.ndarray,
    target_total: float,
) -> np.ndarray:
    return _reweighted_templates(nominal_truth, nominal_truth, truth_edges, nominal_C, scan, target_total)


def reweighted_reco_templates(
    nominal_truth: np.ndarray,
    nominal_reco: np.ndarray,
    reco_edges: np.ndarray,
    nominal_C: float,
    scan: np.ndarray,
    target_total: float,
) -> np.ndarray:
    return _reweighted_templates(nominal_truth, nominal_reco, reco_edges, nominal_C, scan, target_total)


def poisson_deviance(observed: np.ndarray, templates: np.ndarray) -> np.ndarray:
    expected = np.clip(templates, 1.0e-12, None)
    observed = np.asarray(observed, dtype=float)
    log_term = np.zeros_like(expected)
    positive = observed > 0.0
    log_term[:, positive] = observed[positive] * np.log(observed[positive] / expected[:, positive])
    return 2.0 * np.sum(expected - observed[None, :] + log_term, axis=1)


def _fit_scan(curve: np.ndarray, scan: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    best_index = int(np.argmin(curve))
    best = float(scan[best_index])
    minimum = float(curve[best_index])
    local_sigma: float | None = None
    if 0 < best_index < scan.size - 1:
        coefficients = np.polyfit(scan[best_index - 1 : best_index + 2], curve[best_index - 1 : best_index + 2], 2)
        if coefficients[0] > 0.0:
            local_sigma = float(1.0 / np.sqrt(coefficients[0]))
            candidate = float(-coefficients[1] / (2.0 * coefficients[0]))
            if scan[best_index - 1] <= candidate <= scan[best_index + 1]:
                best = candidate
                minimum = float(np.polyval(coefficients, candidate))
    delta = curve - minimum

    def crossing(indices: np.ndarray, fallback: float, lower_side: bool) -> float:
        if indices.size == 0:
            return fallback
        near = int(indices[-1] if lower_side else indices[0])
        other = near + 1 if lower_side else near - 1
        x0, y0 = float(scan[near]), float(delta[near])
        if delta[other] < 1.0:
            x1, y1 = float(scan[other]), float(delta[other])
        elif local_sigma is not None:
            return max(float(scan[0]), best - local_sigma) if lower_side else min(float(scan[-1]), best + local_sigma)
        else:
            x1, y1 = best, 0.0
        return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))

    lower = crossing(np.flatnonzero(delta[:best_index] >= 1.0), float(scan[0]), True)
    upper_indices = np.flatnonzero(delta[best_index + 1 :] >= 1.0)
    upper = crossing(upper_indices + best_index + 1, float(scan[-1]), False)
    return best, best - lower, upper - best, delta


def fit_poisson_parameter(
    observed: np.ndarray,
    templates: np.ndarray,
    scan: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    return _fit_scan(poisson_deviance(observed, templates), scan)
