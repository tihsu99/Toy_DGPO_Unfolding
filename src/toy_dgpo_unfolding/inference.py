from __future__ import annotations

import numpy as np


def reweighted_templates(
    nominal_x: np.ndarray,
    nominal_y: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    nominal_C: float,
    scan: np.ndarray,
    target_exposure: float,
) -> np.ndarray:
    x = nominal_x[valid]
    y = nominal_y[valid]
    bin_index = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, edges.size - 2)
    denominator = 1.0 + nominal_C * x
    templates = []
    for C in scan:
        weights = (1.0 + C * x) / denominator
        counts = np.bincount(bin_index, weights=weights, minlength=edges.size - 1).astype(float)
        templates.append(counts * target_exposure / nominal_x.size)
    return np.stack(templates)


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
            return x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
        if local_sigma is not None:
            return max(float(scan[0]), best - local_sigma) if lower_side else min(float(scan[-1]), best + local_sigma)
        return best

    lower = crossing(np.flatnonzero(delta[:best_index] >= 1.0), float(scan[0]), True)
    upper_indices = np.flatnonzero(delta[best_index + 1 :] >= 1.0) + best_index + 1
    upper = crossing(upper_indices, float(scan[-1]), False)
    return best, best - lower, upper - best, delta


def fit_poisson(observed: np.ndarray, templates: np.ndarray, scan: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    return _fit_scan(poisson_deviance(observed, templates), scan)


def binned_fisher_per_event(
    nominal_x: np.ndarray,
    nominal_y: np.ndarray,
    valid: np.ndarray,
    edges: np.ndarray,
    nominal_C: float,
) -> float:
    x = nominal_x[valid]
    y = nominal_y[valid]
    bins = np.clip(np.searchsorted(edges, y, side="right") - 1, 0, edges.size - 2)
    counts = np.bincount(bins, minlength=edges.size - 1).astype(float)
    derivatives = np.bincount(bins, weights=x / (1.0 + nominal_C * x), minlength=edges.size - 1).astype(float)
    probability = counts / nominal_x.size
    probability_derivative = derivatives / nominal_x.size
    return float(np.sum(probability_derivative**2 / np.clip(probability, 1.0e-12, None)))
