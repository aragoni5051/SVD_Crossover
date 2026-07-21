from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class LemonadeCandidate:
    """Candidate architecture record for paper-style LEMONADE search.

    Objectives are minimized. `cheap_objectives` are available before expensive
    training/evaluation, while `objectives` contains the full objective vector
    for Pareto-front updates after the expensive objective has been measured.
    """

    architecture: Any
    cheap_objectives: tuple[float, ...]
    objectives: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        raise ValueError("Objective vectors must have the same length")
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def pareto_front(candidates: Sequence[LemonadeCandidate]) -> list[LemonadeCandidate]:
    """Return non-dominated candidates using their full objective vectors."""
    evaluated = [candidate for candidate in candidates if candidate.objectives is not None]
    front: list[LemonadeCandidate] = []
    for candidate in evaluated:
        assert candidate.objectives is not None
        if not any(
            other is not candidate
            and other.objectives is not None
            and dominates(other.objectives, candidate.objectives)
            for other in evaluated
        ):
            front.append(candidate)
    return front


def _objective_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2-D objective matrix")
    if matrix.shape[0] == 0:
        raise ValueError("Expected at least one objective vector")
    if not np.isfinite(matrix).all():
        raise ValueError("Objectives must be finite")
    return matrix


def _silverman_bandwidth(matrix: np.ndarray) -> np.ndarray:
    n_samples, n_dims = matrix.shape
    if n_samples <= 1:
        return np.ones(n_dims, dtype=float)
    std = np.std(matrix, axis=0, ddof=1)
    scale = (n_samples * (n_dims + 2.0) / 4.0) ** (-1.0 / (n_dims + 4.0))
    bandwidth = scale * std
    span = np.ptp(matrix, axis=0)
    floor = np.where(span > 0.0, span * 1e-6, 1.0)
    return np.maximum(bandwidth, floor)


def gaussian_kde_density(values: Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate a simple Gaussian KDE at each supplied objective vector."""
    matrix = _objective_matrix(values)
    bandwidth = _silverman_bandwidth(matrix)
    scaled = (matrix[:, None, :] - matrix[None, :, :]) / bandwidth
    exponent = -0.5 * np.sum(scaled * scaled, axis=2)
    kernel_values = np.exp(exponent)
    normalizer = np.prod(bandwidth) * ((2.0 * np.pi) ** (matrix.shape[1] / 2.0))
    densities = np.mean(kernel_values, axis=1) / normalizer
    return np.maximum(densities, np.finfo(float).tiny)


def density_inverse_probabilities(values: Sequence[Sequence[float]]) -> np.ndarray:
    """Selection probabilities favoring sparse regions of cheap-objective space."""
    density = gaussian_kde_density(values)
    weights = 1.0 / density
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(weights), 1.0 / len(weights), dtype=float)
    return weights / total


def sample_indices(
    values: Sequence[Sequence[float]],
    count: int,
    rng: np.random.Generator,
    replace: bool = True,
) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    probabilities = density_inverse_probabilities(values)
    if not replace and count > len(probabilities):
        raise ValueError("Cannot sample more items than available without replacement")
    return [int(index) for index in rng.choice(len(probabilities), size=count, replace=replace, p=probabilities)]
