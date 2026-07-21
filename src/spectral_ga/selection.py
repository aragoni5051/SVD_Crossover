from __future__ import annotations

from typing import Sequence

import numpy as np


def tournament_selection(population: Sequence, fitnesses: Sequence[float], tournament_size: int = 2, rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng()
    selected = []
    for _ in range(len(population)):
        indices = rng.choice(len(population), size=tournament_size, replace=False)
        winner = min(indices, key=lambda i: fitnesses[i])
        selected.append(population[winner])
    return selected
