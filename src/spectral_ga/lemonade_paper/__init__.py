"""Paper-oriented LEMONADE components.

These utilities are kept separate from the existing CIFAR LEMONADE-style runner
so experiments can distinguish paper-faithful baselines from approximations.
"""

from .selection import (
    LemonadeCandidate,
    density_inverse_probabilities,
    dominates,
    gaussian_kde_density,
    pareto_front,
    sample_indices,
)
from .search_space_i import (
    PaperBlock,
    PaperCNNArch,
    PaperLemonadeCNN,
    arch_id,
    cheap_objectives,
    count_params,
    initial_arches,
    mutate_arch,
    snapshot_state,
    warmstart_model,
)

__all__ = [
    "LemonadeCandidate",
    "PaperBlock",
    "PaperCNNArch",
    "PaperLemonadeCNN",
    "arch_id",
    "cheap_objectives",
    "count_params",
    "density_inverse_probabilities",
    "dominates",
    "gaussian_kde_density",
    "initial_arches",
    "mutate_arch",
    "pareto_front",
    "sample_indices",
    "snapshot_state",
    "warmstart_model",
]
