from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .network import SpectralNetwork
from .spectral import SpectralLayer, decompose_dense, reconstruct_layer
from .innovation import CrossoverPlan, align_homologous_layers


@dataclass
class CrossoverConfig:
    method: str = "half_rank"
    uniform_prob: float = 0.5
    allow_shape_mismatch: bool = False


def _pad_layer_rank(layer: SpectralLayer, target_rank: int) -> SpectralLayer:
    """Return a copy of layer with extra zero-valued SVD modes appended."""
    child = layer.copy()
    if target_rank <= child.rank:
        return child
    extra = target_rank - child.rank
    child.u = np.column_stack([child.u, np.zeros((child.u.shape[0], extra), dtype=float)])
    child.alpha = np.concatenate([child.alpha, np.zeros(extra, dtype=float)])
    child.v = np.column_stack([child.v, np.zeros((child.v.shape[0], extra), dtype=float)])
    return child


def _copy_overlapping_mode_packages(
    anchor: SpectralLayer,
    donor: SpectralLayer,
    indices_to_replace: np.ndarray,
) -> SpectralLayer:
    """Copy donor SVD mode pieces into anchor, padding rank with zeros first.

    The child keeps the anchor layer shape but expands to the larger parent rank.
    Missing modes are represented as zero alpha/U/V columns, so padding itself is
    function-preserving. For different-dimension homologous layers, only the
    overlapping U/V coordinates are exchanged.
    """
    target_rank = max(anchor.rank, donor.rank)
    child = _pad_layer_rank(anchor, target_rank)
    common_u = min(anchor.u.shape[0], donor.u.shape[0])
    common_v = min(anchor.v.shape[0], donor.v.shape[0])
    for idx in indices_to_replace.tolist():
        if idx >= target_rank:
            continue
        if idx < donor.rank:
            child.alpha[idx] = donor.alpha[idx]
            child.u[:, idx] = 0.0
            child.v[:, idx] = 0.0
            child.u[:common_u, idx] = donor.u[:common_u, idx]
            child.v[:common_v, idx] = donor.v[:common_v, idx]
        else:
            child.alpha[idx] = 0.0
            child.u[:, idx] = 0.0
            child.v[:, idx] = 0.0
    return child


def _crossover_indices(method: str, rank: int, uniform_prob: float, rng: np.random.Generator) -> np.ndarray:
    if method == "uniform":
        mask = rng.random(rank) < uniform_prob
        return np.nonzero(mask)[0]
    if method == "single_point":
        cut = rng.integers(1, rank)
        return np.arange(cut, rank)
    if method in {"half_rank", "sbx"}:
        pivot = rank // 2
        return np.arange(pivot, rank)
    raise ValueError(f"Unsupported crossover method: {method}")


def anchor_layer_mode_crossover(
    parent_a: SpectralNetwork,
    parent_b: SpectralNetwork,
    layer_index: int,
    config: CrossoverConfig,
    rng: np.random.Generator,
) -> SpectralNetwork:
    child = parent_a.copy()
    if layer_index < 0 or layer_index >= len(child.layers):
        return child

    if layer_index >= len(parent_b.layers):
        return child

    anchor_layer = parent_a.layers[layer_index]
    donor_layer = parent_b.layers[layer_index]
    if anchor_layer.shape != donor_layer.shape and not config.allow_shape_mismatch:
        return child
    rank = max(anchor_layer.rank, donor_layer.rank)
    if rank <= 1:
        return child

    indices = _crossover_indices(config.method, rank, config.uniform_prob, rng)
    new_layer = _copy_overlapping_mode_packages(anchor_layer, donor_layer, indices)
    child.layers[layer_index] = new_layer
    return child


def plan_network_crossover(parent_a: SpectralNetwork, parent_b: SpectralNetwork) -> CrossoverPlan:
    """Return the innovation-aware crossover plan without changing either parent."""
    return align_homologous_layers(parent_a, parent_b)


def anchor_network_mode_crossover(
    parent_a: SpectralNetwork,
    parent_b: SpectralNetwork,
    config: CrossoverConfig,
    rng: np.random.Generator,
) -> SpectralNetwork:
    child = parent_a.copy()
    plan = plan_network_crossover(parent_a, parent_b)
    for pair in plan.matched_pairs:
        anchor_layer = child.layers[pair.index_a]
        donor_layer = parent_b.layers[pair.index_b]
        if anchor_layer.shape != donor_layer.shape and not config.allow_shape_mismatch:
            continue
        rank = max(anchor_layer.rank, donor_layer.rank)
        if rank <= 1:
            continue
        indices = _crossover_indices(config.method, rank, config.uniform_prob, rng)
        child.layers[pair.index_a] = _copy_overlapping_mode_packages(anchor_layer, donor_layer, indices)
    return child


def raw_weight_crossover(
    parent_a: SpectralNetwork,
    parent_b: SpectralNetwork,
    layer_index: int,
    rng: np.random.Generator,
) -> SpectralNetwork:
    child = parent_a.copy()
    if layer_index < 0 or layer_index >= len(child.layers):
        return child

    a_weight = reconstruct_layer(parent_a.layers[layer_index])
    b_weight = reconstruct_layer(parent_b.layers[layer_index])
    flat_a = a_weight.ravel()
    flat_b = b_weight.ravel()
    cut = rng.integers(1, flat_a.size)
    child_weight = np.concatenate([flat_a[:cut], flat_b[cut:]]).reshape(a_weight.shape)
    child_layer = decompose_dense(child_weight, parent_a.layers[layer_index].rank, keep_residual=True)
    child.layers[layer_index] = child_layer
    return child


