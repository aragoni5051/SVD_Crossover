import numpy as np

from spectral_ga.crossover import anchor_layer_mode_crossover, anchor_network_mode_crossover, raw_weight_crossover, CrossoverConfig
from spectral_ga.network import NetworkConfig, SpectralNetwork
from spectral_ga.spectral import reconstruct_layer
from spectral_ga.utils import seed_all


def test_anchor_layer_mode_crossover_preserves_dimensions() -> None:
    seed_all(10)
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=10))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=11))
    rng = np.random.default_rng(12)
    child = anchor_layer_mode_crossover(parent_a, parent_b, layer_index=0, config=CrossoverConfig(method="single_point"), rng=rng)
    assert len(child.layers) == len(parent_a.layers)
    assert child.layers[0].shape == parent_a.layers[0].shape
    assert child.layers[0].u.shape == parent_a.layers[0].u.shape


def test_raw_weight_crossover_preserves_shapes() -> None:
    seed_all(20)
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=20))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=21))
    rng = np.random.default_rng(22)
    child = raw_weight_crossover(parent_a, parent_b, layer_index=0, rng=rng)
    assert len(child.layers) == len(parent_a.layers)
    assert child.layers[0].shape == parent_a.layers[0].shape
    assert child.layers[0].u.shape == parent_a.layers[0].u.shape


def test_sbx_alias_matches_half_rank_crossover() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=30))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 8, 1], r_max=3, seed=31))
    rng = np.random.default_rng(32)

    half_rank_child = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=CrossoverConfig(method="half_rank"),
        rng=rng,
    )
    sbx_child = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=CrossoverConfig(method="sbx"),
        rng=rng,
    )

    assert np.allclose(half_rank_child.layers[0].u, sbx_child.layers[0].u)
    assert np.allclose(half_rank_child.layers[0].v, sbx_child.layers[0].v)
    assert np.allclose(half_rank_child.layers[0].alpha, sbx_child.layers[0].alpha)

def test_network_crossover_skips_same_innovation_different_shape_layers() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 3, 1], r_max=2, seed=40))
    parent_b = parent_a.copy()
    assert parent_b.split_hidden_node(layer_index=0, node_index=1)

    rng = np.random.default_rng(41)
    child = anchor_network_mode_crossover(parent_a, parent_b, CrossoverConfig(method="half_rank"), rng)

    assert child.layers[0].shape == parent_a.layers[0].shape
    assert np.allclose(child.layers[0].u, parent_a.layers[0].u)
    assert np.allclose(child.layers[0].v, parent_a.layers[0].v)
    assert np.allclose(child.layers[0].alpha, parent_a.layers[0].alpha)

def test_network_crossover_can_allow_same_innovation_different_shape_layers() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 3, 1], r_max=2, seed=42))
    parent_b = parent_a.copy()
    assert parent_b.split_hidden_node(layer_index=0, node_index=1)

    rng = np.random.default_rng(43)
    child = anchor_network_mode_crossover(
        parent_a,
        parent_b,
        CrossoverConfig(method="half_rank", allow_shape_mismatch=True),
        rng,
    )

    assert child.layers[0].shape == parent_a.layers[0].shape
    assert not np.allclose(child.layers[0].u, parent_a.layers[0].u)



def test_layer_crossover_pads_lower_rank_anchor_to_donor_rank() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 4], r_max=1, seed=50))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 4], r_max=3, seed=51))

    rng = np.random.default_rng(52)
    child = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=CrossoverConfig(method="half_rank"),
        rng=rng,
    )

    assert child.layers[0].rank == 3
    assert child.layers[0].u.shape == (4, 3)
    assert child.layers[0].v.shape == (4, 3)
    assert np.allclose(child.layers[0].alpha[0], parent_a.layers[0].alpha[0])
    assert np.allclose(child.layers[0].alpha[1:], parent_b.layers[0].alpha[1:])


def test_rank_padding_is_function_preserving_for_uncopied_modes() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 4], r_max=1, seed=53))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 4], r_max=3, seed=54))

    before = reconstruct_layer(parent_a.layers[0])
    rng = np.random.default_rng(55)
    child = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=CrossoverConfig(method="uniform", uniform_prob=0.0),
        rng=rng,
    )

    assert child.layers[0].rank == 3
    assert np.allclose(reconstruct_layer(child.layers[0]), before)


def test_different_shape_crossover_pads_rank_and_copies_overlapping_coordinates() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 4], r_max=1, seed=56))
    parent_b = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[4, 5], r_max=3, seed=57))

    rng = np.random.default_rng(58)
    child = anchor_layer_mode_crossover(
        parent_a,
        parent_b,
        layer_index=0,
        config=CrossoverConfig(method="half_rank", allow_shape_mismatch=True),
        rng=rng,
    )

    assert child.layers[0].shape == parent_a.layers[0].shape
    assert child.layers[0].rank == 3
    assert child.layers[0].u.shape == (4, 3)
    assert child.layers[0].v.shape == (4, 3)
    assert np.allclose(child.layers[0].alpha[1:], parent_b.layers[0].alpha[1:])
    assert np.allclose(child.layers[0].u[:, 1:], parent_b.layers[0].u[:4, 1:])
    assert np.allclose(child.layers[0].v[:, 1:], parent_b.layers[0].v[:, 1:])
