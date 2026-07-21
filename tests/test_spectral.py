import numpy as np

from spectral_ga.network import NetworkConfig, SpectralNetwork
from spectral_ga.spectral import decompose_dense, reconstruct_layer, optimize_spectral_parameters


def test_full_svd_reconstruction() -> None:
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((5, 4), dtype=float)
    layer = decompose_dense(weight, r_max=4, keep_residual=False)
    reconstructed = reconstruct_layer(layer)
    assert reconstructed.shape == weight.shape
    assert np.allclose(reconstructed, weight, atol=1e-6)


def test_truncated_svd_with_residual() -> None:
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((6, 5), dtype=float)
    layer = decompose_dense(weight, r_max=2, keep_residual=True)
    reconstructed = reconstruct_layer(layer)
    assert reconstructed.shape == weight.shape
    assert np.allclose(reconstructed, weight, atol=1e-6)


def test_mode_sum_dimensions() -> None:
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((4, 3), dtype=float)
    layer = decompose_dense(weight, r_max=2, keep_residual=False)
    assert layer.u.shape == (4, 2)
    assert layer.v.shape == (3, 2)
    assert layer.alpha.shape == (2,)
    reconstructed = reconstruct_layer(layer)
    assert reconstructed.shape == (4, 3)


def test_optimize_spectral_parameters_keeps_uv_by_default() -> None:
    config = NetworkConfig(layer_dims=[2, 4, 1], r_max=2, seed=5)
    network = SpectralNetwork.from_architecture(config)
    original_uv = [(layer.u.copy(), layer.v.copy()) for layer in network.layers]
    inputs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    targets = np.array([[0.0], [1.0]], dtype=float)
    optimize_spectral_parameters(network, inputs, targets, loss="mse", steps=5, lr=0.01, optimize_bias=False, seed=5)
    for idx, layer in enumerate(network.layers):
        u_orig, v_orig = original_uv[idx]
        assert np.allclose(layer.u, u_orig)
        assert np.allclose(layer.v, v_orig)


def test_optimize_spectral_parameters_updates_uv_when_requested() -> None:
    config = NetworkConfig(layer_dims=[2, 4, 1], r_max=2, seed=5)
    network = SpectralNetwork.from_architecture(config)
    original_uv = [(layer.u.copy(), layer.v.copy()) for layer in network.layers]
    inputs = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    targets = np.array([[0.0], [1.0]], dtype=float)
    optimize_spectral_parameters(network, inputs, targets, loss="mse", steps=5, lr=0.01, optimize_bias=False, optimize_uv=True, seed=5)
    changed = False
    for idx, layer in enumerate(network.layers):
        u_orig, v_orig = original_uv[idx]
        if not np.allclose(layer.u, u_orig) or not np.allclose(layer.v, v_orig):
            changed = True
    assert changed
