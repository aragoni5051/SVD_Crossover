import numpy as np

from spectral_ga.network import NetworkConfig, SpectralNetwork
from spectral_ga.spectral import decompose_dense, reconstruct_layer




def test_split_hidden_node_random_copy_preserves_function() -> None:
    config = NetworkConfig(layer_dims=[3, 4, 2], r_max=4, seed=13)
    network = SpectralNetwork.from_architecture(config)
    samples = np.random.default_rng(31).standard_normal((8, 3))
    before = network.forward(samples)

    assert network.split_hidden_node(layer_index=0, rng=np.random.default_rng(17))

    assert [layer.shape for layer in network.layers] == [(5, 3), (2, 5)]
    np.testing.assert_allclose(network.forward(samples), before, atol=1e-10)




def test_delete_hidden_node_reduces_width_and_keeps_finite_outputs() -> None:
    config = NetworkConfig(layer_dims=[2, 4, 3, 1], r_max=4, seed=3)
    network = SpectralNetwork.from_architecture(config)
    assert network.delete_hidden_node(layer_index=0, node_index=1)
    assert [layer.shape for layer in network.layers] == [(3, 2), (3, 3), (1, 3)]
    outputs = network.forward(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    assert np.isfinite(outputs).all()


def test_delete_hidden_node_merges_into_closest_correlated_node() -> None:
    config = NetworkConfig(layer_dims=[2, 3, 1], r_max=3, seed=7)
    network = SpectralNetwork.from_architecture(config)
    incoming_gene = network.layers[0].gene
    outgoing_gene = network.layers[1].gene

    incoming = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    outgoing = np.array([[10.0, 5.0, 7.0]], dtype=float)
    network.layers[0] = decompose_dense(incoming, min(incoming.shape), keep_residual=False)
    network.layers[0].gene = incoming_gene
    network.layers[1] = decompose_dense(outgoing, min(outgoing.shape), keep_residual=False)
    network.layers[1].gene = outgoing_gene
    network.biases[0] = np.zeros(3, dtype=float)
    network.biases[1] = np.zeros(1, dtype=float)

    assert network.delete_hidden_node(layer_index=0, node_index=1)

    assert [layer.shape for layer in network.layers] == [(2, 2), (1, 2)]
    np.testing.assert_allclose(
        reconstruct_layer(network.layers[1]),
        np.array([[15.0, 7.0]], dtype=float),
        atol=1e-10,
    )


def test_delete_hidden_layer_ridge_fits_direct_replacement_on_samples() -> None:
    config = NetworkConfig(layer_dims=[2, 2, 1], r_max=2, seed=11)
    network = SpectralNetwork.from_architecture(config)
    first_gene = network.layers[0].gene
    second_gene = network.layers[1].gene

    first_weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    second_weight = np.array([[1.0, -2.0]], dtype=float)
    network.layers[0] = decompose_dense(first_weight, min(first_weight.shape), keep_residual=False)
    network.layers[0].gene = first_gene
    network.layers[1] = decompose_dense(second_weight, min(second_weight.shape), keep_residual=False)
    network.layers[1].gene = second_gene
    network.biases[0] = np.zeros(2, dtype=float)
    network.biases[1] = np.array([0.3], dtype=float)

    samples = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=float)
    expected_outputs = network.forward(samples)

    assert network.delete_hidden_layer_ridge(layer_index=0, samples=samples, ridge_lambda=0.0)

    assert [layer.shape for layer in network.layers] == [(1, 2)]
    assert network.layers[0].gene.operation_type == "ridge_layer_delete_merge"
    np.testing.assert_allclose(network.forward(samples), expected_outputs, atol=1e-10)

def test_delete_hidden_layer_reduces_depth_and_keeps_finite_outputs() -> None:
    config = NetworkConfig(layer_dims=[2, 4, 3, 1], r_max=4, seed=5)
    network = SpectralNetwork.from_architecture(config)
    assert network.delete_hidden_layer(layer_index=0)
    assert [layer.shape for layer in network.layers] == [(3, 2), (1, 3)]
    outputs = network.forward(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float))
    assert np.isfinite(outputs).all()




