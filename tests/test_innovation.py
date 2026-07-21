import json

import numpy as np

from spectral_ga.crossover import plan_network_crossover
from spectral_ga.innovation import InnovationRegistry, align_homologous_layers
from spectral_ga.network import NetworkConfig, SpectralNetwork


def ids(network: SpectralNetwork) -> list[int]:
    return [layer.gene.innovation_id for layer in network.layers]


def test_same_topology_aligns_all_layers_by_innovation_id() -> None:
    registry = InnovationRegistry()
    parent_a = SpectralNetwork.from_architecture(NetworkConfig([2, 3, 1], seed=1, innovation_registry=registry))
    parent_b = parent_a.copy()

    plan = align_homologous_layers(parent_a, parent_b)

    assert [pair.innovation_id for pair in plan.matched_pairs] == ids(parent_a)
    assert plan.unmatched_from_a == []
    assert plan.unmatched_from_b == []
    assert plan.inherited_identity_scaffolds == []
    assert plan.incompatible_pairs == []


def test_identity_insertion_at_first_layer_does_not_shift_alignment() -> None:
    registry = InnovationRegistry()
    parent_a = SpectralNetwork.from_architecture(NetworkConfig([64, 10], seed=2, innovation_registry=registry))
    parent_b = parent_a.copy()
    original_ids = ids(parent_a)

    assert parent_b.add_identity_convolution_layer((8, 8), layer_index=0)
    plan = align_homologous_layers(parent_a, parent_b)

    assert [pair.innovation_id for pair in plan.matched_pairs] == original_ids
    assert [(pair.index_a, pair.index_b) for pair in plan.matched_pairs] == [(0, 1)]
    assert len(plan.inherited_identity_scaffolds) == 1
    assert plan.inherited_identity_scaffolds[0][2].is_identity_scaffold


def test_identity_insertion_in_middle_keeps_downstream_ids() -> None:
    registry = InnovationRegistry()
    parent_a = SpectralNetwork.from_architecture(NetworkConfig([4, 4, 2], seed=3, innovation_registry=registry))
    parent_b = parent_a.copy()
    original_ids = ids(parent_a)

    assert parent_b.add_identity_convolution_layer((2, 2), layer_index=1)
    plan = align_homologous_layers(parent_a, parent_b)

    assert ids(parent_a) == original_ids
    assert ids(parent_b)[0] == original_ids[0]
    assert ids(parent_b)[2] == original_ids[1]
    assert [(pair.index_a, pair.index_b) for pair in plan.matched_pairs] == [(0, 0), (1, 2)]
    assert len(plan.inherited_identity_scaffolds) == 1


def test_same_structural_insertions_reuse_innovation_id() -> None:
    registry = InnovationRegistry()
    parent = SpectralNetwork.from_architecture(NetworkConfig([64, 10], seed=4, innovation_registry=registry))
    a = parent.copy()
    b = parent.copy()

    assert a.add_identity_convolution_layer((8, 8), layer_index=0)
    after_first_next = registry.next_id
    assert b.add_identity_convolution_layer((8, 8), layer_index=0)

    inserted_a = ids(a)[0]
    inserted_b = ids(b)[0]
    assert inserted_a == inserted_b
    assert registry.next_id == after_first_next
    assert len(set(ids(a) + ids(b))) == 2


def test_different_structural_insertions_get_different_innovation_ids() -> None:
    registry = InnovationRegistry()
    parent = SpectralNetwork.from_architecture(NetworkConfig([4, 4, 2], seed=44, innovation_registry=registry))
    a = parent.copy()
    b = parent.copy()

    assert a.add_identity_convolution_layer((2, 2), layer_index=0)
    assert b.add_identity_convolution_layer((2, 2), layer_index=1)

    assert ids(a)[0] != ids(b)[1]


def test_clone_preserves_innovation_ids() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([2, 3, 1], seed=5))
    clone = network.copy()

    assert ids(clone) == ids(network)
    assert clone.innovation_registry.next_id == network.innovation_registry.next_id


def test_signed_relu_insertion_assigns_two_new_ids_and_records_parent() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([2, 1], seed=6))
    old_id = ids(network)[0]
    before_next = network.innovation_registry.next_id

    assert network.add_signed_relu_layer(layer_index=0)

    assert network.innovation_registry.next_id == before_next + 2
    assert old_id not in ids(network)
    assert ids(network)[0] != ids(network)[1]
    assert network.layers[0].gene.parent_innovation_id == old_id
    assert network.layers[1].gene.parent_innovation_id == old_id
    assert network.layers[0].gene.is_identity_scaffold
    assert network.layers[1].gene.is_identity_scaffold


def test_index_based_matching_would_fail_but_innovation_matching_succeeds() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig([64, 10], seed=7))
    parent_b = parent_a.copy()
    original_l1 = ids(parent_a)[0]

    assert parent_b.add_identity_convolution_layer((8, 8), layer_index=0)
    naive_index_pair = (ids(parent_a)[0], ids(parent_b)[0])
    plan = plan_network_crossover(parent_a, parent_b)

    assert naive_index_pair[0] != naive_index_pair[1]
    assert [pair.innovation_id for pair in plan.matched_pairs] == [original_l1]


def test_save_load_preserves_innovation_ids_and_next_counter() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([64, 10], seed=8))
    assert network.add_identity_convolution_layer((8, 8), layer_index=0)
    payload = json.loads(json.dumps(network.to_dict()))

    loaded = SpectralNetwork.from_dict(payload)

    assert ids(loaded) == ids(network)
    assert loaded.innovation_registry.next_id == network.innovation_registry.next_id
    assert loaded.innovation_registry.signature_to_id == network.innovation_registry.signature_to_id



def test_node_split_same_innovation_different_shape_still_matches_for_overlap_crossover() -> None:
    parent_a = SpectralNetwork.from_architecture(NetworkConfig([2, 3, 1], seed=9))
    parent_b = parent_a.copy()
    original_first_id = ids(parent_a)[0]

    assert parent_b.split_hidden_node(layer_index=0, node_index=1)
    plan = align_homologous_layers(parent_a, parent_b)

    assert original_first_id in [pair.innovation_id for pair in plan.matched_pairs]
    assert plan.incompatible_pairs == []
    first_pair = next(pair for pair in plan.matched_pairs if pair.innovation_id == original_first_id)
    assert parent_a.layers[first_pair.index_a].shape != parent_b.layers[first_pair.index_b].shape


def test_layer_deletion_merge_creates_new_innovation_id() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([2, 3, 2, 1], seed=10))
    old_ids = ids(network)
    before_next = network.innovation_registry.next_id

    assert network.delete_hidden_layer(layer_index=0)

    assert network.innovation_registry.next_id == before_next + 1
    assert ids(network)[0] not in old_ids
    assert network.layers[0].gene.operation_type == "linearized_layer_delete_merge"
    assert network.layers[0].gene.morphism["deleted_innovation_id"] == old_ids[0]
    assert network.layers[0].gene.morphism["downstream_innovation_id"] == old_ids[1]



def test_intermediate_dense_layer_addition_uses_compact_width_and_preserves_output_id() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=11))
    old_output_id = ids(network)[0]

    assert network.add_intermediate_dense_layer(layer_index=0, hidden_width=64)

    assert [layer.shape for layer in network.layers] == [(64, 1024), (1, 64)]
    assert ids(network)[0] != old_output_id
    assert ids(network)[1] == old_output_id
    assert network.layers[0].gene.operation_type == "intermediate_linear"



def test_signed_relu_image_growth_sequence_preserves_function_and_replaces_split_layer_ids() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=12))
    first_output_id = ids(network)[0]

    assert network.add_signed_relu_layer(layer_index=0)
    assert [layer.shape for layer in network.layers] == [(2, 1024), (1, 2)]
    first_pair_ids = ids(network)
    assert first_output_id not in first_pair_ids
    assert network.layers[0].gene.parent_innovation_id == first_output_id
    assert network.layers[1].gene.parent_innovation_id == first_output_id

    assert network.add_signed_relu_layer(layer_index=0)
    assert [layer.shape for layer in network.layers] == [(4, 1024), (2, 4), (1, 2)]
    assert first_pair_ids[0] not in ids(network)[:2]
    assert ids(network)[2] == first_pair_ids[1]
    assert network.layers[0].gene.parent_innovation_id == first_pair_ids[0]
    assert network.layers[1].gene.parent_innovation_id == first_pair_ids[0]



def test_signed_relu_uses_input_sign_split_when_input_is_smaller() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([2, 8], seed=13))
    x = np.random.default_rng(13).standard_normal((6, 2))
    before = network.forward(x)
    old_id = ids(network)[0]

    assert network.add_signed_relu_layer(layer_index=0)

    after = network.forward(x)
    assert [layer.shape for layer in network.layers] == [(4, 2), (8, 4)]
    assert old_id not in ids(network)
    assert network.layers[0].gene.morphism["split_kind"] == "input_sign"
    assert network.layers[1].gene.morphism["split_kind"] == "input_sign"
    assert np.max(np.abs(before - after)) < 1e-10



def test_signed_relu_mean_width_policy_preserves_function_with_mid_width() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=14))
    x = np.random.default_rng(14).standard_normal((4, 1024))
    before = network.forward(x)

    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean")

    after = network.forward(x)
    assert [layer.shape for layer in network.layers] == [(512, 1024), (1, 512)]
    assert network.layers[0].gene.morphism["width_policy"] == "mean"
    assert np.max(np.abs(before - after)) < 1e-10


def test_signed_relu_mean_width_policy_uses_average_when_larger_than_minimal() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([2, 8], seed=15))
    x = np.random.default_rng(15).standard_normal((5, 2))
    before = network.forward(x)

    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean")

    after = network.forward(x)
    assert [layer.shape for layer in network.layers] == [(5, 2), (8, 5)]
    assert np.max(np.abs(before - after)) < 1e-10



def test_signed_relu_optional_cap_blocks_when_exact_split_cannot_fit() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=16))
    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean", max_hidden_width=128)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (1, 128)]

    assert not network.add_signed_relu_layer(layer_index=0, width_policy="mean", max_hidden_width=128)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (1, 128)]

    assert network.add_signed_relu_layer(layer_index=1, width_policy="mean", max_hidden_width=128)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (64, 128), (1, 64)]



def test_signed_relu_mean_width_policy_respects_max_hidden_width_cap() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=17))
    x = np.random.default_rng(17).standard_normal((3, 1024))
    before = network.forward(x)

    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean", max_hidden_width=128)

    after = network.forward(x)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (1, 128)]
    assert network.layers[0].gene.morphism["requested_width"] == 512
    assert network.layers[0].gene.morphism["max_hidden_width"] == 128
    assert np.max(np.abs(before - after)) < 1e-10


def test_signed_relu_cap_blocks_when_exact_split_cannot_fit() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=18))
    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean", max_hidden_width=128)
    assert not network.add_signed_relu_layer(layer_index=0, width_policy="mean", max_hidden_width=128)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (1, 128)]
    assert network.add_signed_relu_layer(layer_index=1, width_policy="mean", max_hidden_width=128)
    assert [layer.shape for layer in network.layers] == [(128, 1024), (64, 128), (1, 64)]



def test_best_signed_relu_expansion_layer_chooses_cheapest_target() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=19))
    assert network.best_signed_relu_expansion_layer(width_policy="mean") == 0
    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean")
    assert [layer.shape for layer in network.layers] == [(512, 1024), (1, 512)]

    assert network.best_signed_relu_expansion_layer(width_policy="mean") == 1
    assert network.add_signed_relu_layer(layer_index=1, width_policy="mean")
    assert [layer.shape for layer in network.layers] == [(512, 1024), (256, 512), (1, 256)]



def test_signed_relu_mean_policy_skips_expansion_when_exact_split_exceeds_mean_width() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([1024, 1], seed=20))
    assert network.add_signed_relu_layer(layer_index=0, width_policy="mean")
    assert [layer.shape for layer in network.layers] == [(512, 1024), (1, 512)]

    assert not network.add_signed_relu_layer(layer_index=0, width_policy="mean")
    assert [layer.shape for layer in network.layers] == [(512, 1024), (1, 512)]

    assert network.add_signed_relu_layer(layer_index=1, width_policy="mean")
    assert [layer.shape for layer in network.layers] == [(512, 1024), (256, 512), (1, 256)]


def test_signed_relu_mean_policy_blocks_exact_split_wider_than_mean_width() -> None:
    network = SpectralNetwork.from_architecture(NetworkConfig([64, 37], seed=21))
    before_shapes = [layer.shape for layer in network.layers]

    assert network.signed_relu_expansion_hidden_width(0, width_policy="mean") is None
    assert network.signed_relu_expansion_parameter_cost(0, width_policy="mean") is None
    assert network.best_signed_relu_expansion_layer(width_policy="mean") is None
    assert not network.add_signed_relu_layer(layer_index=0, width_policy="mean")
    assert [layer.shape for layer in network.layers] == before_shapes




