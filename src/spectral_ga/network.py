from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .spectral import SpectralLayer, decompose_dense, reconstruct_layer
from .innovation import InnovationRegistry, LayerGene, default_layer_gene, ensure_network_innovations
from .utils import balanced_bce_loss, bce_loss, mse_loss, ce_loss, seed_all


@dataclass
class NetworkConfig:
    layer_dims: Sequence[int]
    r_max: int = 10
    seed: int = 0
    init_scale: float | None = None
    rank_fraction: float | None = None
    rank_fraction_full_below: int = 5
    innovation_registry: InnovationRegistry | None = None

    def layer_rank(self, input_dim: int, output_dim: int) -> int:
        full_rank = min(input_dim, output_dim)
        if self.rank_fraction is None:
            return min(self.r_max, full_rank)
        if self.rank_fraction <= 0.0:
            raise ValueError("rank_fraction must be positive")
        if full_rank <= self.rank_fraction_full_below:
            return full_rank
        return max(1, min(full_rank, int(np.ceil(self.rank_fraction * full_rank))))


class SpectralNetwork:
    def __init__(
        self,
        layers: List[SpectralLayer],
        biases: List[np.ndarray],
        innovation_registry: InnovationRegistry | None = None,
    ) -> None:
        self.layers = layers
        self.biases = biases
        self.innovation_registry = ensure_network_innovations(self, innovation_registry)

    @classmethod
    def from_architecture(cls, config: NetworkConfig) -> "SpectralNetwork":
        rng = seed_all(config.seed)
        layers: list[SpectralLayer] = []
        biases: list[np.ndarray] = []
        for idx in range(len(config.layer_dims) - 1):
            input_dim = config.layer_dims[idx]
            output_dim = config.layer_dims[idx + 1]
            init_scale = config.init_scale
            if init_scale is None:
                init_scale = np.sqrt(2.0 / input_dim)
            weight = rng.standard_normal((output_dim, input_dim), dtype=float) * init_scale
            bias = np.zeros((output_dim,), dtype=float)
            spectral_layer = decompose_dense(weight, config.layer_rank(input_dim, output_dim), keep_residual=False)
            layers.append(spectral_layer)
            biases.append(bias)
        registry = config.innovation_registry or InnovationRegistry()
        network = cls(layers=layers, biases=biases, innovation_registry=registry)
        for layer in network.layers:
            if layer.gene.operation_type == "linear":
                layer.gene.operation_type = "linear"
        return network

    def copy(self) -> "SpectralNetwork":
        copied_layers = [layer.copy() for layer in self.layers]
        copied_biases = [bias.copy() for bias in self.biases]
        return SpectralNetwork(
            copied_layers,
            copied_biases,
            innovation_registry=self.innovation_registry,
        )


    def to_dict(self) -> dict:
        return {
            "layers": [layer.to_dict() for layer in self.layers],
            "biases": [bias.tolist() for bias in self.biases],
            "innovation_registry": self.innovation_registry.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpectralNetwork":
        layers = [SpectralLayer.from_dict(layer_data) for layer_data in data["layers"]]
        biases = [np.asarray(bias, dtype=float) for bias in data["biases"]]
        registry = InnovationRegistry.from_dict(data.get("innovation_registry", {}))
        return cls(layers=layers, biases=biases, innovation_registry=registry)

    def ensure_innovations(self) -> InnovationRegistry:
        self.innovation_registry = ensure_network_innovations(self, self.innovation_registry)
        return self.innovation_registry

    @property
    def hidden_layer_count(self) -> int:
        return max(0, len(self.layers) - 1)


    def signed_relu_expansion_hidden_width(
        self,
        layer_index: int,
        width_policy: str = "minimal",
        max_hidden_width: int | None = None,
    ) -> int | None:
        """Return hidden width for a signed-ReLU expansion, or None if invalid."""
        if layer_index < 0 or layer_index >= len(self.layers):
            return None
        output_dim, input_dim = self.layers[layer_index].shape
        minimal_width = 2 * min(input_dim, output_dim)
        if width_policy == "minimal":
            requested_width = minimal_width
        elif width_policy in {"mean", "mid", "average"}:
            mean_width = 0.5 * (input_dim + output_dim)
            if minimal_width > mean_width:
                return None
            requested_width = max(minimal_width, (input_dim + output_dim) // 2)
        else:
            raise ValueError(f"Unsupported signed-ReLU width_policy: {width_policy}")
        if max_hidden_width is not None:
            if minimal_width > max_hidden_width:
                return None
            return min(requested_width, max_hidden_width)
        return requested_width

    def signed_relu_expansion_parameter_cost(
        self,
        layer_index: int,
        width_policy: str = "minimal",
        max_hidden_width: int | None = None,
    ) -> int | None:
        """Parameter count of the two replacement layers for expansion ranking."""
        hidden_width = self.signed_relu_expansion_hidden_width(layer_index, width_policy, max_hidden_width)
        if hidden_width is None:
            return None
        output_dim, input_dim = self.layers[layer_index].shape
        return hidden_width * input_dim + output_dim * hidden_width + hidden_width + output_dim

    def best_signed_relu_expansion_layer(
        self,
        width_policy: str = "minimal",
        max_hidden_width: int | None = None,
    ) -> int | None:
        """Choose the cheapest valid signed-ReLU layer expansion target."""
        candidates: list[tuple[int, int]] = []
        for layer_index in range(len(self.layers)):
            cost = self.signed_relu_expansion_parameter_cost(layer_index, width_policy, max_hidden_width)
            if cost is not None:
                candidates.append((cost, layer_index))
        if not candidates:
            return None
        return min(candidates)[1]


    def add_signed_relu_layer(
        self,
        layer_index: int | None = None,
        width_policy: str = "minimal",
        max_hidden_width: int | None = None,
    ) -> bool:
        """Insert a function-preserving signed-ReLU hidden layer.

        A dense layer z = Wx + b is replaced by two layers:
          h = ReLU([z, -z])
          z = [I, -I] h
        For hidden layers, the usual ReLU after the reconstruction layer keeps the
        original activation ReLU(z). For output layers, logits are preserved.
        """
        if not self.layers:
            return False
        if layer_index is None:
            layer_index = len(self.layers) - 1
        if layer_index < 0 or layer_index >= len(self.layers):
            return False

        old_layer = self.layers[layer_index]
        old_gene = old_layer.gene.copy() if old_layer.gene is not None else default_layer_gene(old_layer.shape[0], old_layer.shape[1])
        old_weight = reconstruct_layer(old_layer)
        old_bias = self.biases[layer_index]
        output_dim, input_dim = old_weight.shape

        if input_dim < output_dim:
            split_kind = "input_sign"
            minimal_width = 2 * input_dim
            first_core = np.vstack([np.eye(input_dim), -np.eye(input_dim)])
            first_bias_core = np.zeros(minimal_width, dtype=float)
            second_core = np.column_stack([old_weight, -old_weight])
            second_bias = old_bias.copy()
        else:
            split_kind = "output_sign"
            minimal_width = 2 * output_dim
            first_core = np.vstack([old_weight, -old_weight])
            first_bias_core = np.concatenate([old_bias, -old_bias])
            second_core = np.column_stack([np.eye(output_dim), -np.eye(output_dim)])
            second_bias = np.zeros(output_dim, dtype=float)

        if width_policy == "minimal":
            requested_width = minimal_width
        elif width_policy in {"mean", "mid", "average"}:
            mean_width = 0.5 * (input_dim + output_dim)
            if minimal_width > mean_width:
                return False
            requested_width = max(minimal_width, (input_dim + output_dim) // 2)
        else:
            raise ValueError(f"Unsupported signed-ReLU width_policy: {width_policy}")

        if max_hidden_width is not None:
            if minimal_width > max_hidden_width:
                return False
            hidden_width = min(requested_width, max_hidden_width)
        else:
            hidden_width = requested_width

        if hidden_width == minimal_width:
            first_weight = first_core
            first_bias = first_bias_core
            second_weight = second_core
        else:
            first_weight = np.zeros((hidden_width, input_dim), dtype=float)
            first_weight[:minimal_width, :] = first_core
            first_bias = np.zeros(hidden_width, dtype=float)
            first_bias[:minimal_width] = first_bias_core
            second_weight = np.zeros((output_dim, hidden_width), dtype=float)
            second_weight[:, :minimal_width] = second_core

        first_layer = decompose_dense(first_weight, min(first_weight.shape), keep_residual=False)
        first_layer.gene = default_layer_gene(
            first_weight.shape[0],
            first_weight.shape[1],
            operation_type=f"relu_identity_{split_kind}_first",
            innovation_id=self.innovation_registry.get_or_create({
                "kind": "signed_relu_first",
                "split_kind": split_kind,
                "parent_innovation_id": old_gene.innovation_id,
                "input_shape": [first_weight.shape[1]],
                "output_shape": [first_weight.shape[0]],
                "width_policy": width_policy,
                "max_hidden_width": max_hidden_width,
            }),
            parent_innovation_id=old_gene.innovation_id,
            is_identity_scaffold=True,
            morphism={"kind": "signed_relu_first", "split_kind": split_kind, "width_policy": width_policy, "minimal_width": minimal_width, "requested_width": requested_width, "max_hidden_width": max_hidden_width},
        )
        second_layer = decompose_dense(second_weight, min(second_weight.shape), keep_residual=False)
        second_layer.gene = default_layer_gene(
            second_weight.shape[0],
            second_weight.shape[1],
            operation_type=f"relu_identity_{split_kind}_second",
            innovation_id=self.innovation_registry.get_or_create({
                "kind": "signed_relu_second",
                "split_kind": split_kind,
                "parent_innovation_id": old_gene.innovation_id,
                "input_shape": [second_weight.shape[1]],
                "output_shape": [second_weight.shape[0]],
                "width_policy": width_policy,
            }),
            parent_innovation_id=old_gene.innovation_id,
            is_identity_scaffold=True,
            morphism={"kind": "signed_relu_second", "split_kind": split_kind, "width_policy": width_policy, "minimal_width": minimal_width, "requested_width": requested_width, "max_hidden_width": max_hidden_width},
        )

        self.layers[layer_index] = first_layer
        self.biases[layer_index] = first_bias
        self.layers.insert(layer_index + 1, second_layer)
        self.biases.insert(layer_index + 1, second_bias)
        return True


    def add_intermediate_dense_layer(
        self,
        layer_index: int | None = None,
        hidden_width: int = 64,
        rng: np.random.Generator | None = None,
    ) -> bool:
        """Insert a compact trainable hidden layer before an existing layer.

        This is an approximate topology mutation intended for large flattened
        image inputs where a full identity scaffold such as 1024->1024 is too
        expensive. The new layer is not function-preserving; Lamarckian SGD is
        expected to refine the child after insertion.
        """
        if not self.layers:
            return False
        if layer_index is None:
            layer_index = len(self.layers) - 1
        if layer_index < 0 or layer_index >= len(self.layers):
            return False

        old_layer = self.layers[layer_index]
        old_gene = old_layer.gene.copy() if old_layer.gene is not None else default_layer_gene(old_layer.shape[0], old_layer.shape[1])
        old_weight = reconstruct_layer(old_layer)
        old_bias = self.biases[layer_index]
        output_dim, input_dim = old_weight.shape
        width = max(1, min(int(hidden_width), input_dim))
        rng = rng or np.random.default_rng()

        first_scale = np.sqrt(2.0 / input_dim)
        second_scale = np.sqrt(2.0 / width)
        first_weight = rng.standard_normal((width, input_dim), dtype=float) * first_scale
        first_bias = np.zeros(width, dtype=float)
        second_weight = rng.standard_normal((output_dim, width), dtype=float) * second_scale
        second_bias = old_bias.copy()

        new_innovation_id = self.innovation_registry.get_or_create({
            "kind": "intermediate_dense_insert",
            "parent_innovation_id": old_gene.innovation_id,
            "input_shape": [input_dim],
            "hidden_width": width,
            "output_shape": [width],
            "layer_index": layer_index,
        })
        first_layer = decompose_dense(first_weight, min(first_weight.shape), keep_residual=False)
        first_layer.gene = default_layer_gene(
            width,
            input_dim,
            operation_type="intermediate_linear",
            innovation_id=new_innovation_id,
            parent_innovation_id=old_gene.innovation_id,
            is_identity_scaffold=False,
            morphism={"kind": "intermediate_dense_insert", "hidden_width": width},
        )

        second_layer = decompose_dense(second_weight, min(second_weight.shape), keep_residual=False)
        old_gene.input_shape = (width,)
        old_gene.output_shape = (output_dim,)
        old_gene.input_channels = width
        old_gene.output_channels = output_dim
        old_gene.morphism = {"kind": "intermediate_dense_continuation", "previous_input_dim": input_dim}
        second_layer.gene = old_gene

        self.layers[layer_index] = first_layer
        self.biases[layer_index] = first_bias
        self.layers.insert(layer_index + 1, second_layer)
        self.biases.insert(layer_index + 1, second_bias)
        return True


    def add_identity_convolution_layer(
        self,
        image_shape: tuple[int, int],
        layer_index: int = 0,
        kernel_size: int = 3,
    ) -> bool:
        """Insert a same-size identity convolution layer into a flattened image net.

        The project stores operations as dense spectral matrices. This builds the
        dense matrix equivalent of a one-channel, same-padding convolution, with
        an identity kernel at insertion time: all entries are zero except the
        center weight, which is one.

        For image inputs in [0, 1], and for hidden activations after ReLU, the
        following ReLU is identity-preserving at insertion time.
        """
        if not self.layers:
            return False
        if layer_index < 0 or layer_index >= len(self.layers):
            return False
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        height, width = image_shape
        flattened_dim = height * width
        expected_dim = self.layers[layer_index].shape[1]
        if expected_dim != flattened_dim:
            return False

        radius = kernel_size // 2
        conv_weight = np.zeros((flattened_dim, flattened_dim), dtype=float)
        for row in range(height):
            for col in range(width):
                out_index = row * width + col
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        in_row = row + dy
                        in_col = col + dx
                        if 0 <= in_row < height and 0 <= in_col < width and dy == 0 and dx == 0:
                            in_index = in_row * width + in_col
                            conv_weight[out_index, in_index] = 1.0

        conv_bias = np.zeros(flattened_dim, dtype=float)
        identity_basis = np.eye(flattened_dim, dtype=float)
        conv_layer = SpectralLayer(
            residual=np.zeros_like(conv_weight, dtype=float),
            u=identity_basis.copy(),
            alpha=np.ones(flattened_dim, dtype=float),
            v=identity_basis.copy(),
            gene=default_layer_gene(
                flattened_dim,
                flattened_dim,
                operation_type="identity_conv",
                innovation_id=self.innovation_registry.get_or_create({
                    "kind": "identity_conv_insert",
                    "layer_index": layer_index,
                    "image_shape": [height, width],
                    "kernel_size": kernel_size,
                    "input_shape": [flattened_dim],
                    "output_shape": [flattened_dim],
                    "next_layer_innovation_id": (
                        None
                        if layer_index >= len(self.layers) or self.layers[layer_index].gene is None
                        else self.layers[layer_index].gene.innovation_id
                    ),
                }),
                parent_innovation_id=(
                    None
                    if layer_index >= len(self.layers) or self.layers[layer_index].gene is None
                    else self.layers[layer_index].gene.innovation_id
                ),
                is_identity_scaffold=True,
                morphism={"kind": "identity_conv_insert", "image_shape": [height, width], "kernel_size": kernel_size},
            ),
        )
        self.layers.insert(layer_index, conv_layer)
        self.biases.insert(layer_index, conv_bias)
        return True



    @staticmethod
    def _updated_gene_for_shape(gene: LayerGene | None, weight: np.ndarray) -> LayerGene:
        updated = gene.copy() if gene is not None else default_layer_gene(weight.shape[0], weight.shape[1])
        updated.input_shape = (weight.shape[1],)
        updated.output_shape = (weight.shape[0],)
        updated.input_channels = weight.shape[1]
        updated.output_channels = weight.shape[0]
        return updated


    def split_hidden_node(self, layer_index: int | None = None, node_index: int | None = None, rng: np.random.Generator | None = None) -> bool:
        """Function-preserving split of one hidden node into two twin nodes.

        The selected node's incoming weights and bias are duplicated. Its outgoing
        weights are split in half across the original and new twin, so the next
        layer receives the same total contribution immediately after mutation.
        Later SGD can make the twins diverge.
        """
        if self.hidden_layer_count <= 0:
            return False
        if layer_index is None:
            layer_index = self.hidden_layer_count - 1
        if layer_index < 0 or layer_index >= self.hidden_layer_count:
            return False

        incoming_gene = self.layers[layer_index].gene
        outgoing_gene = self.layers[layer_index + 1].gene
        incoming = reconstruct_layer(self.layers[layer_index])
        outgoing = reconstruct_layer(self.layers[layer_index + 1])
        bias = self.biases[layer_index]
        width = incoming.shape[0]
        if width <= 0:
            return False
        if node_index is None:
            if rng is None:
                rng = np.random.default_rng()
            node_index = int(rng.integers(width))
        if node_index < 0 or node_index >= width:
            return False

        new_incoming = np.insert(incoming, node_index + 1, incoming[node_index], axis=0)
        new_bias = np.insert(bias, node_index + 1, bias[node_index])

        original_outgoing_column = outgoing[:, node_index].copy()
        outgoing[:, node_index] = 0.5 * original_outgoing_column
        new_outgoing = np.insert(outgoing, node_index + 1, 0.5 * original_outgoing_column, axis=1)

        self.layers[layer_index] = decompose_dense(new_incoming, min(new_incoming.shape), keep_residual=False)
        self.layers[layer_index].gene = self._updated_gene_for_shape(incoming_gene, new_incoming)
        self.biases[layer_index] = new_bias
        self.layers[layer_index + 1] = decompose_dense(new_outgoing, min(new_outgoing.shape), keep_residual=False)
        self.layers[layer_index + 1].gene = self._updated_gene_for_shape(outgoing_gene, new_outgoing)
        return True



    @staticmethod
    def _closest_correlated_node(features: np.ndarray, node_index: int) -> tuple[int, float] | None:
        """Return the non-deleted node most correlated with node_index.

        The scale is the Pearson correlation coefficient. The caller can use it
        as a cheap local linear approximation h_deleted ~= alpha * h_target.
        """
        width = features.shape[0]
        if width <= 1:
            return None
        centered = features - np.mean(features, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        target_norm = norms[node_index]
        if target_norm <= 1e-12:
            centered = features
            norms = np.linalg.norm(centered, axis=1)
            target_norm = norms[node_index]
        if target_norm <= 1e-12:
            return None

        best_index: int | None = None
        best_corr = 0.0
        best_abs_corr = -1.0
        for candidate in range(width):
            if candidate == node_index or norms[candidate] <= 1e-12:
                continue
            corr = float(np.dot(centered[node_index], centered[candidate]) / (target_norm * norms[candidate]))
            abs_corr = abs(corr)
            if abs_corr > best_abs_corr:
                best_index = candidate
                best_corr = corr
                best_abs_corr = abs_corr
        if best_index is None:
            return None
        return best_index, best_corr


    def delete_hidden_node(self, layer_index: int | None = None, node_index: int | None = None) -> bool:
        """Delete one hidden node by merging its outgoing effect into a neighbor.

        The removed node is matched to the most correlated remaining node using
        its incoming weights plus bias. Its outgoing column is then added to the
        matched node's outgoing column, scaled by that correlation coefficient.
        Layer deletion remains unchanged for now.
        """
        if self.hidden_layer_count <= 0:
            return False
        if layer_index is None:
            layer_index = self.hidden_layer_count - 1
        if layer_index < 0 or layer_index >= self.hidden_layer_count:
            return False

        incoming_gene = self.layers[layer_index].gene
        outgoing_gene = self.layers[layer_index + 1].gene
        incoming = reconstruct_layer(self.layers[layer_index])
        outgoing = reconstruct_layer(self.layers[layer_index + 1])
        bias = self.biases[layer_index]
        width = incoming.shape[0]
        if width <= 1:
            return False
        if node_index is None:
            if rng is None:
                rng = np.random.default_rng()
            node_index = int(rng.integers(width))
        if node_index < 0 or node_index >= width:
            return False

        features = np.column_stack([incoming, bias])
        merge_target = self._closest_correlated_node(features, node_index)
        if merge_target is not None:
            target_index, alpha = merge_target
            outgoing[:, target_index] = outgoing[:, target_index] + alpha * outgoing[:, node_index]

        new_incoming = np.delete(incoming, node_index, axis=0)
        new_bias = np.delete(bias, node_index)
        new_outgoing = np.delete(outgoing, node_index, axis=1)

        self.layers[layer_index] = decompose_dense(new_incoming, min(new_incoming.shape), keep_residual=False)
        self.layers[layer_index].gene = self._updated_gene_for_shape(incoming_gene, new_incoming)
        self.biases[layer_index] = new_bias
        self.layers[layer_index + 1] = decompose_dense(new_outgoing, min(new_outgoing.shape), keep_residual=False)
        self.layers[layer_index + 1].gene = self._updated_gene_for_shape(outgoing_gene, new_outgoing)
        return True

    def _activation_before_layer(self, inputs: np.ndarray, layer_index: int) -> np.ndarray | None:
        """Return the activation that enters layer_index for a batch of inputs."""
        if layer_index < 0 or layer_index >= len(self.layers):
            return None
        activation = inputs.astype(float)
        for idx in range(layer_index):
            weight = reconstruct_layer(self.layers[idx])
            activation = activation @ weight.T + self.biases[idx]
            activation = np.maximum(activation, 0.0)
        expected_dim = self.layers[layer_index].shape[1]
        if activation.ndim != 2 or activation.shape[1] != expected_dim:
            return None
        return activation

    def delete_hidden_layer_ridge(
        self,
        layer_index: int | None = None,
        samples: np.ndarray | None = None,
        ridge_lambda: float = 1e-3,
    ) -> bool:
        """Delete one hidden layer by fitting a direct replacement with ridge regression.

        The fitted replacement maps the activation entering the deleted layer to
        the pre-activation output of the downstream layer in the original block.
        That preserves more information than fitting post-activation outputs.
        """
        if samples is None:
            return self.delete_hidden_layer(layer_index)
        if self.hidden_layer_count <= 0:
            return False
        if layer_index is None:
            layer_index = self.hidden_layer_count - 1
        if layer_index < 0 or layer_index >= self.hidden_layer_count:
            return False

        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 2 or samples.shape[0] == 0:
            return False
        block_input = self._activation_before_layer(samples, layer_index)
        if block_input is None:
            return False

        first_gene = self.layers[layer_index].gene
        second_gene = self.layers[layer_index + 1].gene
        first_weight = reconstruct_layer(self.layers[layer_index])
        second_weight = reconstruct_layer(self.layers[layer_index + 1])
        first_bias = self.biases[layer_index]
        second_bias = self.biases[layer_index + 1]

        hidden_pre_activation = block_input @ first_weight.T + first_bias
        hidden_activation = np.maximum(hidden_pre_activation, 0.0)
        target_pre_activation = hidden_activation @ second_weight.T + second_bias

        augmented_input = np.column_stack([block_input, np.ones(block_input.shape[0], dtype=float)])
        penalty = np.eye(augmented_input.shape[1], dtype=float) * max(0.0, float(ridge_lambda))
        penalty[-1, -1] = 0.0
        normal_matrix = augmented_input.T @ augmented_input + penalty
        rhs = augmented_input.T @ target_pre_activation
        try:
            theta = np.linalg.solve(normal_matrix, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(normal_matrix) @ rhs

        merged_weight = theta[:-1, :].T
        merged_bias = theta[-1, :]

        self.layers[layer_index] = decompose_dense(merged_weight, min(merged_weight.shape), keep_residual=False)
        merged_gene = default_layer_gene(
            merged_weight.shape[0],
            merged_weight.shape[1],
            operation_type="ridge_layer_delete_merge",
            innovation_id=self.innovation_registry.get_or_create({
                "kind": "delete_hidden_layer_ridge_merge",
                "deleted_innovation_id": None if first_gene is None else first_gene.innovation_id,
                "downstream_innovation_id": None if second_gene is None else second_gene.innovation_id,
                "input_shape": [merged_weight.shape[1]],
                "output_shape": [merged_weight.shape[0]],
                "ridge_lambda": float(ridge_lambda),
            }),
            parent_innovation_id=None if second_gene is None else second_gene.innovation_id,
            is_identity_scaffold=False,
            morphism={
                "kind": "delete_hidden_layer_ridge_merge",
                "deleted_innovation_id": None if first_gene is None else first_gene.innovation_id,
                "downstream_innovation_id": None if second_gene is None else second_gene.innovation_id,
                "ridge_lambda": float(ridge_lambda),
                "sample_count": int(samples.shape[0]),
            },
        )
        self.layers[layer_index].gene = merged_gene
        self.biases[layer_index] = merged_bias
        del self.layers[layer_index + 1]
        del self.biases[layer_index + 1]
        return True


    def delete_hidden_layer(self, layer_index: int | None = None) -> bool:
        """Delete one hidden layer by approximately merging it into the next layer.

        For adjacent dense maps x -> W1 x + b1 -> ReLU -> W2 h + b2, deletion
        uses the linearized merge W = W2 W1 and b = W2 b1 + b2. This intentionally
        ignores the removed ReLU, so it is an approximate LEMONADE-style reduction
        that should be followed by short SGD refinement.
        """
        if self.hidden_layer_count <= 0:
            return False
        if layer_index is None:
            layer_index = self.hidden_layer_count - 1
        if layer_index < 0 or layer_index >= self.hidden_layer_count:
            return False

        first_gene = self.layers[layer_index].gene
        second_gene = self.layers[layer_index + 1].gene
        first_weight = reconstruct_layer(self.layers[layer_index])
        second_weight = reconstruct_layer(self.layers[layer_index + 1])
        first_bias = self.biases[layer_index]
        second_bias = self.biases[layer_index + 1]

        merged_weight = second_weight @ first_weight
        merged_bias = second_weight @ first_bias + second_bias

        self.layers[layer_index] = decompose_dense(merged_weight, min(merged_weight.shape), keep_residual=False)
        merged_gene = default_layer_gene(
            merged_weight.shape[0],
            merged_weight.shape[1],
            operation_type="linearized_layer_delete_merge",
            innovation_id=self.innovation_registry.get_or_create({
                "kind": "delete_hidden_layer_linearized_merge",
                "deleted_innovation_id": None if first_gene is None else first_gene.innovation_id,
                "downstream_innovation_id": None if second_gene is None else second_gene.innovation_id,
                "input_shape": [merged_weight.shape[1]],
                "output_shape": [merged_weight.shape[0]],
            }),
            parent_innovation_id=None if second_gene is None else second_gene.innovation_id,
            is_identity_scaffold=False,
            morphism={
                "kind": "delete_hidden_layer_linearized_merge",
                "deleted_innovation_id": None if first_gene is None else first_gene.innovation_id,
                "downstream_innovation_id": None if second_gene is None else second_gene.innovation_id,
            },
        )
        self.layers[layer_index].gene = merged_gene
        self.biases[layer_index] = merged_bias
        del self.layers[layer_index + 1]
        del self.biases[layer_index + 1]
        return True


    def forward(self, x: np.ndarray) -> np.ndarray:
        activation = x.astype(float)
        for idx, layer in enumerate(self.layers):
            weight = reconstruct_layer(layer)
            bias = self.biases[idx]
            activation = activation @ weight.T + bias
            if idx < len(self.layers) - 1:
                activation = np.maximum(activation, 0.0)
            else:
                # return raw logits for final layer; caller decides activation/loss
                activation = activation
        return activation

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray, loss: str = "mse") -> float:
        outputs = self.forward(inputs)
        if loss == "mse":
            return mse_loss(outputs, targets)
        elif loss == "bce":
            # apply sigmoid to logits then compute BCE
            probs = 1.0 / (1.0 + np.exp(-outputs))
            return bce_loss(probs, targets)
        elif loss == "balanced_bce":
            probs = 1.0 / (1.0 + np.exp(-outputs))
            return balanced_bce_loss(probs, targets)
        elif loss == "ce":
            return ce_loss(outputs, targets)
        else:
            raise ValueError(f"Unsupported loss: {loss}")

    def accuracy(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        outputs = self.forward(inputs)
        outputs = np.asarray(outputs)
        if outputs.ndim == 2 and outputs.shape[1] > 1:
            preds = np.argmax(outputs, axis=1)
            return float(np.mean(preds == np.asarray(targets).reshape(-1).astype(int)))
        else:
            predictions = outputs.reshape(-1) >= 0.0
            return float(np.mean(predictions == targets.reshape(-1)))

    def to_dense_weights(self) -> List[np.ndarray]:
        return [reconstruct_layer(layer) for layer in self.layers]





