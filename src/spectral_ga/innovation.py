from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json


@dataclass
class InnovationRegistry:
    """Monotonic innovation ID source with NEAT-style signature reuse.

    `fresh()` is still available for legacy migration and unique initial genes.
    Structural morphisms should call `get_or_create(signature)` so the same
    operation in the same ancestral context receives the same innovation ID.
    """

    next_id: int = 0
    signature_to_id: dict[str, int] = field(default_factory=dict)

    @staticmethod
    def canonical_signature(signature: Any) -> str:
        if isinstance(signature, str):
            return signature
        return json.dumps(signature, sort_keys=True, separators=(",", ":"))

    def fresh(self) -> int:
        innovation_id = int(self.next_id)
        self.next_id += 1
        return innovation_id

    def get_or_create(self, signature: Any) -> int:
        key = self.canonical_signature(signature)
        if key in self.signature_to_id:
            return int(self.signature_to_id[key])
        innovation_id = self.fresh()
        self.signature_to_id[key] = innovation_id
        return innovation_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_id": int(self.next_id),
            "signature_to_id": dict(self.signature_to_id),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InnovationRegistry":
        return cls(
            next_id=int(data.get("next_id", 0)),
            signature_to_id={str(k): int(v) for k, v in data.get("signature_to_id", {}).items()},
        )


@dataclass
class LayerGene:
    """Persistent evolutionary ancestry metadata for one evolvable layer."""

    innovation_id: int | None = None
    parent_innovation_id: int | None = None
    operation_type: str = "linear"
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    input_channels: int | None = None
    output_channels: int | None = None
    is_identity_scaffold: bool = False
    morphism: dict[str, Any] | None = None

    def copy(self) -> "LayerGene":
        return LayerGene.from_dict(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.input_shape is not None:
            data["input_shape"] = list(self.input_shape)
        if self.output_shape is not None:
            data["output_shape"] = list(self.output_shape)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LayerGene":
        if data is None:
            return cls()
        input_shape = data.get("input_shape")
        output_shape = data.get("output_shape")
        return cls(
            innovation_id=None if data.get("innovation_id") is None else int(data["innovation_id"]),
            parent_innovation_id=(
                None if data.get("parent_innovation_id") is None else int(data["parent_innovation_id"])
            ),
            operation_type=str(data.get("operation_type", "linear")),
            input_shape=None if input_shape is None else tuple(int(v) for v in input_shape),
            output_shape=None if output_shape is None else tuple(int(v) for v in output_shape),
            input_channels=(None if data.get("input_channels") is None else int(data["input_channels"])),
            output_channels=(None if data.get("output_channels") is None else int(data["output_channels"])),
            is_identity_scaffold=bool(data.get("is_identity_scaffold", False)),
            morphism=data.get("morphism"),
        )


@dataclass
class MatchedLayerPair:
    innovation_id: int
    index_a: int
    index_b: int
    gene_a: LayerGene
    gene_b: LayerGene


@dataclass
class IncompatibleLayerPair:
    innovation_id: int
    index_a: int
    index_b: int
    reason: str


@dataclass
class CrossoverPlan:
    matched_pairs: list[MatchedLayerPair]
    unmatched_from_a: list[tuple[int, LayerGene]]
    unmatched_from_b: list[tuple[int, LayerGene]]
    inherited_identity_scaffolds: list[tuple[str, int, LayerGene]]
    incompatible_pairs: list[IncompatibleLayerPair]


def default_layer_gene(
    output_dim: int,
    input_dim: int,
    operation_type: str = "linear",
    innovation_id: int | None = None,
    parent_innovation_id: int | None = None,
    is_identity_scaffold: bool = False,
    morphism: dict[str, Any] | None = None,
) -> LayerGene:
    return LayerGene(
        innovation_id=innovation_id,
        parent_innovation_id=parent_innovation_id,
        operation_type=operation_type,
        input_shape=(input_dim,),
        output_shape=(output_dim,),
        input_channels=input_dim,
        output_channels=output_dim,
        is_identity_scaffold=is_identity_scaffold,
        morphism=morphism,
    )


def ensure_network_innovations(network: Any, registry: InnovationRegistry | None = None) -> InnovationRegistry:
    """Assign missing innovation IDs once, preserving existing IDs afterward."""
    if registry is None:
        max_existing = -1
        for layer in network.layers:
            gene = getattr(layer, "gene", None)
            if gene is not None and gene.innovation_id is not None:
                max_existing = max(max_existing, int(gene.innovation_id))
        registry = InnovationRegistry(max_existing + 1)

    for layer in network.layers:
        if getattr(layer, "gene", None) is None:
            layer.gene = default_layer_gene(layer.shape[0], layer.shape[1])
        if layer.gene.innovation_id is None:
            layer.gene.innovation_id = registry.fresh()
        if layer.gene.input_shape is None or layer.gene.output_shape is None:
            layer.gene.input_shape = (layer.shape[1],)
            layer.gene.output_shape = (layer.shape[0],)
        if layer.gene.input_channels is None:
            layer.gene.input_channels = layer.shape[1]
        if layer.gene.output_channels is None:
            layer.gene.output_channels = layer.shape[0]
    return registry


def _layers_by_innovation(network: Any) -> dict[int, tuple[int, Any]]:
    by_id: dict[int, tuple[int, Any]] = {}
    for index, layer in enumerate(network.layers):
        gene = getattr(layer, "gene", None)
        if gene is None or gene.innovation_id is None:
            continue
        by_id[int(gene.innovation_id)] = (index, layer)
    return by_id


def align_homologous_layers(parent_a: Any, parent_b: Any) -> CrossoverPlan:
    """Align layers by innovation ancestry, never by current list position."""
    ensure_network_innovations(parent_a)
    ensure_network_innovations(parent_b)
    a_by_id = _layers_by_innovation(parent_a)
    b_by_id = _layers_by_innovation(parent_b)

    matched_pairs: list[MatchedLayerPair] = []
    incompatible_pairs: list[IncompatibleLayerPair] = []
    unmatched_from_a: list[tuple[int, LayerGene]] = []
    unmatched_from_b: list[tuple[int, LayerGene]] = []
    identity_scaffolds: list[tuple[str, int, LayerGene]] = []

    for innovation_id in sorted(set(a_by_id) & set(b_by_id)):
        index_a, layer_a = a_by_id[innovation_id]
        index_b, layer_b = b_by_id[innovation_id]
        matched_pairs.append(MatchedLayerPair(innovation_id, index_a, index_b, layer_a.gene.copy(), layer_b.gene.copy()))

    for side, only_ids, by_id, unmatched in (
        ("a", sorted(set(a_by_id) - set(b_by_id)), a_by_id, unmatched_from_a),
        ("b", sorted(set(b_by_id) - set(a_by_id)), b_by_id, unmatched_from_b),
    ):
        for innovation_id in only_ids:
            index, layer = by_id[innovation_id]
            gene = layer.gene.copy()
            if gene.is_identity_scaffold:
                identity_scaffolds.append((side, index, gene))
            else:
                unmatched.append((index, gene))

    return CrossoverPlan(
        matched_pairs=matched_pairs,
        unmatched_from_a=unmatched_from_a,
        unmatched_from_b=unmatched_from_b,
        inherited_identity_scaffolds=identity_scaffolds,
        incompatible_pairs=incompatible_pairs,
    )
