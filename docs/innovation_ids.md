# Innovation IDs for LEMONADE-style topology morphisms

Sequential layer position is not a valid gene identity in this project. A function-preserving morphism can insert an identity scaffold before or between existing layers, shifting list positions while leaving the older layers as the same evolutionary genes. If crossover aligned by list index, an older semantic layer could be crossed with a newly inserted identity layer.

Each evolvable `SpectralLayer` therefore carries a `LayerGene` with a persistent `innovation_id`. The ID is ancestry metadata only: it must not affect the forward pass, loss, optimizer, or SVD parameterization.

Rules:

- Cloning preserves innovation IDs.
- Gradient descent and weight mutation never change innovation IDs.
- New structural genes receive an ID from `InnovationRegistry.get_or_create(signature)`: the first occurrence creates a fresh ID, later identical structural morphisms reuse it.
- Existing downstream layers keep their IDs even when insertion shifts their current list position.
- If a morphism replaces one layer with several physical layers, the original layer ID is not reused; each new physical layer receives/reuses an innovation ID from its structural signature and records the old ID as `parent_innovation_id`.
- Legacy layers without metadata are migrated once by `ensure_network_innovations` in current sequential order.

Crossover should use `align_homologous_layers(parent_a, parent_b)` or `plan_network_crossover(...)`. The returned `CrossoverPlan` separates:

- `matched_pairs`: same innovation ID, even if node split/delete changed tensor shape; crossover uses overlapping SVD coordinates and keeps the anchor parent shape;
- `incompatible_pairs`: reserved for future cases that cannot be safely crossed;
- `unmatched_from_a` / `unmatched_from_b`: non-identity genes found in only one parent;
- `inherited_identity_scaffolds`: unmatched identity-preserving scaffolds, kept separate so they do not shift primary matching.

The key distinction is: a layer's current list position is its present execution order, while its innovation ID is its evolutionary ancestry. Crossover aligns ancestry first, then decides what to do with unmatched or scaffold genes.


Layer deletion is special: the linearized merge is treated as a new structural gene. It receives a fresh innovation ID and records the deleted/upstream and downstream source IDs in morphism metadata. This prevents a post-deletion merged layer from being treated as directly homologous to the old downstream layer.


Innovation reuse policy: structural morphisms are keyed by a canonical signature containing the morphism kind, relevant ancestral innovation IDs, shapes, and operation metadata. This means two individuals that independently insert the same identity scaffold in the same ancestral context receive the same innovation ID, even if the insertion happens in different generations. Different contexts or different morphism parameters receive different IDs.


Signed-ReLU expansion is function-preserving but not gene-preserving. A layer `W` is replaced by `[W; -W]` followed by `[I, -I]`, so the composed function is identical. However, neither new physical layer is the same structural gene as the original single layer. Both new layers therefore receive innovation IDs keyed by their signed-ReLU morphism signatures, and both store the original layer ID as their parent.


Signed-ReLU expansion has two exact variants. For a layer `n -> m`, output-sign expansion gives `n -> 2m -> m` by splitting `Wx+b` into positive and negative channels. Input-sign expansion gives `n -> 2n -> m` by splitting the input coordinates into positive and negative channels, then applying the original weight as `[W, -W]`. The implementation chooses the smaller exact variant: `n -> 2n -> m` when `n < m`, otherwise `n -> 2m -> m`.
