# Design Document

## Formal Notation

Let `W_l in R^(n_l x m_l)` denote the dense weight matrix for layer `l`.

A spectral decomposition of `W_l` is:

```text
W_l = U_l Sigma_l V_l^T
```

or equivalently:

```text
W_l = R_l + sum_{i=1}^{r_l} alpha_{l,i} u_{l,i} v_{l,i}^T
```

where:

- `r_l = min(r_max, n_l, m_l)`;
- `u_{l,i} in R^n_l` is the output-space direction;
- `v_{l,i} in R^m_l` is the input-space direction;
- `alpha_{l,i}` is the mode strength;
- `R_l in R^(n_l x m_l)` is the residual matrix capturing discarded components.

## Algorithm Pseudocode

```text
initialize population of fixed-architecture networks
for generation = 1..G:
    evaluate each individual on whole-network loss
    select parents by tournament selection
    generate children by spectral mutation and crossover
    optionally apply local spectral-parameter refinement to children
    replace population with new generation
```

## Mutation Rules

For version 1, use conservative directional mutation on one selected mode in one selected layer.

Given mode `i` in layer `l`:

```text
u'_i = normalize(u_i + epsilon_u)
v'_i = normalize(v_i + epsilon_v)
```

where `epsilon_u` and `epsilon_v` are Gaussian perturbations.

Supported options:

- mutate only `u`;
- mutate only `v`;
- mutate both;
- mutate a single mode in a single selected layer.

Mutation preserves the layer shape and leaves `alpha` unchanged.

## Crossover Rules

### Spectral Mode Crossover

Given Parent A and Parent B:

- choose Parent A as anchor;
- select one corresponding layer `l`;
- independently decompose both parents' `W_l` matrices by SVD;
- retain the same rank budget `r_l` for the child layer;
- select a subset of complete mode packages from Parent B;
- replace the corresponding mode packages in Parent A;
- keep Parent A's residual matrix unchanged.

### Crossover Strategies

1. `half_rank`: swap the upper half of spectral mode packages from the donor into the anchor.
2. `single_point`: choose a cut point and swap all modes after it.
3. `uniform`: choose the donor parent independently per mode package.
4. raw-weight crossover baseline: perform crossover on flattened weight entries.

The code accepts `"sbx"` as a compatibility alias for `half_rank`, but this is not classical simulated binary crossover.

## Local Spectral-Parameter Refinement

Given a spectral network:

- treat `alpha` as trainable PyTorch parameters;
- optionally treat `u` and `v` as trainable parameters;
- optionally treat biases as trainable parameters;
- perform `K` gradient steps on the whole-network loss;
- renormalize `u` and `v` directions when they are optimized, folding their scale back into `alpha`.

This is used after spectral mutation or crossover to fine-tune mode strengths and, when enabled, mode directions.

## Planned Baselines And Metrics

### Baselines

- A: standard GA with Gaussian weight mutation, no crossover
- B: standard GA with raw-weight crossover
- C: spectral mutation, no crossover
- D: spectral mutation + spectral mode crossover
- E: spectral mutation + spectral mode crossover + spectral-parameter refinement

### Metrics

Track per generation:

- best loss;
- mean loss;
- best accuracy;
- number of evaluations;
- child loss relative to parent losses;
- fraction of children better than the worse parent;
- fraction of children better than the better parent;
- runtime.

## Minimal Benchmark

Use a fixed small MLP for XOR classification, e.g. `2 -> 8 -> 1`.

- input: 2 features
- hidden: 8 units
- output: 1 logit with sigmoid used by the binary cross-entropy loss
- loss: mean squared error or binary cross entropy
- population: small, run on CPU quickly

## Limitations

- fixed architecture only
- no topology mutation
- corresponding layer dimensions must match for crossover
- mode indices from independent SVDs are not semantically aligned
- this is a research prototype, not a production system
