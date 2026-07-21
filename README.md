# Spectral Evolution Research Prototype

This repository contains a Python research prototype for evolutionary optimization of fixed-architecture neural networks using a generation-wise spectral representation.

## Research question

Can spectral mode mutation and mode-based crossover produce more meaningful evolutionary variation than raw weight mutation and raw weight crossover?

The prototype evaluates a hypothesis under controlled conditions:

- each individual is a fixed-architecture neural network;
- dense layer weights are represented with SVD-derived mode packages;
- the algorithm mutates and recombines rank-one spectral modes, not raw weights;
- an optional local refinement stage updates spectral parameters: strengths (`alpha`), biases, and, when enabled, mode directions (`u` and `v`).

## SVD representation

For each dense weight matrix `W_l`:

- compute `W_l = U_l Sigma_l V_l^T`;
- keep up to `r_max` dominant modes;
- optionally preserve a residual matrix so discarded modes are not lost;
- represent the trainable subspace as `W_l = residual + sum_i alpha_i u_i v_i^T`.

Each mode package contains:

- input direction `v_i`;
- output direction `u_i`;
- strength coefficient `alpha_i`.

The SVD is a temporary coordinate system for each generation. After mutation, crossover, or local refinement, `U` and `V` may no longer be orthogonal, and later decomposition can recompute SVD afresh.

## Generation loop

1. evaluate the complete neural network using a global loss;
2. select parents by tournament selection;
3. create offspring with half-rank spectral mode crossover and spectral mutation;
4. optionally apply local spectral-parameter refinement to offspring;
5. replace the population with the new generation.

## Limitations

- fixed architecture only;
- corresponding layers must have identical matrix dimensions for crossover;
- mode index `i` from two independently computed SVDs is not guaranteed to have the same semantic meaning;
- whole-network coadaptation may make crossover destructive.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the minimal XOR benchmark:

```bash
python experiments/run_xor.py
```

Run tests:

```bash
pytest
```
