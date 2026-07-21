# LEMONADE Paper-Exact Baseline

This project keeps the old `LEMONADE-style` CIFAR runner as an approximate baseline, and adds a separate paper-oriented Search Space I runner at `experiments/run_cifar10_lemonade_paper.py`. Use the separate runner for baseline comparisons.

The exact paper baseline for comparison with SVD-driven crossover should be a
separate implementation so the comparison does not mix ideas.

Primary source:

- Thomas Elsken, Jan Hendrik Metzen, Frank Hutter, "Efficient Multi-objective
  Neural Architecture Search via Lamarckian Evolution", ICLR 2019,
  arXiv:1804.09081.

## Implemented Path

Use `experiments/run_cifar10_lemonade_paper.py` for the comparison baseline. It implements Algorithm 1 mechanics for CIFAR-10 Search Space I: Pareto-front population, KDE inverse-density parent sampling, generated-child cheap-objective acceptance, uniform paper mutation families, and paper-scale search defaults. Final 600-epoch retraining stays in the retrain script.

## What The Current Runner Does Not Match

`experiments/run_cifar10_cnn_lemonade.py` differs from the paper in important
ways:

- It uses a fixed three-stage CNN architecture encoding.
- It mutates channels, stage depths, separable flags, and residual flags.
- It warm-starts children by copying overlapping PyTorch tensors.
- It keeps a fixed-size selected population using Pareto fronts plus crowding.
- It does not implement LEMONADE's KDE-density parent distribution.
- It does not implement LEMONADE's generated-child acceptance distribution.
- It does not implement exact Net2Net-style morphism operators.
- It does not implement distillation-based approximate network morphisms.

This runner is useful for smoke tests and rough intuition, but it must not be
reported as "LEMONADE" in a paper comparison.

## Required Paper Algorithm

The paper algorithm maintains a trained population on the Pareto front.

For each generation:

1. Estimate a density over cheap objectives using KDE.
2. Select parents with probability inversely related to that density.
3. Generate `n_pc` possible children by network morphism or approximate network
   morphism mutations.
4. Estimate a density over generated children using cheap objectives.
5. Accept only `n_ac` children for expensive training/evaluation, again favoring
   underrepresented cheap-objective regions.
6. Train/evaluate accepted children on the expensive objective.
7. Replace the population with the Pareto front of previous population plus
   accepted children.

The comparison objectives used in our CIFAR baseline should start with:

- validation error or validation loss;
- parameter count.

Other paper objectives such as inference time, memory, and energy can be added
after the two-objective version is correct.

## Required Paper Operators

### Network Morphisms

These should preserve the represented function at mutation time:

- Insert `Conv-BatchNorm-ReLU` block.
  - The convolution is initialized as an identity mapping.
  - BatchNorm is initialized to preserve the input distribution.
  - ReLU is valid when inserted after an existing ReLU because ReLU is
    idempotent.
- Increase filters of a convolution.
  - The changed convolution and subsequent convolution are padded so the
    pre-existing function is preserved.
- Add skip connection.
  - Concatenation skip: subsequent convolutions are zero-padded for added
    channels.
  - Additive skip: use a learnable convex-combination parameter initialized so
    the original path is unchanged.

### Approximate Network Morphisms

These are allowed to shrink capacity and are not exactly function-preserving.
The child should be initialized by knowledge distillation from the parent, using 5 SGD repair epochs by default in `experiments/run_cifar10_lemonade_paper.py`:

- remove a randomly chosen layer;
- remove a randomly chosen skip connection;
- prune a randomly chosen convolutional layer by removing `1/2` or `1/4` of its
  filters;
- replace a randomly chosen convolution with a depthwise separable convolution.

## Required Search Spaces

The exact comparison should implement the paper's CIFAR search space rather than
the current fixed three-stage shorthand. The paper evaluates:

- a non-modularized architecture search space;
- a cell-based search space.

The first implementation target should be the non-modularized CIFAR search space
with two objectives: validation error and parameter count.

## Required Training Protocol

Search-time training:

- CIFAR-10 split: `45,000` train / `5,000` validation.
- Batch size: `64`.
- Optimizer: SGD with momentum.
- Learning rate: cosine annealing from `0.01` to `0`.
- Weight decay: `5e-4`.
- Search child training budget: paper-scale, not the current tiny smoke budget.

Final evaluation:

- Retrain selected final architectures from scratch.
- Paper-scale final retraining is long, e.g. hundreds of epochs.
- Report test accuracy/error and parameter count for selected Pareto points.

## Implementation Plan In This Repo

Create a new runner rather than modifying the approximate one:

- `src/spectral_ga/lemonade_paper/`
  - architecture graph representation;
  - operators for exact morphisms;
  - operators for approximate morphisms plus distillation;
  - KDE-density parent and child selection;
  - Pareto front maintenance.
- `experiments/run_cifar10_lemonade_paper.py`
  - paper-style search driver;
  - output `population.csv`, `children.csv`, `curves.csv`, `raw_runs.csv`.
- `experiments/retrain_cifar10_lemonade_paper.py`
  - final retraining for selected architectures.

Keep `experiments/run_cifar10_cnn_lemonade.py` as the approximate baseline, but
label it as `LEMONADE-style`, not exact LEMONADE.
## Current Reproduction Commands

CPU correctness probe on this AMD/CPU-only machine:

```powershell
python experiments\run_cifar10_lemonade_paper.py `
  --generations 5 `
  --n-pc 6 `
  --n-ac 2 `
  --epochs-per-child 1 `
  --distill-epochs 2 `
  --max-train-samples 1000 `
  --max-val-samples 500 `
  --batch-size 32 `
  --output-dir results\cifar10_lemonade_paper_cpu_probe_fixed2 `
  --device auto
```

Retrain architectures from a search result:

```powershell
python experiments\retrain_cifar10_lemonade_paper.py `
  --pareto-csv results\cifar10_lemonade_paper_cpu_probe_fixed2\pareto_population.csv `
  --limit 3 `
  --epochs 20 `
  --output-dir results\cifar10_lemonade_paper_retrain_probe `
  --device auto
```

Full paper-scale search requires an NVIDIA CUDA machine:

```powershell
python experiments\run_cifar10_lemonade_paper.py `
  --output-dir results\cifar10_lemonade_paper_search_space_i_full `
  --device cuda
```

The search runner now writes `curves.csv`, `pareto_population.csv`, and `children.csv` so accepted, rejected, and duplicate proposals can be audited.

## Remaining Known Gaps

- The implemented Search Space I is still a simplified three-stage graph rather than a fully general paper graph.
- Concatenation skip morphisms are not implemented; additive convex skips are implemented.
- ANM repair freezes exactly inherited parameters, which is a practical proxy for freezing unaffected layers.
- This local machine has AMD graphics and CPU-only PyTorch, so paper-scale timing/results must be run elsewhere.

