# Codex Project Prompt

We are developing a research prototype called `SVD_Crossover`.

Repository: <https://github.com/aragoni5051/SVD_Crossover>

## Research objective

We want to determine whether neural-network genetic algorithms can perform more meaningful crossover by recombining SVD-derived rank-one modes instead of cutting or averaging raw weight matrices.

For each dense weight matrix `W`, we use:

```text
W = residual + sum_i alpha_i * u_i * v_i^T
```

A mode package consists of `(alpha_i, u_i, v_i)`. Our crossover transfers complete mode packages between homologous layers. The primary operator is half-rank crossover: the child keeps the leading modes from the anchor parent and receives the remaining modes from the donor.

## Intended evolutionary cycle

1. Refine every individual using gradient descent.
2. Evaluate individuals on validation data.
3. Select parents using tournament selection.
4. Produce children using SVD-mode crossover.
5. Apply spectral mutation to alpha, u, v, and biases.
6. Preserve an elite and repeat.

## Current CIFAR-10 implementation

The main experiment is `experiments/run_cifar10_svd_crossover.py`.

It currently uses:

- CIFAR-10 RGB images resized and flattened into vectors.
- A fixed-topology dense MLP.
- Adam-based local GD refinement.
- Validation loss for evolutionary selection.
- An untouched test set for final reporting.
- Innovation-aware homologous-layer alignment.
- Half-rank SVD crossover.
- Spectral-mode mutation.
- Elitism and tournament selection.

It writes `curves.csv`, `run.json`, and `best_network.json`.

## Important limitations

- This CIFAR experiment is a dense network, not a CNN.
- Native 32x32 RGB input is expensive on CPU because it creates 3,072-dimensional dense inputs.
- Use resized 16x16 images and sampled datasets for CPU experiments.
- This runner evolves weights while keeping the MLP topology fixed. It is not yet complete NAS.
- LEMONADE code exists, but that experiment is postponed while no NVIDIA GPU is available.
- Do not describe the current runner as paper-exact LEMONADE.
- Never use the CIFAR-10 test set for parent selection or hyperparameter tuning.

## Work completed

- Spectral decomposition and reconstruction of dense layers.
- Complete SVD-mode package crossover for homologous layers.
- Innovation IDs and layer alignment.
- Shape-aware crossover experiments.
- Function-preserving and approximate topology mutations in `SpectralNetwork`.
- Dense and spectral GD refinement.
- Digits experiments and controlled crossover comparisons.
- LEMONADE-oriented experiments kept separate from our method.
- A standalone CPU-oriented CIFAR-10 SVD crossover runner.
- Focused tests for crossover, spectral operations, innovation IDs, and mutations. The 40 focused tests passed before the initial push.

## First run: smoke test

Run this first to verify the complete pipeline rather than measure accuracy:

```powershell
python experiments\run_cifar10_svd_crossover.py `
  --image-size 8 `
  --max-train-samples 1000 `
  --max-test-samples 500 `
  --validation-size 200 `
  --population 2 `
  --generations 2 `
  --hidden-widths 32 `
  --r-max 8 `
  --gd-steps 2 `
  --output-dir results\cifar10_svd_smoke
```

Confirm that GD, selection, crossover, mutation, CSV output, and model serialization all execute. Inspect `run.json` and verify that `crossover_attempts` and `mutated_modes` are nonzero when their rates are enabled.

## Practical CPU experiment

After the smoke test passes, run:

```powershell
python experiments\run_cifar10_svd_crossover.py `
  --image-size 16 `
  --max-train-samples 12000 `
  --max-test-samples 2000 `
  --validation-size 2000 `
  --population 4 `
  --generations 10 `
  --gd-steps 5 `
  --hidden-widths 128,64 `
  --r-max 16 `
  --crossover-rate 0.7 `
  --mode-mutation-rate 0.1 `
  --output-dir results\cifar10_svd_cpu_run1
```

Report validation accuracy, test accuracy, runtime, crossover count, and mutated-mode count. Do not silently expand to native 32x32 or the full dataset because CPU runtime can increase sharply.

## Next scientific experiment

After verifying the runner, make a controlled comparison between:

1. GD plus spectral mutation with crossover disabled.
2. Raw-weight crossover.
3. Same-shape SVD-mode crossover.

Use identical seeds, population sizes, training budgets, initialization, and data splits. The immediate goal is to isolate whether SVD crossover itself provides an advantage. A single SVD-crossover run cannot establish that crossover helped.

Before changing code, inspect the repository and preserve the separation between our method and the LEMONADE baseline. Do not make unrelated changes unless a concrete bug is found.
