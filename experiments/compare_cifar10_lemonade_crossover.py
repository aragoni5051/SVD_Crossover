from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LEMONADE-style CIFAR-10 crossover comparison with safe CIFAR defaults."
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--methods", choices=["three", "no_and_same_dim", "same_dim_only", "annealed_only", "all_four"], default="no_and_same_dim")
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_signedrelu_same_dim_100x300_rate_0.3"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--refine-steps", type=int, default=1)
    parser.add_argument("--refine-lr", type=float, default=0.01)
    parser.add_argument("--refine-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "experiments/compare_evolution_crossover.py",
        "--dataset",
        "cifar10",
        "--cifar-image-size",
        "32",
        "--cifar-rgb",
        "--methods",
        args.methods,
        "--runs",
        str(args.runs),
        "--generations",
        str(args.generations),
        "--population",
        str(args.population),
        "--crossover-rate",
        str(args.crossover_rate),
        "--refine-steps",
        str(args.refine_steps),
        "--refine-lr",
        str(args.refine_lr),
        "--refine-batch-size",
        str(args.refine_batch_size),
        "--signed-relu-width-policy",
        "minimal",
        "--topology-mutation-rate",
        "0.1",
        "--node-split-rate",
        "0.1",
        "--layer-delete-rate",
        "0.1",
        "--node-delete-rate",
        "0.1",
        "--max-hidden-layers",
        "4",
        "--max-hidden-width",
        "128",
        "--parsimony-tolerance",
        "0.01",
        "--elitism",
        "0",
        "--r-max",
        "1000000",
        "--seed",
        str(args.seed),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.max_train_samples is not None:
        cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_test_samples is not None:
        cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    if args.no_plots:
        cmd.append("--no-plots")

    print("Running:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

