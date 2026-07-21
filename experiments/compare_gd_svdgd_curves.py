"""Average GD vs SVD-GD curves for XOR, digits, and Sign-MNIST.

Outputs two plots per problem:
  1. mean performance vs optimization step
  2. mean performance vs wall-clock time

Evolution is intentionally not used here. This script compares only the dense
GD baseline and the spectral/SVD-GD baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_ga.benchmarks import (  # noqa: E402
    load_digits_dataset,
    load_cats_dogs_dataset,
    load_sign_mnist_dataset,
    xor_dataset,
)
from spectral_ga.network import NetworkConfig, SpectralNetwork  # noqa: E402
from spectral_ga.spectral import optimize_spectral_parameters  # noqa: E402
from spectral_ga.utils import seed_all, sync_cuda  # noqa: E402

METHODS = ("gd", "svd_gd")
PROBLEMS = ("xor", "digits", "sign_mnist", "cats_dogs")


def parse_rank_fraction(raw: str | None) -> float | None:
    if raw is None or raw.strip().lower() in {"", "none", "off"}:
        return None
    value = raw.strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        fraction = float(numerator) / float(denominator)
    else:
        fraction = float(value)
    if fraction <= 0.0:
        raise argparse.ArgumentTypeError("rank fraction must be positive")
    return fraction


def rank_description(
    layer_dims: list[int],
    rank_fraction: float | None,
    r_max: int,
    full_below: int = 5,
) -> str:
    ranks = []
    for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
        full_rank = min(fan_in, fan_out)
        if rank_fraction is None:
            rank = min(r_max, full_rank)
        elif full_rank <= full_below:
            rank = full_rank
        else:
            rank = max(1, min(full_rank, int(np.ceil(rank_fraction * full_rank))))
        ranks.append(f"{rank}/{full_rank}")
    return ",".join(ranks)


def softmax_loss_and_accuracy(logits: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    loss = float(nn.functional.cross_entropy(logits, y).item())
    pred = torch.argmax(logits, dim=1)
    acc = float((pred == y).float().mean().item())
    return loss, acc


def bce_loss_and_accuracy(probabilities: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    loss = float(nn.functional.binary_cross_entropy(probabilities, y).item())
    pred = (probabilities >= 0.5).float()
    acc = float((pred == y).float().mean().item())
    return loss, acc


def load_problem(problem: str):
    if problem == "xor":
        x_train, y_train = xor_dataset()
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_train,
            "y_test": y_train,
            "layer_dims": [2, 8, 1],
            "loss": "bce",
        }
    if problem == "digits":
        x_train, y_train, x_test, y_test = load_digits_dataset()
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "layer_dims": [64, 32, 10],
            "loss": "ce",
        }
    if problem == "sign_mnist":
        x_train, y_train, x_test, y_test = load_sign_mnist_dataset()
        num_classes = int(max(y_train.max(), y_test.max()) + 1)
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "layer_dims": [784, 512, 256, 128, num_classes],
            "loss": "ce",
        }
    if problem == "cats_dogs":
        x_train, y_train, x_test, y_test = load_cats_dogs_dataset(
            image_size=load_problem.image_size,
            max_samples=load_problem.max_samples,
            seed=load_problem.seed,
        )
        input_dim = load_problem.image_size * load_problem.image_size
        return {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "layer_dims": [input_dim, 256, 64, 16, 1],
            "loss": "bce",
        }
    raise ValueError(f"Unknown problem: {problem}")


def make_dense_model(layer_dims: list[int], loss: str) -> nn.Module:
    modules: list[nn.Module] = []
    for index, (fan_in, fan_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
        modules.append(nn.Linear(fan_in, fan_out))
        if index < len(layer_dims) - 2:
            modules.append(nn.ReLU())
    if loss == "bce":
        modules.append(nn.Sigmoid())
    return nn.Sequential(*modules)


def evaluate_dense(model: nn.Module, x: np.ndarray, y: np.ndarray, loss: str, device: torch.device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        if loss == "ce":
            y_t = torch.tensor(y.reshape(-1), dtype=torch.long, device=device)
            return softmax_loss_and_accuracy(model(x_t), y_t)
        y_t = torch.tensor(y.astype(float), dtype=torch.float32, device=device)
        return bce_loss_and_accuracy(model(x_t), y_t)


def run_gd_curve(problem: str, data: dict, args: argparse.Namespace, seed: int) -> list[dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_dense_model(data["layer_dims"], data["loss"]).to(device)
    loss_fn = nn.CrossEntropyLoss() if data["loss"] == "ce" else nn.BCELoss()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.gd_lr)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.gd_lr)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    x_train = torch.tensor(data["x_train"], dtype=torch.float32)
    y_dtype = torch.long if data["loss"] == "ce" else torch.float32
    y_train = torch.tensor(data["y_train"].reshape(-1) if data["loss"] == "ce" else data["y_train"], dtype=y_dtype)
    dataset = TensorDataset(x_train, y_train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(args.gd_batch_size, len(dataset))),
        shuffle=True,
        generator=generator,
    )

    rows = []
    start = time.perf_counter()
    test_loss, test_accuracy = evaluate_dense(model, data["x_test"], data["y_test"], data["loss"], device)
    rows.append(curve_row(problem, "gd", seed, 0, 0.0, test_loss, test_accuracy))

    for step in range(1, args.steps + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            prediction = model(xb)
            loss_value = loss_fn(prediction, yb)
            loss_value.backward()
            optimizer.step()
        sync_cuda()
        test_loss, test_accuracy = evaluate_dense(model, data["x_test"], data["y_test"], data["loss"], device)
        rows.append(curve_row(problem, "gd", seed, step, time.perf_counter() - start, test_loss, test_accuracy))
    return rows


def run_svd_gd_curve(problem: str, data: dict, args: argparse.Namespace, seed: int) -> list[dict]:
    seed_all(seed)
    network = SpectralNetwork.from_architecture(
        NetworkConfig(
            layer_dims=data["layer_dims"],
            r_max=args.r_max,
            seed=seed,
            rank_fraction=parse_rank_fraction(args.rank_fraction),
            rank_fraction_full_below=args.rank_fraction_full_below,
        )
    )
    rows = []
    start = time.perf_counter()
    rows.append(
        curve_row(
            problem,
            "svd_gd",
            seed,
            0,
            0.0,
            network.evaluate(data["x_test"], data["y_test"], loss=data["loss"]),
            network.accuracy(data["x_test"], data["y_test"]),
        )
    )

    def record(step: int, net: SpectralNetwork) -> None:
        rows.append(
            curve_row(
                problem,
                "svd_gd",
                seed,
                step,
                time.perf_counter() - start,
                net.evaluate(data["x_test"], data["y_test"], loss=data["loss"]),
                net.accuracy(data["x_test"], data["y_test"]),
            )
        )

    optimize_spectral_parameters(
        network,
        data["x_train"],
        data["y_train"],
        loss=data["loss"],
        steps=args.steps,
        lr=args.svd_gd_lr,
        alpha_lr=args.svd_gd_alpha_lr,
        uv_lr=args.svd_gd_uv_lr,
        bias_lr=args.svd_gd_bias_lr,
        optimize_bias=args.optimize_bias,
        optimize_uv=True,
        seed=seed,
        progress_interval=1,
        progress_fn=record,
        batch_size=args.svd_gd_batch_size,
        optimizer_name=args.optimizer,
    )
    return rows


def curve_row(problem: str, method: str, seed: int, step: int, wall: float, loss: float, accuracy: float) -> dict:
    return {
        "problem": problem,
        "method": method,
        "seed": seed,
        "step": step,
        "wall_seconds": wall,
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }


def average_curves(rows: Iterable[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["problem"], row["method"], int(row["step"]))].append(row)

    averaged = []
    for (problem, method, step), values in sorted(grouped.items()):
        losses = [float(value["test_loss"]) for value in values]
        accuracies = [float(value["test_accuracy"]) for value in values]
        times = [float(value["wall_seconds"]) for value in values]
        averaged.append(
            {
                "problem": problem,
                "method": method,
                "step": step,
                "runs": len(values),
                "mean_test_loss": mean(losses),
                "std_test_loss": stdev(losses) if len(losses) > 1 else 0.0,
                "mean_test_accuracy": mean(accuracies),
                "std_test_accuracy": stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "mean_wall_seconds": mean(times),
            }
        )
    return averaged


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completed_runs(rows: list[dict], final_step: int) -> set[tuple[str, str, int]]:
    complete: set[tuple[str, str, int]] = set()
    for row in rows:
        if int(float(row["step"])) == final_step:
            complete.add((row["problem"], row["method"], int(float(row["seed"]))))
    return complete


def plot_problem(problem: str, averaged: list[dict], output_dir: Path, metric: str) -> list[str]:
    problem_rows = [row for row in averaged if row["problem"] == problem]
    if metric == "accuracy":
        value_key = "mean_test_accuracy"
        ylabel = "Mean test accuracy"
    else:
        value_key = "mean_test_loss"
        ylabel = "Mean KL / cross-entropy error"
    paths = []

    plt.figure(figsize=(8, 5))
    for method in METHODS:
        rows = sorted([row for row in problem_rows if row["method"] == method], key=lambda row: row["step"])
        plt.plot([row["step"] for row in rows], [row[value_key] for row in rows], label=method, linewidth=1.8)
    plt.xlabel("Optimization step")
    plt.ylabel(ylabel)
    plt.title(f"{problem}: GD vs SVD-GD by step")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    step_path = output_dir / f"{problem}_by_step.png"
    plt.savefig(step_path, dpi=180)
    plt.close()
    paths.append(str(step_path))

    plt.figure(figsize=(8, 5))
    for method in METHODS:
        rows = sorted([row for row in problem_rows if row["method"] == method], key=lambda row: row["mean_wall_seconds"])
        plt.plot([row["mean_wall_seconds"] for row in rows], [row[value_key] for row in rows], label=method, linewidth=1.8)
    plt.xlabel("Mean wall-clock seconds")
    plt.ylabel(ylabel)
    plt.title(f"{problem}: GD vs SVD-GD by time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    time_path = output_dir / f"{problem}_by_time.png"
    plt.savefig(time_path, dpi=180)
    plt.close()
    paths.append(str(time_path))
    return paths


def final_rows(averaged: list[dict]) -> list[dict]:
    output = []
    for problem in PROBLEMS:
        problem_rows = [row for row in averaged if row["problem"] == problem]
        if not problem_rows:
            continue
        final_step = max(int(row["step"]) for row in problem_rows)
        output.extend(row for row in problem_rows if int(row["step"]) == final_step)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Average GD vs SVD-GD curves without evolution.")
    parser.add_argument("--problems", default=",".join(PROBLEMS), help="Comma-separated: xor,digits,sign_mnist")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=200, help="GD epochs and SVD-GD steps per run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/gd_vs_svdgd_curves")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from raw_curves.csv when present")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Ignore previous raw_curves.csv")
    parser.add_argument(
        "--metric",
        choices=["kl", "loss", "accuracy"],
        default="kl",
        help="Plot KL/cross-entropy error by default; 'loss' is an alias for 'kl'.",
    )
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Use the same optimizer family for dense GD and SVD-GD")
    parser.add_argument("--gd-lr", type=float, default=0.005)
    parser.add_argument("--gd-batch-size", type=int, default=256)
    parser.add_argument("--svd-gd-lr", type=float, default=0.005)
    parser.add_argument("--svd-gd-alpha-lr", type=float, default=None)
    parser.add_argument("--svd-gd-uv-lr", type=float, default=None)
    parser.add_argument("--svd-gd-bias-lr", type=float, default=None)
    parser.add_argument("--svd-gd-batch-size", type=int, default=256)
    parser.add_argument("--r-max", type=int, default=4, help="Absolute rank cap when --rank-fraction is not set")
    parser.add_argument("--rank-fraction", default=None, help="Per-layer rank as a fraction of min(m,n), e.g. 1/2")
    parser.add_argument("--rank-fraction-full-below", type=int, default=5, help="Use full rank when min(m,n) is this value or smaller")
    parser.add_argument("--image-size", type=int, default=32, help="Cats-dogs crop size, image_size x image_size grayscale")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for image experiments")
    parser.add_argument("--optimize-bias", action="store_true", default=True)
    parser.add_argument("--no-optimize-bias", dest="optimize_bias", action="store_false")
    args = parser.parse_args()

    problems = [problem.strip() for problem in args.problems.split(",") if problem.strip()]
    unknown = sorted(set(problems) - set(PROBLEMS))
    if unknown:
        raise ValueError(f"Unknown problems: {', '.join(unknown)}")

    load_problem.image_size = args.image_size
    load_problem.max_samples = args.max_samples
    load_problem.seed = args.seed

    if args.metric == "loss":
        args.metric = "kl"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "raw_curves.csv"
    raw_rows = read_csv(raw_path) if args.resume else []
    completed = completed_runs(raw_rows, args.steps)
    total = len(problems) * len(METHODS) * args.runs
    done = len(completed)
    if completed:
        print(f"Resuming from {raw_path}: {done}/{total} runs already complete", flush=True)

    for problem in problems:
        data = load_problem(problem)
        for method in METHODS:
            for run_index in range(args.runs):
                seed = args.seed + run_index
                run_key = (problem, method, seed)
                if run_key in completed:
                    continue
                if method == "gd":
                    new_rows = run_gd_curve(problem, data, args, seed)
                else:
                    new_rows = run_svd_gd_curve(problem, data, args, seed)
                for row in new_rows:
                    if method == "svd_gd":
                        row["rank_fraction"] = args.rank_fraction or "r_max"
                        row["layer_ranks"] = rank_description(
                            data["layer_dims"],
                            parse_rank_fraction(args.rank_fraction),
                            args.r_max,
                            args.rank_fraction_full_below,
                        )
                    else:
                        row["rank_fraction"] = "dense"
                        row["layer_ranks"] = "full"
                raw_rows.extend(new_rows)
                completed.add(run_key)
                done += 1
                write_csv(raw_path, raw_rows)
                print(f"[{done}/{total}] {problem} {method} seed={seed}", flush=True)

    averaged = average_curves(raw_rows)
    write_csv(raw_path, raw_rows)
    write_csv(output_dir / "average_curves.csv", averaged)

    plots = []
    for problem in problems:
        plots.extend(plot_problem(problem, averaged, output_dir, args.metric))

    summary = {
        "config": vars(args),
        "problems": problems,
        "methods": METHODS,
        "plots": plots,
        "final": final_rows(averaged),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFinal averaged test results")
    print("problem      method   loss      accuracy  mean_time")
    print("-----------  -------  --------  --------  ---------")
    for row in summary["final"]:
        print(
            f"{row['problem']:<11}  {row['method']:<7}  "
            f"{row['mean_test_loss']:.4f}    {row['mean_test_accuracy']:.4f}    "
            f"{row['mean_wall_seconds']:.3f}s"
        )
    print(f"\nSaved plots and tables to {output_dir}")


if __name__ == "__main__":
    main()

