from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def read_curves(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rate_label(folder: Path) -> str:
    match = re.search(r"crossoverrate_([0-9.]+)", folder.name)
    if not match:
        return folder.name
    return f"crossover {match.group(1)}"


def averaged_curve(rows: list[dict[str, str]], method: str, metric: str) -> tuple[list[int], list[float]]:
    by_generation: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        by_generation[int(row["generation"])].append(float(row[metric]))
    generations = sorted(by_generation)
    values = [mean(by_generation[generation]) for generation in generations]
    return generations, values


def plot_metric(series: list[tuple[str, list[int], list[float]]], metric: str, title: str, ylabel: str, output: Path) -> None:
    plt.figure(figsize=(10, 6))
    for label, generations, values in series:
        plt.plot(generations, values, linewidth=1.1, label=label)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot combined SVD crossover-rate curves.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--pattern", default="SVD_crossover_100x300_crossoverrate_*")
    parser.add_argument("--output-dir", type=Path, default=Path("results/by_rates"))
    args = parser.parse_args()

    folders = sorted(args.results_root.glob(args.pattern), key=lambda p: float(re.search(r"crossoverrate_([0-9.]+)", p.name).group(1)))
    if not folders:
        raise SystemExit(f"No result folders matched {args.results_root / args.pattern}")

    loss_series: list[tuple[str, list[int], list[float]]] = []
    accuracy_series: list[tuple[str, list[int], list[float]]] = []
    no_crossover_added = False

    for folder in folders:
        curves_path = folder / "curves.csv"
        if not curves_path.exists():
            continue
        rows = read_curves(curves_path)

        if not no_crossover_added:
            generations, values = averaged_curve(rows, "no_crossover", "best_loss")
            if generations:
                loss_series.append(("no crossover", generations, values))
            generations, values = averaged_curve(rows, "no_crossover", "best_accuracy")
            if generations:
                accuracy_series.append(("no crossover", generations, values))
            no_crossover_added = True

        label = rate_label(folder)
        generations, values = averaged_curve(rows, "svd_crossover_same_dim", "best_loss")
        if generations:
            loss_series.append((label, generations, values))
        generations, values = averaged_curve(rows, "svd_crossover_same_dim", "best_accuracy")
        if generations:
            accuracy_series.append((label, generations, values))

    plot_metric(
        loss_series,
        "best_loss",
        "Mean best loss by crossover rate",
        "Mean best loss",
        args.output_dir / "SVD_crossover_100x300_all_rates_loss.png",
    )
    plot_metric(
        accuracy_series,
        "best_accuracy",
        "Mean best accuracy by crossover rate",
        "Mean best accuracy",
        args.output_dir / "SVD_crossover_100x300_all_rates_accuracy.png",
    )

    print("Wrote:")
    print(args.output_dir / "SVD_crossover_100x300_all_rates_loss.png")
    print(args.output_dir / "SVD_crossover_100x300_all_rates_accuracy.png")


if __name__ == "__main__":
    main()
