from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from spectral_ga.benchmarks import (
    load_cifar10_dataset,
    load_digits_dataset,
    refine_lamarckian_child,
    topology_signature,
)
from spectral_ga.evolution import EvolutionConfig, dense_parameter_count, run_evolution
from spectral_ga.innovation import InnovationRegistry
from spectral_ga.network import NetworkConfig, SpectralNetwork


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def run_one_digits(seed: int, crossover_rate: float, crossover_rate_final: float | None, allow_shape_mismatch_crossover: bool, args: argparse.Namespace) -> dict:
    if args.dataset == "digits":
        x_train, y_train, x_test, y_test = load_digits_dataset()
    elif args.dataset == "cifar10":
        x_train, y_train, x_test, y_test = load_cifar10_dataset(
            dataset_dir=args.cifar_dir,
            image_size=args.cifar_image_size,
            grayscale=not args.cifar_rgb,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    input_dim = int(x_train.shape[1])
    output_dim = int(np.max(np.concatenate([y_train, y_test])) + 1)
    registry = InnovationRegistry()
    population = [
        SpectralNetwork.from_architecture(
            NetworkConfig(
                layer_dims=[input_dim, output_dim],
                r_max=args.r_max,
                seed=seed * 10000 + i,
                innovation_registry=registry,
            )
        )
        for i in range(args.population)
    ]
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        r_max=args.r_max,
        elite_size=args.elitism,
        crossover_rate=crossover_rate,
        crossover_rate_final=crossover_rate_final,
        allow_shape_mismatch_crossover=allow_shape_mismatch_crossover,
        topology_mutation_rate=args.topology_mutation_rate,
        node_split_rate=args.node_split_rate,
        layer_delete_rate=args.layer_delete_rate,
        node_delete_rate=args.node_delete_rate,
        layer_delete_ridge_lambda=args.layer_delete_ridge_lambda,
        max_hidden_layers=args.max_hidden_layers,
        max_hidden_width=args.max_hidden_width,
        convolution_image_shape=None,
        signed_relu_width_policy=args.signed_relu_width_policy,
        refine_steps=args.refine_steps,
        refine_lr=args.refine_lr,
        refine_method="dense-gd",
        refine_optimizer="sgd",
        refine_batch_size=args.refine_batch_size or None,
        optimize_bias=True,
        parsimony_tolerance=args.parsimony_tolerance,
        seed=seed,
    )

    generation_rows: list[dict] = []

    def evaluate(individual: SpectralNetwork) -> float:
        return individual.evaluate(x_test, y_test, loss="ce")

    def refine_child(individual: SpectralNetwork) -> None:
        refine_lamarckian_child(individual, x_train, y_train, "ce", config)

    def layer_delete_samples(rng: np.random.Generator) -> np.ndarray:
        batch_size = config.refine_batch_size or len(x_train)
        batch_size = max(1, min(int(batch_size), len(x_train)))
        if batch_size == len(x_train):
            return x_train
        indices = rng.choice(len(x_train), size=batch_size, replace=False)
        return x_train[indices]

    def progress(generation: int, best_net: SpectralNetwork, loss: float, pop: list[SpectralNetwork]) -> None:
        counts = Counter(topology_signature(ind) for ind in pop)
        best_topology = topology_signature(best_net)
        generation_rows.append(
            {
                "generation": generation,
                "best_loss": float(loss),
                "best_accuracy": float(best_net.accuracy(x_test, y_test)),
                "best_params": dense_parameter_count(best_net),
                "best_topology": best_topology,
                "best_topology_population_share": counts[best_topology] / len(pop),
                "most_common_topology": counts.most_common(1)[0][0],
                "most_common_topology_share": counts.most_common(1)[0][1] / len(pop),
                "unique_topologies": len(counts),
                "population_topologies": json.dumps(dict(counts), sort_keys=True),
            }
        )

    start = time.perf_counter()
    result = run_evolution(
        population,
        evaluate,
        config,
        refine_fn=refine_child,
        progress_fn=progress,
        layer_delete_sample_fn=layer_delete_samples,
    )
    wall = time.perf_counter() - start

    best = result.best_network
    final_loss = float(best.evaluate(x_test, y_test, loss="ce"))
    final_acc = float(best.accuracy(x_test, y_test))
    final_params = dense_parameter_count(best)
    final_topology = topology_signature(best)
    final_counts = Counter(topology_signature(ind) for ind in result.population)
    final_topology_share = final_counts[final_topology] / len(result.population)
    most_common_topology, most_common_count = final_counts.most_common(1)[0]

    first_final_topology_generation = None
    first_majority_topology_generation = None
    final_topology_share_curve: list[float] = []
    for row in generation_rows:
        counts = json.loads(row["population_topologies"])
        final_share = counts.get(final_topology, 0) / args.population
        final_topology_share_curve.append(final_share)
        if first_final_topology_generation is None and final_share > 0.0:
            first_final_topology_generation = int(row["generation"])
        if first_majority_topology_generation is None and final_share >= 0.5:
            first_majority_topology_generation = int(row["generation"])

    if first_final_topology_generation is None:
        first_final_topology_generation = args.generations + 1
    if first_majority_topology_generation is None:
        first_majority_topology_generation = args.generations + 1

    crossover_shape_match_rate = (
        result.crossover_shape_match_events / result.crossover_attempts
        if result.crossover_attempts > 0
        else 0.0
    )
    crossover_shape_mismatch_rate = (
        result.crossover_shape_mismatch_events / result.crossover_attempts
        if result.crossover_attempts > 0
        else 0.0
    )

    return {
        "seed": seed,
        "crossover_rate": crossover_rate,
        "crossover_rate_final": crossover_rate_final,
        "allow_shape_mismatch_crossover": allow_shape_mismatch_crossover,
        "crossover_attempts": result.crossover_attempts,
        "crossover_shape_match_events": result.crossover_shape_match_events,
        "crossover_shape_match_layers": result.crossover_shape_match_layers,
        "crossover_shape_match_rate": crossover_shape_match_rate,
        "crossover_shape_mismatch_events": result.crossover_shape_mismatch_events,
        "crossover_shape_mismatch_layers": result.crossover_shape_mismatch_layers,
        "crossover_shape_mismatch_rate": crossover_shape_mismatch_rate,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "final_params": final_params,
        "final_topology": final_topology,
        "final_topology_share": final_topology_share,
        "most_common_topology": most_common_topology,
        "most_common_topology_share": most_common_count / len(result.population),
        "first_final_topology_generation": first_final_topology_generation,
        "first_majority_topology_generation": first_majority_topology_generation,
        "wall_seconds": wall,
        "generation_rows": generation_rows,
        "final_topology_share_curve": final_topology_share_curve,
    }


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, curve_rows: list[dict], summary_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in curve_rows})

    for metric, ylabel, name in [
        ("best_accuracy", "Mean best accuracy", "accuracy_by_generation.png"),
        ("best_loss", "Mean best loss", "loss_by_generation.png"),
        ("best_params", "Mean best parameters", "params_by_generation.png"),
        ("final_topology_share", "Mean share of final best topology", "final_topology_share_by_generation.png"),
    ]:
        plt.figure(figsize=(8, 5))
        for method in methods:
            xs = []
            ys = []
            by_gen: dict[int, list[float]] = defaultdict(list)
            for row in curve_rows:
                if row["method"] == method:
                    by_gen[int(row["generation"])].append(float(row[metric]))
            for generation in sorted(by_gen):
                xs.append(generation)
                ys.append(float(np.mean(by_gen[generation])))
            plt.plot(xs, ys, label=method)
        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160)
        plt.close()

    labels = [row["method"] for row in summary_rows]
    for metric, ylabel, name in [
        ("mean_final_accuracy", "Mean final accuracy", "summary_accuracy.png"),
        ("mean_final_loss", "Mean final loss", "summary_loss.png"),
        ("mean_final_params", "Mean final parameters", "summary_params.png"),
        ("mean_first_final_topology_generation", "Mean first generation where final best topology appears", "summary_convergence_generation.png"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.bar(labels, [float(row[metric]) for row in summary_rows])
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evolution with no crossover, same-dim SVD crossover, and all-dim SVD crossover.")
    parser.add_argument("--dataset", choices=["digits", "cifar10"], default="digits")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--r-max", type=int, default=1000000, help="SVD rank cap per layer; default is effectively full rank")
    parser.add_argument("--topology-mutation-rate", type=float, default=0.05)
    parser.add_argument("--node-split-rate", type=float, default=0.05)
    parser.add_argument("--layer-delete-rate", type=float, default=0.05)
    parser.add_argument("--node-delete-rate", type=float, default=0.05)
    parser.add_argument("--layer-delete-ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--max-hidden-layers", type=int, default=4)
    parser.add_argument("--max-hidden-width", type=int, default=128)
    parser.add_argument("--signed-relu-width-policy", choices=["minimal", "mean"], default="mean")
    parser.add_argument("--refine-steps", type=int, default=2)
    parser.add_argument("--refine-lr", type=float, default=0.01)
    parser.add_argument("--refine-batch-size", type=int, default=128)
    parser.add_argument("--parsimony-tolerance", type=float, default=0.01)
    parser.add_argument("--elitism", type=int, default=1)
    parser.add_argument("--crossover-rate", type=float, default=0.5)
    parser.add_argument("--annealed-crossover-start", type=float, default=0.5)
    parser.add_argument("--annealed-crossover-end", type=float, default=0.0)
    parser.add_argument("--methods", choices=["three", "no_and_same_dim", "same_dim_only", "annealed_only", "all_four"], default="all_four")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/evolution_crossover_compare"))
    parser.add_argument("--cifar-dir", type=Path, default=None, help="Path to cifar-10-batches-py")
    parser.add_argument("--cifar-image-size", type=int, default=64, help="CIFAR resize dimension; default 64 gives 64x64x3 RGB inputs")
    parser.add_argument("--cifar-rgb", dest="cifar_rgb", action="store_true", default=True, help="Keep CIFAR RGB channels")
    parser.add_argument("--cifar-grayscale", dest="cifar_rgb", action="store_false", help="Convert CIFAR images to grayscale")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional training sample cap for larger datasets")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional test sample cap for larger datasets")
    parser.add_argument("--extra-curves-csv", type=Path, action="append", default=[], help="Existing curves.csv to include in plots")
    parser.add_argument("--extra-summary-csv", type=Path, action="append", default=[], help="Existing summary.csv to include in summary plots")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    base_methods = [
        ("no_crossover", 0.0, None, False),
        ("svd_crossover_same_dim", args.crossover_rate, None, False),
        ("svd_crossover_all_dims", args.crossover_rate, None, True),
    ]
    annealed_method = (
        "svd_crossover_annealed_same_dim",
        args.annealed_crossover_start,
        args.annealed_crossover_end,
        False,
    )
    if args.methods == "three":
        methods = base_methods
    elif args.methods == "no_and_same_dim":
        methods = base_methods[:2]
    elif args.methods == "same_dim_only":
        methods = [base_methods[1]]
    elif args.methods == "annealed_only":
        methods = [annealed_method]
    else:
        methods = base_methods + [annealed_method]
    raw_rows: list[dict] = []
    curve_rows: list[dict] = []
    topology_rows: list[dict] = []

    for method, crossover_rate, crossover_rate_final, allow_shape_mismatch_crossover in methods:
        print(f"Running {method} for {args.runs} runs...", flush=True)
        for run_index in range(args.runs):
            seed = args.seed + run_index
            result = run_one_digits(seed, crossover_rate, crossover_rate_final, allow_shape_mismatch_crossover, args)
            raw_rows.append(
                {
                    key: result[key]
                    for key in [
                        "seed",
                        "crossover_rate",
                        "crossover_rate_final",
                        "allow_shape_mismatch_crossover",
                        "crossover_attempts",
                        "crossover_shape_match_events",
                        "crossover_shape_match_layers",
                        "crossover_shape_match_rate",
                        "crossover_shape_mismatch_events",
                        "crossover_shape_mismatch_layers",
                        "crossover_shape_mismatch_rate",
                        "final_loss",
                        "final_accuracy",
                        "final_params",
                        "final_topology",
                        "final_topology_share",
                        "most_common_topology",
                        "most_common_topology_share",
                        "first_final_topology_generation",
                        "first_majority_topology_generation",
                        "wall_seconds",
                    ]
                }
                | {"method": method, "run": run_index}
            )
            for gen_row, final_share in zip(result["generation_rows"], result["final_topology_share_curve"]):
                curve_rows.append(
                    {
                        "method": method,
                        "run": run_index,
                        "seed": seed,
                        "generation": gen_row["generation"],
                        "best_loss": gen_row["best_loss"],
                        "best_accuracy": gen_row["best_accuracy"],
                        "best_params": gen_row["best_params"],
                        "best_topology": gen_row["best_topology"],
                        "most_common_topology": gen_row["most_common_topology"],
                        "most_common_topology_share": gen_row["most_common_topology_share"],
                        "unique_topologies": gen_row["unique_topologies"],
                        "final_topology_share": final_share,
                    }
                )
            topology_rows.append(
                {
                    "method": method,
                    "run": run_index,
                    "seed": seed,
                    "final_topology": result["final_topology"],
                    "most_common_topology": result["most_common_topology"],
                }
            )
            print(
                f"  {method} run {run_index + 1}/{args.runs}: "
                f"acc={result['final_accuracy']:.4f} loss={result['final_loss']:.4f} "
                f"params={result['final_params']} topology={result['final_topology']} "
                f"same_shape_xover={result['crossover_shape_match_events']}/{result['crossover_attempts']} "
                f"shape_mismatch_xover={result['crossover_shape_mismatch_events']}/{result['crossover_attempts']}",
                flush=True,
            )

    summary_rows: list[dict] = []
    for method, _, _, _ in methods:
        rows = [row for row in raw_rows if row["method"] == method]
        acc_mean, acc_std = mean_std([float(row["final_accuracy"]) for row in rows])
        loss_mean, loss_std = mean_std([float(row["final_loss"]) for row in rows])
        params_mean, params_std = mean_std([float(row["final_params"]) for row in rows])
        conv_mean, conv_std = mean_std([float(row["first_final_topology_generation"]) for row in rows])
        majority_conv_mean, majority_conv_std = mean_std([float(row["first_majority_topology_generation"]) for row in rows])
        share_mean, share_std = mean_std([float(row["final_topology_share"]) for row in rows])
        match_rate_mean, match_rate_std = mean_std([float(row["crossover_shape_match_rate"]) for row in rows])
        match_events_mean, match_events_std = mean_std([float(row["crossover_shape_match_events"]) for row in rows])
        match_layers_mean, match_layers_std = mean_std([float(row["crossover_shape_match_layers"]) for row in rows])
        mismatch_rate_mean, mismatch_rate_std = mean_std([float(row["crossover_shape_mismatch_rate"]) for row in rows])
        mismatch_events_mean, mismatch_events_std = mean_std([float(row["crossover_shape_mismatch_events"]) for row in rows])
        mismatch_layers_mean, mismatch_layers_std = mean_std([float(row["crossover_shape_mismatch_layers"]) for row in rows])
        topologies = Counter(str(row["final_topology"]) for row in rows)
        summary_rows.append(
            {
                "method": method,
                "runs": len(rows),
                "mean_final_accuracy": acc_mean,
                "std_final_accuracy": acc_std,
                "mean_final_loss": loss_mean,
                "std_final_loss": loss_std,
                "mean_final_params": params_mean,
                "std_final_params": params_std,
                "mean_first_final_topology_generation": conv_mean,
                "std_first_final_topology_generation": conv_std,
                "mean_first_majority_topology_generation": majority_conv_mean,
                "std_first_majority_topology_generation": majority_conv_std,
                "mean_final_topology_share": share_mean,
                "std_final_topology_share": share_std,
                "mean_crossover_shape_match_rate": match_rate_mean,
                "std_crossover_shape_match_rate": match_rate_std,
                "mean_crossover_shape_match_events": match_events_mean,
                "std_crossover_shape_match_events": match_events_std,
                "mean_crossover_shape_match_layers": match_layers_mean,
                "std_crossover_shape_match_layers": match_layers_std,
                "mean_crossover_shape_mismatch_rate": mismatch_rate_mean,
                "std_crossover_shape_mismatch_rate": mismatch_rate_std,
                "mean_crossover_shape_mismatch_events": mismatch_events_mean,
                "std_crossover_shape_mismatch_events": mismatch_events_std,
                "mean_crossover_shape_mismatch_layers": mismatch_layers_mean,
                "std_crossover_shape_mismatch_layers": mismatch_layers_std,
                "most_common_final_topology": topologies.most_common(1)[0][0],
                "most_common_final_topology_count": topologies.most_common(1)[0][1],
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "raw_runs.csv",
        raw_rows,
        [
            "method",
            "run",
            "seed",
            "crossover_rate",
            "crossover_rate_final",
            "allow_shape_mismatch_crossover",
            "crossover_attempts",
            "crossover_shape_match_events",
            "crossover_shape_match_layers",
            "crossover_shape_match_rate",
            "crossover_shape_mismatch_events",
            "crossover_shape_mismatch_layers",
            "crossover_shape_mismatch_rate",
            "final_loss",
            "final_accuracy",
            "final_params",
            "final_topology",
            "final_topology_share",
            "most_common_topology",
            "most_common_topology_share",
            "first_final_topology_generation",
            "first_majority_topology_generation",
            "wall_seconds",
        ],
    )
    write_csv(
        output_dir / "curves.csv",
        curve_rows,
        [
            "method",
            "run",
            "seed",
            "generation",
            "best_loss",
            "best_accuracy",
            "best_params",
            "best_topology",
            "most_common_topology",
            "most_common_topology_share",
            "unique_topologies",
            "final_topology_share",
        ],
    )
    write_csv(output_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    if not args.no_plots:
        plot_curve_rows: list[dict] = []
        for path in args.extra_curves_csv:
            plot_curve_rows.extend(read_csv_rows(path))
        plot_curve_rows.extend(curve_rows)
        plot_summary_rows: list[dict] = []
        for path in args.extra_summary_csv:
            plot_summary_rows.extend(read_csv_rows(path))
        plot_summary_rows.extend(summary_rows)
        make_plots(output_dir, plot_curve_rows, plot_summary_rows)

    print("\nSummary")
    for row in summary_rows:
        print(
            f"{row['method']}: "
            f"acc={row['mean_final_accuracy']:.4f}±{row['std_final_accuracy']:.4f}, "
            f"loss={row['mean_final_loss']:.4f}±{row['std_final_loss']:.4f}, "
            f"params={row['mean_final_params']:.1f}±{row['std_final_params']:.1f}, "
            f"first_final_topology_gen={row['mean_first_final_topology_generation']:.1f}±{row['std_first_final_topology_generation']:.1f}, "
            f"final_topology_share={row['mean_final_topology_share']:.3f}±{row['std_final_topology_share']:.3f}, "
            f"same_shape_xover_rate={row['mean_crossover_shape_match_rate']:.3f}±{row['std_crossover_shape_match_rate']:.3f}, "
            f"shape_mismatch_xover_rate={row['mean_crossover_shape_mismatch_rate']:.3f}±{row['std_crossover_shape_mismatch_rate']:.3f}, "
            f"common_topology={row['most_common_final_topology']} ({row['most_common_final_topology_count']}/{row['runs']})"
        )
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()




















