from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spectral_ga.benchmarks import load_cifar10_dataset, topology_signature
from spectral_ga.crossover import CrossoverConfig, anchor_network_mode_crossover
from spectral_ga.evolution import dense_parameter_count
from spectral_ga.innovation import InnovationRegistry
from spectral_ga.network import NetworkConfig, SpectralNetwork
from spectral_ga.spectral import optimize_dense_parameters


@dataclass(frozen=True)
class RunConfig:
    population: int
    generations: int
    hidden_widths: tuple[int, ...]
    r_max: int
    gd_steps: int
    gd_lr: float
    batch_size: int
    crossover_rate: float
    mode_mutation_rate: float
    alpha_mutation_scale: float
    direction_mutation_scale: float
    bias_mutation_scale: float
    tournament_size: int
    elite_size: int
    seed: int


def parse_hidden_widths(value: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("hidden widths must be comma-separated integers") from exc
    if not widths or any(width <= 0 for width in widths):
        raise argparse.ArgumentTypeError("at least one positive hidden width is required")
    return widths


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def standardize_from_train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    mean = float(np.mean(x_train))
    std = max(float(np.std(x_train)), 1e-6)
    stats = {"mean": mean, "std": std}
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
        stats,
    )


def split_validation(
    x: np.ndarray,
    y: np.ndarray,
    validation_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if validation_size <= 0 or validation_size >= len(x):
        raise ValueError("validation size must be between 1 and train_size - 1")
    order = rng.permutation(len(x))
    val_indices = order[:validation_size]
    train_indices = order[validation_size:]
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]


def mutate_spectral_modes(
    network: SpectralNetwork,
    rng: np.random.Generator,
    mode_rate: float,
    alpha_scale: float,
    direction_scale: float,
    bias_scale: float,
) -> int:
    """Mutate complete rank-one packages while keeping U/V columns normalized."""
    mutated_modes = 0
    for layer in network.layers:
        for mode_index in range(layer.rank):
            if rng.random() >= mode_rate:
                continue
            mutated_modes += 1

            alpha_reference = max(abs(float(layer.alpha[mode_index])), 1e-3)
            layer.alpha[mode_index] += rng.normal(0.0, alpha_scale * alpha_reference)

            if direction_scale > 0.0:
                layer.u[:, mode_index] += rng.normal(0.0, direction_scale, layer.u.shape[0])
                layer.v[:, mode_index] += rng.normal(0.0, direction_scale, layer.v.shape[0])
                u_norm = float(np.linalg.norm(layer.u[:, mode_index]))
                v_norm = float(np.linalg.norm(layer.v[:, mode_index]))
                if u_norm > 0.0 and v_norm > 0.0:
                    layer.u[:, mode_index] /= u_norm
                    layer.v[:, mode_index] /= v_norm
                    layer.alpha[mode_index] *= u_norm * v_norm

    if bias_scale > 0.0:
        for bias in network.biases:
            mask = rng.random(bias.shape) < mode_rate
            bias[mask] += rng.normal(0.0, bias_scale, int(np.sum(mask)))
    return mutated_modes


def tournament_index(
    losses: list[float],
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    candidates = rng.integers(0, len(losses), size=max(1, tournament_size))
    return min((int(index) for index in candidates), key=lambda index: losses[index])


def make_population(
    input_dim: int,
    output_dim: int,
    config: RunConfig,
) -> list[SpectralNetwork]:
    registry = InnovationRegistry()
    dimensions = [input_dim, *config.hidden_widths, output_dim]
    return [
        SpectralNetwork.from_architecture(
            NetworkConfig(
                layer_dims=dimensions,
                r_max=config.r_max,
                seed=config.seed * 10_000 + index,
                innovation_registry=registry,
            )
        )
        for index in range(config.population)
    ]


def evolve(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: RunConfig,
) -> tuple[SpectralNetwork, list[dict], dict[str, int]]:
    rng = np.random.default_rng(config.seed)
    population = make_population(x_train.shape[1], 10, config)
    crossover_config = CrossoverConfig(method="half_rank", allow_shape_mismatch=False)
    history: list[dict] = []
    best_network = population[0].copy()
    best_loss = float("inf")
    crossover_attempts = 0
    mutated_modes = 0

    for generation in range(1, config.generations + 1):
        generation_start = time.perf_counter()
        for index, individual in enumerate(population):
            optimize_dense_parameters(
                individual,
                x_train,
                y_train,
                loss="ce",
                steps=config.gd_steps,
                lr=config.gd_lr,
                optimize_bias=True,
                seed=config.seed * 1_000_000 + generation * 1_000 + index,
                batch_size=config.batch_size,
                optimizer_name="adam",
            )

        losses = [float(individual.evaluate(x_val, y_val, loss="ce")) for individual in population]
        ranking = sorted(range(len(population)), key=lambda index: losses[index])
        generation_best = population[ranking[0]]
        generation_loss = losses[ranking[0]]
        generation_accuracy = float(generation_best.accuracy(x_val, y_val))
        if generation_loss < best_loss:
            best_loss = generation_loss
            best_network = generation_best.copy()

        row = {
            "generation": generation,
            "validation_loss": generation_loss,
            "validation_accuracy": generation_accuracy,
            "best_loss_so_far": best_loss,
            "best_accuracy_so_far": float(best_network.accuracy(x_val, y_val)),
            "parameters": dense_parameter_count(generation_best),
            "topology": topology_signature(generation_best),
            "seconds": time.perf_counter() - generation_start,
        }
        history.append(row)
        print(
            f"gen {generation}/{config.generations}: "
            f"val_acc={generation_accuracy:.4f} val_loss={generation_loss:.4f} "
            f"best_acc={row['best_accuracy_so_far']:.4f} params={row['parameters']}"
        )

        if generation == config.generations:
            break

        elite_count = min(config.elite_size, config.population)
        offspring = [population[index].copy() for index in ranking[:elite_count]]
        while len(offspring) < config.population:
            first_index = tournament_index(losses, config.tournament_size, rng)
            child = population[first_index].copy()
            if rng.random() < config.crossover_rate:
                second_index = tournament_index(losses, config.tournament_size, rng)
                child = anchor_network_mode_crossover(
                    population[first_index],
                    population[second_index],
                    crossover_config,
                    rng,
                )
                crossover_attempts += 1

            mutation_rng = np.random.default_rng(rng.integers(0, 2**63 - 1))
            mutated_modes += mutate_spectral_modes(
                child,
                mutation_rng,
                config.mode_mutation_rate,
                config.alpha_mutation_scale,
                config.direction_mutation_scale,
                config.bias_mutation_scale,
            )
            offspring.append(child)
        population = offspring

    return best_network, history, {
        "crossover_attempts": crossover_attempts,
        "mutated_modes": mutated_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 dense classifier with GD, selection, SVD-mode crossover, and spectral mutation."
    )
    parser.add_argument("--cifar-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_svd_crossover"))
    parser.add_argument("--image-size", type=int, default=16, help="CPU default; use 32 for native CIFAR resolution")
    parser.add_argument("--max-train-samples", type=int, default=12000, help="Use 0 for all 50,000 training images")
    parser.add_argument("--max-test-samples", type=int, default=2000, help="Use 0 for all 10,000 test images")
    parser.add_argument("--validation-size", type=int, default=2000)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--hidden-widths", type=parse_hidden_widths, default=(128, 64))
    parser.add_argument("--r-max", type=int, default=32)
    parser.add_argument("--gd-steps", type=int, default=10)
    parser.add_argument("--gd-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--crossover-rate", type=float, default=0.7)
    parser.add_argument("--mode-mutation-rate", type=float, default=0.10)
    parser.add_argument("--alpha-mutation-scale", type=float, default=0.05)
    parser.add_argument("--direction-mutation-scale", type=float, default=0.01)
    parser.add_argument("--bias-mutation-scale", type=float, default=0.01)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.population < 2:
        parser.error("--population must be at least 2")
    if args.generations < 1 or args.gd_steps < 1:
        parser.error("--generations and --gd-steps must be positive")
    for name in ("crossover_rate", "mode_mutation_rate"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")

    max_train = args.max_train_samples or None
    max_test = args.max_test_samples or None
    x_all, y_all, x_test, y_test = load_cifar10_dataset(
        dataset_dir=args.cifar_dir,
        image_size=args.image_size,
        grayscale=False,
        max_train_samples=max_train,
        max_test_samples=max_test,
        seed=args.seed,
    )
    split_rng = np.random.default_rng(args.seed)
    x_train, y_train, x_val, y_val = split_validation(
        x_all, y_all, args.validation_size, split_rng
    )
    x_train, x_val, x_test, normalization = standardize_from_train(x_train, x_val, x_test)

    config = RunConfig(
        population=args.population,
        generations=args.generations,
        hidden_widths=args.hidden_widths,
        r_max=args.r_max,
        gd_steps=args.gd_steps,
        gd_lr=args.gd_lr,
        batch_size=args.batch_size,
        crossover_rate=args.crossover_rate,
        mode_mutation_rate=args.mode_mutation_rate,
        alpha_mutation_scale=args.alpha_mutation_scale,
        direction_mutation_scale=args.direction_mutation_scale,
        bias_mutation_scale=args.bias_mutation_scale,
        tournament_size=args.tournament_size,
        elite_size=args.elite_size,
        seed=args.seed,
    )

    print(
        f"CIFAR-10 SVD crossover: train={len(x_train)} val={len(x_val)} test={len(x_test)} "
        f"input={x_train.shape[1]} population={config.population} generations={config.generations}"
    )
    start = time.perf_counter()
    best_network, history, counters = evolve(x_train, y_train, x_val, y_val, config)
    elapsed = time.perf_counter() - start

    summary = {
        "validation_loss": float(best_network.evaluate(x_val, y_val, loss="ce")),
        "validation_accuracy": float(best_network.accuracy(x_val, y_val)),
        "test_loss": float(best_network.evaluate(x_test, y_test, loss="ce")),
        "test_accuracy": float(best_network.accuracy(x_test, y_test)),
        "parameters": dense_parameter_count(best_network),
        "topology": topology_signature(best_network),
        "elapsed_seconds": elapsed,
        **counters,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "curves.csv", history)
    with (args.output_dir / "best_network.json").open("w", encoding="utf-8") as handle:
        json.dump(best_network.to_dict(), handle)
    with (args.output_dir / "run.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "data": {
                    "image_size": args.image_size,
                    "train_samples": len(x_train),
                    "validation_samples": len(x_val),
                    "test_samples": len(x_test),
                    "normalization": normalization,
                },
                "summary": summary,
            },
            handle,
            indent=2,
        )

    print(
        f"done: test_acc={summary['test_accuracy']:.4f} test_loss={summary['test_loss']:.4f} "
        f"elapsed={elapsed:.1f}s wrote={args.output_dir}"
    )


if __name__ == "__main__":
    main()


