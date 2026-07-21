from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .selection import tournament_selection
from .crossover import CrossoverConfig, anchor_network_mode_crossover, plan_network_crossover
from .utils import seed_all
from .network import SpectralNetwork


logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 30
    r_max: int = 4
    tournament_size: int = 2
    elite_size: int = 1
    crossover_rate: float = 0.5
    crossover_rate_final: float | None = None
    allow_shape_mismatch_crossover: bool = False
    topology_mutation_rate: float = 0.05
    node_split_rate: float = 0.05
    layer_delete_rate: float = 0.05
    node_delete_rate: float = 0.05
    layer_delete_ridge_lambda: float = 1e-3
    max_hidden_layers: int = 3
    max_hidden_width: int = 64
    convolution_image_shape: tuple[int, int] | None = None
    convolution_kernel_size: int = 3
    signed_relu_width_policy: str = "minimal"
    refine_steps: int = 1
    refine_lr: float = 0.01
    refine_method: str = "dense-gd"
    refine_optimizer: str = "sgd"
    refine_batch_size: int | None = 256
    optimize_bias: bool = True
    parsimony_tolerance: float = 0.01
    seed: int = 0


@dataclass
class EvolutionResult:
    population: List[SpectralNetwork]
    fitness_history: List[float]
    best_network: SpectralNetwork
    crossover_attempts: int = 0
    crossover_shape_match_events: int = 0
    crossover_shape_match_layers: int = 0
    crossover_shape_mismatch_events: int = 0
    crossover_shape_mismatch_layers: int = 0


def dense_parameter_count(network: SpectralNetwork) -> int:
    return int(sum(layer.shape[0] * layer.shape[1] + layer.shape[0] for layer in network.layers))


def loss_size_key(network: SpectralNetwork, loss: float, tolerance: float) -> tuple[float, int]:
    if tolerance <= 0.0:
        bucket = float(loss)
    else:
        bucket = round(float(loss) / tolerance) * tolerance
    return (bucket, dense_parameter_count(network))


def rank_indices_by_loss_then_size(
    population: List[SpectralNetwork],
    losses: List[float],
    tolerance: float,
) -> list[int]:
    return sorted(
        range(len(population)),
        key=lambda index: loss_size_key(population[index], losses[index], tolerance),
    )


def run_evolution(
    initial_population: List[SpectralNetwork],
    evaluate_fn: Callable[[SpectralNetwork], float],
    config: EvolutionConfig,
    refine_fn: Callable[[SpectralNetwork], None] | None = None,
    progress_fn: Callable[[int, SpectralNetwork, float, List[SpectralNetwork]], None] | None = None,
    layer_delete_sample_fn: Callable[[np.random.Generator], np.ndarray] | None = None,
) -> EvolutionResult:
    rng = seed_all(config.seed)
    population = [ind.copy() for ind in initial_population]
    fitness_history: list[float] = []
    best_network = population[0].copy()
    best_fitness_so_far = float("inf")
    best_key_so_far: tuple[float, int] = (float("inf"), 2**63 - 1)
    crossover_attempts = 0
    crossover_shape_match_events = 0
    crossover_shape_match_layers = 0
    crossover_shape_mismatch_events = 0
    crossover_shape_mismatch_layers = 0

    for generation in range(1, config.generations + 1):
        if refine_fn is not None:
            for individual in population:
                refine_fn(individual)

        fitnesses = [evaluate_fn(ind) for ind in population]
        ranked_indices = rank_indices_by_loss_then_size(population, fitnesses, config.parsimony_tolerance)
        selection_ranks = [0] * len(population)
        for rank, index in enumerate(ranked_indices):
            selection_ranks[index] = rank
        best_idx = int(ranked_indices[0])
        best_fitness = float(fitnesses[best_idx])
        best_key = loss_size_key(population[best_idx], best_fitness, config.parsimony_tolerance)
        if best_key < best_key_so_far:
            best_key_so_far = best_key
            best_fitness_so_far = best_fitness
            best_network = population[best_idx].copy()
        fitness_history.append(best_fitness_so_far)
        if progress_fn is not None:
            progress_fn(generation, population[best_idx], best_fitness, population)

        parents = tournament_selection(population, selection_ranks, tournament_size=config.tournament_size, rng=rng)
        elite_count = max(0, min(config.elite_size, config.population_size, len(population)))
        offspring: list[SpectralNetwork] = [population[index].copy() for index in ranked_indices[:elite_count]]

        if config.crossover_rate_final is None or config.generations <= 1:
            active_crossover_rate = config.crossover_rate
        else:
            progress = (generation - 1) / (config.generations - 1)
            active_crossover_rate = config.crossover_rate + progress * (
                config.crossover_rate_final - config.crossover_rate
            )
        active_crossover_rate = max(0.0, min(1.0, active_crossover_rate))

        while len(offspring) < config.population_size:
            parent = parents[rng.integers(len(parents))]
            child = parent.copy()

            if rng.random() < active_crossover_rate:
                other = parents[rng.integers(len(parents))]
                plan = plan_network_crossover(parent, other)
                matched_layers = sum(
                    1
                    for pair in plan.matched_pairs
                    if parent.layers[pair.index_a].shape == other.layers[pair.index_b].shape
                )
                mismatched_layers = sum(
                    1
                    for pair in plan.matched_pairs
                    if parent.layers[pair.index_a].shape != other.layers[pair.index_b].shape
                )
                crossover_attempts += 1
                if matched_layers > 0:
                    crossover_shape_match_events += 1
                    crossover_shape_match_layers += matched_layers
                if mismatched_layers > 0:
                    crossover_shape_mismatch_events += 1
                    crossover_shape_mismatch_layers += mismatched_layers
                crossover_cfg = CrossoverConfig(
                    method="half_rank",
                    allow_shape_mismatch=config.allow_shape_mismatch_crossover,
                )
                child = anchor_network_mode_crossover(parent, other, crossover_cfg, rng)

            initial_layer_count = len(child.layers)
            layer_index = 0
            layers_seen = 0
            while layer_index < len(child.layers) and layers_seen < initial_layer_count:
                mutation_roll = rng.random()
                node_delete_cutoff = config.node_delete_rate
                layer_delete_cutoff = node_delete_cutoff + config.layer_delete_rate
                node_split_cutoff = layer_delete_cutoff + config.node_split_rate
                layer_add_cutoff = node_split_cutoff + config.topology_mutation_rate

                if mutation_roll < node_delete_cutoff:
                    if layer_index < child.hidden_layer_count and child.layers[layer_index].shape[0] > 1:
                        node_index = int(rng.integers(child.layers[layer_index].shape[0]))
                        child.delete_hidden_node(layer_index, node_index)
                    layer_index += 1
                elif mutation_roll < layer_delete_cutoff:
                    if layer_index < child.hidden_layer_count:
                        if layer_delete_sample_fn is None:
                            child.delete_hidden_layer(layer_index)
                        else:
                            samples = layer_delete_sample_fn(rng)
                            child.delete_hidden_layer_ridge(
                                layer_index,
                                samples=samples,
                                ridge_lambda=config.layer_delete_ridge_lambda,
                            )
                    layer_index += 1
                elif mutation_roll < node_split_cutoff:
                    if layer_index < child.hidden_layer_count and child.layers[layer_index].shape[0] < config.max_hidden_width:
                        node_index = int(rng.integers(child.layers[layer_index].shape[0]))
                        child.split_hidden_node(layer_index, node_index)
                    layer_index += 1
                elif mutation_roll < layer_add_cutoff:
                    if child.hidden_layer_count < config.max_hidden_layers:
                        added_layer = child.add_signed_relu_layer(
                            layer_index=layer_index,
                            width_policy=config.signed_relu_width_policy,
                            max_hidden_width=None,
                        )
                        layer_index += 2 if added_layer else 1
                    else:
                        layer_index += 1
                else:
                    layer_index += 1
                layers_seen += 1


            offspring.append(child)

        population = offspring

    if refine_fn is not None:
        for individual in population:
            refine_fn(individual)
    final_fitnesses = [evaluate_fn(ind) for ind in population]
    final_ranked_indices = rank_indices_by_loss_then_size(population, final_fitnesses, config.parsimony_tolerance)
    best_idx = int(final_ranked_indices[0])
    final_best_fitness = float(final_fitnesses[best_idx])
    final_best_key = loss_size_key(population[best_idx], final_best_fitness, config.parsimony_tolerance)
    if final_best_key < best_key_so_far:
        best_network = population[best_idx].copy()
    return EvolutionResult(
        population=population,
        fitness_history=fitness_history,
        best_network=best_network,
        crossover_attempts=crossover_attempts,
        crossover_shape_match_events=crossover_shape_match_events,
        crossover_shape_match_layers=crossover_shape_match_layers,
        crossover_shape_mismatch_events=crossover_shape_mismatch_events,
        crossover_shape_mismatch_layers=crossover_shape_mismatch_layers,
    )















