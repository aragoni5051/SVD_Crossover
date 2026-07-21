import random

import torch

from spectral_ga.lemonade_paper import PaperLemonadeCNN, arch_id, count_params, initial_arches, mutate_arch


def test_initial_search_space_i_matches_paper_scale() -> None:
    arches = initial_arches()
    params = [count_params(PaperLemonadeCNN(arch)) for arch in arches]
    first_ops = [arch.stages[0][0].op for arch in arches]

    assert 10_000 <= params[0] <= 25_000
    assert 40_000 <= params[1] <= 70_000
    assert 80_000 <= params[2] <= 130_000
    assert 300_000 <= params[3] <= 500_000
    assert first_ops.count("sepconv") == 2


def test_paper_model_forward_shape() -> None:
    model = PaperLemonadeCNN(initial_arches()[0])

    logits = model(torch.zeros(2, 3, 32, 32))

    assert logits.shape == (2, 10)


def test_mutation_changes_or_preserves_valid_architecture() -> None:
    arch = initial_arches()[1]
    child, ops, needs_distillation = mutate_arch(arch, random.Random(3))

    assert arch_id(child).startswith("ss1_")
    assert ops
    assert isinstance(needs_distillation, bool)
    assert count_params(PaperLemonadeCNN(child)) >= 10_000
    assert PaperLemonadeCNN(child)(torch.zeros(1, 3, 32, 32)).shape == (1, 10)
