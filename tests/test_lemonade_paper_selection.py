import numpy as np

from spectral_ga.lemonade_paper import (
    LemonadeCandidate,
    density_inverse_probabilities,
    pareto_front,
    sample_indices,
)


def candidate(name: str, cheap: tuple[float, ...], full: tuple[float, ...]) -> LemonadeCandidate:
    return LemonadeCandidate(architecture=name, cheap_objectives=cheap, objectives=full)


def test_pareto_front_keeps_non_dominated_tradeoffs() -> None:
    candidates = [
        candidate("small_weak", (100.0,), (0.4, 100.0)),
        candidate("large_strong", (300.0,), (0.2, 300.0)),
        candidate("dominated", (250.0,), (0.5, 250.0)),
    ]

    front = pareto_front(candidates)

    assert {item.architecture for item in front} == {"small_weak", "large_strong"}


def test_density_inverse_probabilities_favor_sparse_regions() -> None:
    values = [
        (100.0,),
        (101.0,),
        (102.0,),
        (1000.0,),
    ]

    probabilities = density_inverse_probabilities(values)

    assert np.isclose(float(np.sum(probabilities)), 1.0)
    assert probabilities[-1] > probabilities[0]
    assert probabilities[-1] > probabilities[1]
    assert probabilities[-1] > probabilities[2]


def test_sample_indices_without_replacement_returns_requested_count() -> None:
    indices = sample_indices(
        [(1.0,), (2.0,), (10.0,)],
        count=2,
        rng=np.random.default_rng(3),
        replace=False,
    )

    assert len(indices) == 2
    assert len(set(indices)) == 2
