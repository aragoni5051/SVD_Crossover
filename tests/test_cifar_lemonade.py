import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_cifar10_cnn_lemonade import dominates, pareto_fronts, select_population


def row(arch_id: str, loss: float, params: int) -> dict:
    return {"arch_id": arch_id, "loss": loss, "params": params}


def test_dominates_requires_no_worse_loss_and_params_with_one_strict() -> None:
    assert dominates(row("small_good", 0.4, 100), row("large_bad", 0.5, 200))
    assert not dominates(row("accurate_large", 0.3, 300), row("small_less_accurate", 0.4, 100))
    assert not dominates(row("same", 0.4, 100), row("same_copy", 0.4, 100))


def test_pareto_fronts_keep_accuracy_size_tradeoffs_together() -> None:
    rows = [
        row("large_accurate", 0.2, 300),
        row("medium_middle", 0.3, 200),
        row("small_weak", 0.4, 100),
        row("dominated", 0.5, 250),
    ]

    fronts = pareto_fronts(rows)

    assert {item["arch_id"] for item in fronts[0]} == {"large_accurate", "medium_middle", "small_weak"}
    assert {item["arch_id"] for item in fronts[1]} == {"dominated"}


def test_select_population_uses_fronts_before_scalar_loss_ranking() -> None:
    rows = [
        row("large_accurate", 0.2, 300),
        row("small_weak", 0.4, 100),
        row("dominated_low_params", 0.5, 150),
    ]

    selected = select_population(rows, population_size=2, loss_tolerance=0.01)

    assert {item["arch_id"] for item in selected} == {"large_accurate", "small_weak"}
    assert all(item["pareto_rank"] == 0 for item in selected)


def test_select_population_deduplicates_architecture_before_fronts() -> None:
    rows = [
        row("same_arch", 0.51, 100),
        row("same_arch", 0.49, 100),
        row("tradeoff", 0.4, 300),
    ]

    selected = select_population(rows, population_size=3, loss_tolerance=0.01)

    assert [item for item in selected if item["arch_id"] == "same_arch"][0]["loss"] == 0.49
    assert len(selected) == 2

def test_warmstart_model_copies_matching_and_overlapping_tensors() -> None:
    import torch
    from experiments.run_cifar10_cnn_lemonade import CNNArch, LemonadeCNN, warmstart_model

    parent = LemonadeCNN(CNNArch((16, 32, 64), (1, 1, 1), (True, True, True), (False, False, False)))
    child = LemonadeCNN(CNNArch((24, 32, 64), (1, 1, 1), (True, True, True), (False, False, False)))
    with torch.no_grad():
        for value in parent.state_dict().values():
            if value.is_floating_point():
                value.fill_(0.25)
        for value in child.state_dict().values():
            if value.is_floating_point():
                value.zero_()

    copied = warmstart_model(child, parent.state_dict())

    assert copied > 0
    child_state = child.state_dict()
    assert torch.allclose(child_state["classifier.weight"], torch.full_like(child_state["classifier.weight"], 0.25))
    assert torch.allclose(child_state["features.0.blocks.0.bn.weight"][:16], torch.full((16,), 0.25))
