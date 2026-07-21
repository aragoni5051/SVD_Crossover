from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_ga.benchmarks import load_cifar10_dataset


@dataclass(frozen=True)
class CNNArch:
    channels: tuple[int, int, int]
    depths: tuple[int, int, int]
    separable: tuple[bool, bool, bool]
    skips: tuple[bool, bool, bool]


def arch_id(arch: CNNArch) -> str:
    sep = "".join("S" if value else "C" for value in arch.separable)
    skip = "".join("R" if value else "-" for value in arch.skips)
    return f"ch{'-'.join(map(str, arch.channels))}_d{'-'.join(map(str, arch.depths))}_{sep}_{skip}"


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, separable: bool) -> None:
        super().__init__()
        if separable:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, depth: int, separable: bool, skip: bool) -> None:
        super().__init__()
        blocks = []
        current = in_ch
        for _ in range(depth):
            blocks.append(ConvBNReLU(current, out_ch, separable))
            current = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.use_skip = skip and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        if self.use_skip and out.shape == x.shape:
            out = out + x
        return out


class LemonadeCNN(nn.Module):
    def __init__(self, arch: CNNArch, num_classes: int = 10) -> None:
        super().__init__()
        c1, c2, c3 = arch.channels
        d1, d2, d3 = arch.depths
        s1, s2, s3 = arch.separable
        k1, k2, k3 = arch.skips
        self.features = nn.Sequential(
            ResidualStage(3, c1, d1, s1, k1),
            nn.MaxPool2d(2),
            ResidualStage(c1, c2, d2, s2, k2),
            nn.MaxPool2d(2),
            ResidualStage(c2, c3, d3, s3, k3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def copy_tensor_overlap(target: torch.Tensor, source: torch.Tensor) -> int:
    if target.ndim != source.ndim:
        return 0
    if target.shape == source.shape:
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        return int(target.numel())
    if target.ndim == 0:
        return 0
    slices = tuple(slice(0, min(target.shape[dim], source.shape[dim])) for dim in range(target.ndim))
    if any(part.stop == 0 for part in slices):
        return 0
    target[slices].copy_(source[slices].to(device=target.device, dtype=target.dtype))
    return int(math.prod(part.stop for part in slices))


def warmstart_model(model: nn.Module, parent_state: dict[str, torch.Tensor] | None) -> int:
    if parent_state is None:
        return 0
    child_state = model.state_dict()
    copied_values = 0
    with torch.no_grad():
        for name, target in child_state.items():
            source = parent_state.get(name)
            if source is None:
                continue
            copied_values += copy_tensor_overlap(target, source)
    model.load_state_dict(child_state)
    return copied_values


def snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def initial_arches() -> list[CNNArch]:
    return [
        CNNArch((16, 32, 64), (1, 1, 1), (True, True, True), (False, False, False)),
        CNNArch((24, 48, 96), (1, 1, 1), (False, True, True), (False, False, False)),
        CNNArch((32, 64, 128), (1, 1, 1), (False, False, True), (False, False, False)),
        CNNArch((48, 96, 192), (1, 1, 1), (False, False, False), (False, False, False)),
    ]


def mutate_arch(arch: CNNArch, rng: random.Random, max_depth: int, max_channels: int) -> CNNArch:
    channels = list(arch.channels)
    depths = list(arch.depths)
    separable = list(arch.separable)
    skips = list(arch.skips)
    op = rng.choice([
        "insert_conv",
        "increase_filters",
        "add_skip",
        "remove_layer",
        "decrease_filters",
        "toggle_separable",
    ])
    stage = rng.randrange(3)
    if op == "insert_conv":
        depths[stage] = min(max_depth, depths[stage] + 1)
    elif op == "increase_filters":
        channels[stage] = min(max_channels, int(math.ceil(channels[stage] * 1.25 / 8.0)) * 8)
    elif op == "add_skip":
        skips[stage] = True
    elif op == "remove_layer":
        depths[stage] = max(1, depths[stage] - 1)
    elif op == "decrease_filters":
        channels[stage] = max(8, int(math.floor(channels[stage] * 0.75 / 8.0)) * 8)
    elif op == "toggle_separable":
        separable[stage] = not separable[stage]
    return CNNArch(tuple(channels), tuple(depths), tuple(separable), tuple(skips))


def mutate_arch_lemonade(arch: CNNArch, rng: random.Random, max_depth: int, max_channels: int) -> CNNArch:
    child = arch
    for _ in range(rng.randint(1, 3)):
        child = mutate_arch(child, rng, max_depth=max_depth, max_channels=max_channels)
    return child


def tensors_from_cifar(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train, y_train, x_val, y_val = load_cifar10_dataset(
        dataset_dir=args.cifar_dir,
        image_size=32,
        grayscale=False,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_val_samples,
        seed=args.seed,
    )
    x_train = x_train.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    x_val = x_val.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )


def train_and_eval(
    arch: CNNArch,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    parent_state: dict[str, torch.Tensor] | None = None,
) -> dict:
    model = LemonadeCNN(arch).to(device)
    inherited_values = warmstart_model(model, parent_state)
    params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_per_child), eta_min=0.0)
    start = time.perf_counter()
    for _ in range(args.epochs_per_child):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        logits = model(x_val.to(device))
        val_loss = float(criterion(logits, y_val.to(device)).item())
        val_acc = float((logits.argmax(dim=1).cpu() == y_val).float().mean().item())
    return {
        "arch": arch,
        "arch_id": arch_id(arch),
        "params": params,
        "loss": val_loss,
        "accuracy": val_acc,
        "wall_seconds": time.perf_counter() - start,
        "warmstarted_values": inherited_values,
        "state_dict": snapshot_state(model),
    }


def loss_bucket(loss: float, loss_tolerance: float) -> float:
    return round(float(loss) / loss_tolerance) * loss_tolerance if loss_tolerance > 0 else float(loss)


def duplicate_rank_key(row: dict, loss_tolerance: float) -> tuple[float, int, float]:
    return loss_bucket(float(row["loss"]), loss_tolerance), int(row["params"]), float(row["loss"])


def objective_values(row: dict) -> tuple[float, int]:
    return float(row["loss"]), int(row["params"])


def dominates(left: dict, right: dict) -> bool:
    left_loss, left_params = objective_values(left)
    right_loss, right_params = objective_values(right)
    return (
        left_loss <= right_loss
        and left_params <= right_params
        and (left_loss < right_loss or left_params < right_params)
    )


def pareto_fronts(rows: list[dict]) -> list[list[dict]]:
    remaining = list(rows)
    fronts: list[list[dict]] = []
    while remaining:
        front = [
            row
            for row in remaining
            if not any(dominates(other, row) for other in remaining if other is not row)
        ]
        fronts.append(front)
        front_ids = {id(row) for row in front}
        remaining = [row for row in remaining if id(row) not in front_ids]
    return fronts


def crowding_distances(front: list[dict]) -> dict[int, float]:
    distances = {id(row): 0.0 for row in front}
    if len(front) <= 2:
        for row in front:
            distances[id(row)] = float("inf")
        return distances

    for objective in ("loss", "params"):
        ordered = sorted(front, key=lambda row: float(row[objective]))
        distances[id(ordered[0])] = float("inf")
        distances[id(ordered[-1])] = float("inf")
        low = float(ordered[0][objective])
        high = float(ordered[-1][objective])
        if high <= low:
            continue
        for index in range(1, len(ordered) - 1):
            if math.isinf(distances[id(ordered[index])]):
                continue
            previous_value = float(ordered[index - 1][objective])
            next_value = float(ordered[index + 1][objective])
            distances[id(ordered[index])] += (next_value - previous_value) / (high - low)
    return distances


def select_population(rows: list[dict], population_size: int, loss_tolerance: float) -> list[dict]:
    """LEMONADE-style survival: deduplicate arches, then keep Pareto fronts."""
    unique: dict[str, dict] = {}
    for row in rows:
        key = str(row["arch_id"])
        if key not in unique or duplicate_rank_key(row, loss_tolerance) < duplicate_rank_key(unique[key], loss_tolerance):
            unique[key] = row

    selected: list[dict] = []
    for rank, front in enumerate(pareto_fronts(list(unique.values()))):
        front = sorted(front, key=lambda row: (float(row["loss"]), int(row["params"]), str(row["arch_id"])))
        if len(selected) + len(front) <= population_size:
            for row in front:
                row["pareto_rank"] = rank
                row["crowding_distance"] = float("inf")
            selected.extend(front)
            continue

        distances = crowding_distances(front)
        front = sorted(
            front,
            key=lambda row: (
                -distances[id(row)],
                float(row["loss"]),
                int(row["params"]),
                str(row["arch_id"]),
            ),
        )
        for row in front[: population_size - len(selected)]:
            row["pareto_rank"] = rank
            row["crowding_distance"] = distances[id(row)]
            selected.append(row)
        break
    return selected


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 CNN LEMONADE-style approximate baseline; not paper-exact LEMONADE.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--epochs-per-child", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--loss-tolerance", type=float, default=0.01)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-channels", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_cnn_lemonade_baseline"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    x_train, y_train, x_val, y_val = tensors_from_cifar(args)
    dataset = TensorDataset(x_train, y_train)
    output_dir = args.output_dir
    raw_rows: list[dict] = []
    curve_rows: list[dict] = []

    for run in range(args.runs):
        seed = args.seed + run
        rng = random.Random(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        arches = []
        bases = initial_arches()
        while len(arches) < args.population:
            arches.append(bases[len(arches) % len(bases)])

        population = []
        for arch in arches:
            population.append(train_and_eval(arch, train_loader, x_val, y_val, args, device))
        population = select_population(population, args.population, args.loss_tolerance)

        for generation in range(1, args.generations + 1):
            candidates = list(population)
            while len(candidates) < args.population * 2:
                parent = rng.choice(population)
                child_arch = mutate_arch_lemonade(parent["arch"], rng, args.max_depth, args.max_channels)
                candidates.append(train_and_eval(child_arch, train_loader, x_val, y_val, args, device, parent_state=parent["state_dict"]))
            population = select_population(candidates, args.population, args.loss_tolerance)
            best = population[0]
            front0_size = sum(1 for row in population if int(row.get("pareto_rank", 0)) == 0)
            curve_rows.append(
                {
                    "method": "cnn_lemonade_no_crossover",
                    "run": run,
                    "seed": seed,
                    "generation": generation,
                    "best_loss": best["loss"],
                    "best_accuracy": best["accuracy"],
                    "best_params": best["params"],
                    "best_arch": best["arch_id"],
                    "best_pareto_rank": best.get("pareto_rank", 0),
                    "pareto_front0_size": front0_size,
                    "best_warmstarted_values": best.get("warmstarted_values", 0),
                }
            )
            print(
                f"run {run + 1}/{args.runs} gen {generation}/{args.generations}: "
                f"acc={best['accuracy']:.4f} loss={best['loss']:.4f} params={best['params']} arch={best['arch_id']}",
                flush=True,
            )
        best = population[0]
        raw_rows.append(
            {
                "method": "cnn_lemonade_no_crossover",
                "run": run,
                "seed": seed,
                "final_loss": best["loss"],
                "final_accuracy": best["accuracy"],
                "final_params": best["params"],
                "final_arch": best["arch_id"],
                "final_pareto_rank": best.get("pareto_rank", 0),
                "final_warmstarted_values": best.get("warmstarted_values", 0),
                "final_arch_json": json.dumps(asdict(best["arch"]), sort_keys=True),
            }
        )

    write_csv(output_dir / "raw_runs.csv", raw_rows, list(raw_rows[0].keys()))
    write_csv(output_dir / "curves.csv", curve_rows, list(curve_rows[0].keys()))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
