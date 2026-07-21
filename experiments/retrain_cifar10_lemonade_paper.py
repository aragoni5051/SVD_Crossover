from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_ga.benchmarks import load_cifar10_dataset
from spectral_ga.lemonade_paper.search_space_i import PaperBlock, PaperCNNArch, PaperLemonadeCNN, arch_id, count_params

from run_cifar10_lemonade_paper import evaluate, resolve_device, train_epoch


def arch_from_json(value: str) -> PaperCNNArch:
    data = json.loads(value)
    stages = []
    for stage in data["stages"]:
        stages.append(tuple(PaperBlock(**block) for block in stage))
    return PaperCNNArch((stages[0], stages[1], stages[2]), int(data["next_block_id"]))


def load_arches(path: Path, limit: int | None) -> list[PaperCNNArch]:
    arches: list[PaperCNNArch] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            arches.append(arch_from_json(row["arch_json"]))
    unique: dict[str, PaperCNNArch] = {}
    for arch in arches:
        unique.setdefault(arch_id(arch), arch)
    result = list(unique.values())
    return result if limit is None else result[:limit]


def tensors_from_cifar(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_train, y_train, x_test, y_test = load_cifar10_dataset(
        dataset_dir=args.cifar_dir,
        image_size=32,
        grayscale=False,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    x_train = x_train.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    x_test = x_test.reshape(-1, 32, 32, 3).transpose(0, 3, 1, 2)
    return (
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain paper-style LEMONADE Search Space I architectures from scratch.")
    parser.add_argument("--pareto-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_lemonade_paper_retrain"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutout-length", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if not args.pareto_csv.exists():
        nearby = sorted(args.pareto_csv.parent.glob("*.csv")) if args.pareto_csv.parent.exists() else []
        available = ", ".join(str(path) for path in nearby) if nearby else "no CSV files found in that directory"
        raise FileNotFoundError(f"Pareto CSV not found: {args.pareto_csv}. Available nearby: {available}")
    arches = load_arches(args.pareto_csv, args.limit)
    x_train, y_train, x_test, y_test = tensors_from_cifar(args)
    dataset = TensorDataset(x_train, y_train)
    criterion = nn.CrossEntropyLoss()
    curve_rows: list[dict] = []
    summary_rows: list[dict] = []

    for arch_index, arch in enumerate(arches):
        seed = args.seed + arch_index
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        model = PaperLemonadeCNN(arch).to(device)
        params = count_params(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=0.0)
        best_acc = 0.0
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_epoch(model, loader, criterion, optimizer, device, args.mixup_alpha, args.cutout_length)
            scheduler.step()
            test_loss, test_acc = evaluate(model, criterion, x_test, y_test, device)
            best_acc = max(best_acc, test_acc)
            best_loss = min(best_loss, test_loss)
            curve_rows.append(
                {
                    "arch_index": arch_index,
                    "arch_id": arch_id(arch),
                    "epoch": epoch,
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "best_test_loss": best_loss,
                    "best_test_accuracy": best_acc,
                    "params": params,
                }
            )
            print(
                f"arch {arch_index + 1}/{len(arches)} epoch {epoch}/{args.epochs}: "
                f"acc={test_acc:.4f} best={best_acc:.4f} params={params} arch={arch_id(arch)}",
                flush=True,
            )
        summary_rows.append(
            {
                "arch_index": arch_index,
                "arch_id": arch_id(arch),
                "final_test_loss": curve_rows[-1]["test_loss"],
                "final_test_accuracy": curve_rows[-1]["test_accuracy"],
                "best_test_loss": best_loss,
                "best_test_accuracy": best_acc,
                "params": params,
                "arch_json": json.dumps(arch, default=lambda obj: obj.__dict__, sort_keys=True),
            }
        )

    if curve_rows:
        write_csv(args.output_dir / "curves.csv", curve_rows, list(curve_rows[0].keys()))
    if summary_rows:
        write_csv(args.output_dir / "summary.csv", summary_rows, list(summary_rows[0].keys()))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()


