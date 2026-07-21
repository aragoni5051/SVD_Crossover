from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spectral_ga.benchmarks import load_cifar10_dataset
from spectral_ga.lemonade_paper import (
    LemonadeCandidate,
    PaperCNNArch,
    PaperLemonadeCNN,
    arch_id,
    cheap_objectives,
    count_params,
    initial_arches,
    mutate_arch,
    pareto_front,
    sample_indices,
    snapshot_state,
    warmstart_model,
)


_CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
_CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(3, 1, 1)


def normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = _CIFAR10_MEAN.to(device=images.device, dtype=images.dtype)
    std = _CIFAR10_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def random_crop_flip(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    padded = nn.functional.pad(images, (padding, padding, padding, padding), mode="reflect")
    _, _, height, width = images.shape
    cropped = torch.empty_like(images)
    for index in range(images.shape[0]):
        top = int(torch.randint(0, padding * 2 + 1, (1,), device=images.device).item())
        left = int(torch.randint(0, padding * 2 + 1, (1,), device=images.device).item())
        cropped[index] = padded[index, :, top : top + height, left : left + width]
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    cropped[flip_mask] = torch.flip(cropped[flip_mask], dims=(3,))
    return cropped


def apply_cutout(images: torch.Tensor, length: int = 16) -> torch.Tensor:
    if length <= 0:
        return images
    _, _, height, width = images.shape
    output = images.clone()
    half = length // 2
    for index in range(images.shape[0]):
        center_y = int(torch.randint(0, height, (1,), device=images.device).item())
        center_x = int(torch.randint(0, width, (1,), device=images.device).item())
        y0 = max(0, center_y - half)
        y1 = min(height, center_y + half)
        x0 = max(0, center_x - half)
        x1 = min(width, center_x + half)
        output[index, :, y0:y1, x0:x1] = 0.0
    return output


def apply_mixup(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(inputs.shape[0], device=inputs.device)
    return lam * inputs + (1.0 - lam) * inputs[permutation], targets, targets[permutation], lam


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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mixup_alpha: float,
    cutout_length: int,
) -> None:
    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        xb = apply_cutout(random_crop_flip(xb), length=cutout_length)
        xb = normalize_cifar10(xb)
        xb, y_a, y_b, lam = apply_mixup(xb, yb, mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        loss.backward()
        optimizer.step()


def freeze_exactly_copied_parameters(child: nn.Module, parent_state: dict[str, torch.Tensor]) -> None:
    for name, parameter in child.named_parameters():
        source = parent_state.get(name)
        parameter.requires_grad = not (source is not None and tuple(source.shape) == tuple(parameter.shape))


def unfreeze_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def distill_child(
    parent: nn.Module,
    child: nn.Module,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if args.distill_epochs <= 0:
        return
    parent.eval()
    child.train()
    trainable_parameters = [parameter for parameter in child.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        return
    optimizer = torch.optim.SGD(trainable_parameters, lr=args.distill_lr, momentum=0.9, weight_decay=args.weight_decay)
    for _ in range(args.distill_epochs):
        for xb, _ in loader:
            xb = normalize_cifar10(xb.to(device))
            with torch.no_grad():
                teacher_probs = torch.softmax(parent(xb), dim=1)
            optimizer.zero_grad(set_to_none=True)
            student_logits = child(xb)
            student_log_probs = nn.functional.log_softmax(student_logits, dim=1)
            loss = -(teacher_probs * student_log_probs).sum(dim=1).mean()
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, criterion: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, device: torch.device) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(normalize_cifar10(x_val.to(device)))
        val_loss = float(criterion(logits, y_val.to(device)).item())
        val_acc = float((logits.argmax(dim=1).cpu() == y_val).float().mean().item())
    return val_loss, val_acc


def train_and_eval(
    arch: PaperCNNArch,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    parent_state: dict[str, torch.Tensor] | None = None,
    parent_arch: PaperCNNArch | None = None,
    needs_distillation: bool = False,
) -> dict:
    model = PaperLemonadeCNN(arch).to(device)
    inherited_values = warmstart_model(model, parent_state)
    if needs_distillation and parent_state is not None and parent_arch is not None:
        parent = PaperLemonadeCNN(parent_arch).to(device)
        parent.load_state_dict(parent_state, strict=False)
        freeze_exactly_copied_parameters(model, parent_state)
        distill_child(parent, model, train_loader, args, device)
        unfreeze_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs_per_child), eta_min=0.0)
    start = time.perf_counter()
    for _ in range(args.epochs_per_child):
        train_epoch(model, train_loader, criterion, optimizer, device, args.mixup_alpha, args.cutout_length)
        scheduler.step()
    val_loss, val_acc = evaluate(model, criterion, x_val, y_val, device)
    params = count_params(model)
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


def to_candidate(row: dict) -> LemonadeCandidate:
    return LemonadeCandidate(
        architecture=row,
        cheap_objectives=(float(np.log(float(row["params"]))),),
        objectives=(float(row["loss"]), float(np.log(float(row["params"])))),
    )


def dedupe_front(rows: list[dict]) -> list[dict]:
    best_by_arch: dict[str, dict] = {}
    for row in rows:
        key = str(row["arch_id"])
        if key not in best_by_arch or (float(row["loss"]), int(row["params"])) < (float(best_by_arch[key]["loss"]), int(best_by_arch[key]["params"])):
            best_by_arch[key] = row
    front = pareto_front([to_candidate(row) for row in best_by_arch.values()])
    return [candidate.architecture for candidate in front]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested with --device cuda, but this PyTorch install does not have CUDA available. "
            "Install a CUDA-enabled PyTorch build, or rerun with --device cpu / --device auto."
        )
    return torch.device(device_arg)


def estimate_training_work(args: argparse.Namespace, train_size: int, val_size: int) -> dict[str, int]:
    trained_models = args.runs * (len(initial_arches()) + args.generations * min(args.n_ac, args.n_pc))
    train_epochs = trained_models * args.epochs_per_child
    train_images = train_epochs * train_size
    max_distill_epochs = args.runs * args.generations * min(args.n_ac, args.n_pc) * args.distill_epochs
    val_passes = trained_models
    val_images = val_passes * val_size
    return {
        "trained_models": int(trained_models),
        "train_epochs": int(train_epochs),
        "max_distill_epochs": int(max_distill_epochs),
        "train_images": int(train_images),
        "val_passes": int(val_passes),
        "val_images": int(val_images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-oriented LEMONADE Algorithm 1 runner for CIFAR-10 Search Space I.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--n-pc", type=int, default=64, help="Generated children per generation before cheap-objective acceptance.")
    parser.add_argument("--n-ac", type=int, default=16, help="Accepted children per generation for expensive training/evaluation.")
    parser.add_argument("--epochs-per-child", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutout-length", type=int, default=0)
    parser.add_argument("--distill-epochs", type=int, default=5)
    parser.add_argument("--distill-lr", type=float, default=0.001)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_lemonade_paper_search_space_i"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-cpu-full-scale", action="store_true", help="Allow the paper-scale run on CPU after printing the work estimate.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    x_train, y_train, x_val, y_val = tensors_from_cifar(args)
    work = estimate_training_work(args, train_size=len(y_train), val_size=len(y_val))
    print(
        "LEMONADE work estimate: "
        f"device={device}, trained_models={work['trained_models']}, "
        f"train_epochs={work['train_epochs']}, max_distill_epochs={work['max_distill_epochs']}, "
        f"train_images={work['train_images']:,}, val_passes={work['val_passes']}, "
        f"val_images={work['val_images']:,}",
        flush=True,
    )
    is_full_scale = args.max_train_samples is None and args.max_val_samples is None and args.generations >= 100 and args.epochs_per_child >= 20
    if device.type == "cpu" and is_full_scale and not args.allow_cpu_full_scale:
        raise RuntimeError(
            "This is a paper-scale LEMONADE run on CPU. It will be extremely slow. "
            "Install CUDA-enabled PyTorch and use --device cuda, or add --allow-cpu-full-scale if you really want to run it on CPU."
        )
    dataset = TensorDataset(x_train, y_train)
    output_dir = args.output_dir
    raw_rows: list[dict] = []
    curve_rows: list[dict] = []
    child_rows: list[dict] = []

    for run in range(args.runs):
        seed = args.seed + run
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

        population = [
            train_and_eval(arch, train_loader, x_val, y_val, args, device)
            for arch in initial_arches()
        ]
        population = dedupe_front(population)

        for generation in range(1, args.generations + 1):
            proposed = []
            existing_arch_ids = {str(row["arch_id"]) for row in population}
            proposed_arch_ids: set[str] = set()
            proposal_attempts = 0
            max_attempts = max(args.n_pc * 10, args.n_pc + 10)
            parent_values = [(float(np.log(float(row["params"]))),) for row in population]
            while len(proposed) < args.n_pc and proposal_attempts < max_attempts:
                proposal_attempts += 1
                parent_index = sample_indices(parent_values, count=1, rng=np_rng, replace=True)[0]
                parent = population[parent_index]
                child_arch, ops, needs_distillation = mutate_arch(parent["arch"], rng)
                child_arch_id = arch_id(child_arch)
                child_row_base = {
                    "method": "lemonade_paper_search_space_i",
                    "run": run,
                    "seed": seed,
                    "generation": generation,
                    "proposal_attempt": proposal_attempts,
                    "parent_arch_id": parent["arch_id"],
                    "child_arch_id": child_arch_id,
                    "mutation_ops": ";".join(ops),
                    "needs_distillation": bool(needs_distillation),
                }
                if child_arch_id in existing_arch_ids:
                    child_rows.append({**child_row_base, "status": "duplicate_population", "loss": "", "accuracy": "", "params": count_params(PaperLemonadeCNN(child_arch))})
                    continue
                if child_arch_id in proposed_arch_ids:
                    child_rows.append({**child_row_base, "status": "duplicate_proposed", "loss": "", "accuracy": "", "params": count_params(PaperLemonadeCNN(child_arch))})
                    continue
                proposed_arch_ids.add(child_arch_id)
                proposed.append(
                    {
                        "arch": child_arch,
                        "arch_id": child_arch_id,
                        "parent": parent,
                        "ops": ops,
                        "needs_distillation": needs_distillation,
                        "cheap_objectives": cheap_objectives(child_arch),
                        "proposal_attempt": proposal_attempts,
                    }
                )
            accepted_indices = set(sample_indices(
                [proposal["cheap_objectives"] for proposal in proposed],
                count=min(args.n_ac, len(proposed)),
                rng=np_rng,
                replace=False,
            )) if proposed else set()
            children = []
            for proposal_index, proposal in enumerate(proposed):
                parent = proposal["parent"]
                child_row_base = {
                    "method": "lemonade_paper_search_space_i",
                    "run": run,
                    "seed": seed,
                    "generation": generation,
                    "proposal_attempt": proposal["proposal_attempt"],
                    "parent_arch_id": parent["arch_id"],
                    "child_arch_id": proposal["arch_id"],
                    "mutation_ops": ";".join(proposal["ops"]),
                    "needs_distillation": bool(proposal["needs_distillation"]),
                }
                if proposal_index not in accepted_indices:
                    child_rows.append({**child_row_base, "status": "rejected_cheap_objective", "loss": "", "accuracy": "", "params": count_params(PaperLemonadeCNN(proposal["arch"]))})
                    continue
                child = train_and_eval(
                    proposal["arch"],
                    train_loader,
                    x_val,
                    y_val,
                    args,
                    device,
                    parent_state=parent["state_dict"],
                    parent_arch=parent["arch"],
                    needs_distillation=bool(proposal["needs_distillation"]),
                )
                child["parent_arch_id"] = parent["arch_id"]
                child["mutation_ops"] = ";".join(proposal["ops"])
                child["needs_distillation"] = bool(proposal["needs_distillation"])
                children.append(child)
                child_rows.append({**child_row_base, "status": "accepted_trained", "loss": child["loss"], "accuracy": child["accuracy"], "params": child["params"]})

            population = dedupe_front(population + children)
            best = max(population, key=lambda row: float(row["accuracy"]))
            smallest = min(population, key=lambda row: int(row["params"]))
            curve_rows.append(
                {
                    "method": "lemonade_paper_search_space_i",
                    "run": run,
                    "seed": seed,
                    "generation": generation,
                    "pareto_size": len(population),
                    "best_loss": best["loss"],
                    "best_accuracy": best["accuracy"],
                    "best_params": best["params"],
                    "best_arch": best["arch_id"],
                    "smallest_params": smallest["params"],
                    "smallest_accuracy": smallest["accuracy"],
                    "accepted_children": len(children),
                }
            )
            print(
                f"run {run + 1}/{args.runs} gen {generation}/{args.generations}: "
                f"pareto={len(population)} best_acc={best['accuracy']:.4f} "
                f"best_params={best['params']} smallest_params={smallest['params']}",
                flush=True,
            )

        for row in population:
            raw_rows.append(
                {
                    "method": "lemonade_paper_search_space_i",
                    "run": run,
                    "seed": seed,
                    "loss": row["loss"],
                    "accuracy": row["accuracy"],
                    "params": row["params"],
                    "arch_id": row["arch_id"],
                    "arch_json": json.dumps(asdict(row["arch"]), sort_keys=True),
                    "warmstarted_values": row.get("warmstarted_values", 0),
                    "parent_arch_id": row.get("parent_arch_id", ""),
                    "mutation_ops": row.get("mutation_ops", ""),
                    "needs_distillation": row.get("needs_distillation", False),
                }
            )

    if curve_rows:
        write_csv(output_dir / "curves.csv", curve_rows, list(curve_rows[0].keys()))
    if raw_rows:
        write_csv(output_dir / "pareto_population.csv", raw_rows, list(raw_rows[0].keys()))
    if child_rows:
        write_csv(output_dir / "children.csv", child_rows, list(child_rows[0].keys()))
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()







