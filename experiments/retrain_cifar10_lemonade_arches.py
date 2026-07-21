from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.run_cifar10_cnn_lemonade import CNNArch, LemonadeCNN, count_params
from spectral_ga.benchmarks import load_cifar10_dataset



_CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
_CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616], dtype=torch.float32).view(3, 1, 1)
ARCH_RE = re.compile(
    r"^ch(?P<channels>\d+-\d+-\d+)_d(?P<depths>\d+-\d+-\d+)_(?P<separable>[CS]{3})_(?P<skips>[-R]{3})$"
)


def parse_arch_id(arch_id: str) -> CNNArch:
    match = ARCH_RE.match(arch_id)
    if match is None:
        raise ValueError(f"Unsupported architecture id: {arch_id}")
    channels = tuple(int(value) for value in match.group("channels").split("-"))
    depths = tuple(int(value) for value in match.group("depths").split("-"))
    separable = tuple(value == "S" for value in match.group("separable"))
    skips = tuple(value == "R" for value in match.group("skips"))
    return CNNArch(channels=channels, depths=depths, separable=separable, skips=skips)


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


class CifarTrainDataset(Dataset):
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        augment: bool = True,
        cutout_length: int = 16,
    ) -> None:
        self.images = images
        self.labels = labels
        self.augment = augment
        self.cutout_length = int(cutout_length)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]
        if self.augment:
            image = random_crop_flip(image)
        image = normalize_cifar10(image)
        if self.augment and self.cutout_length > 0:
            image = apply_cutout(image, self.cutout_length)
        return image, label


def normalize_cifar10(images: torch.Tensor) -> torch.Tensor:
    mean = _CIFAR10_MEAN.to(device=images.device, dtype=images.dtype)
    std = _CIFAR10_STD.to(device=images.device, dtype=images.dtype)
    return (images - mean) / std


def random_crop_flip(image: torch.Tensor, padding: int = 4) -> torch.Tensor:
    padded = F.pad(image.unsqueeze(0), (padding, padding, padding, padding), mode="reflect").squeeze(0)
    max_offset = 2 * padding
    top = int(torch.randint(0, max_offset + 1, (1,)).item())
    left = int(torch.randint(0, max_offset + 1, (1,)).item())
    cropped = padded[:, top : top + image.shape[1], left : left + image.shape[2]]
    if bool(torch.randint(0, 2, (1,)).item()):
        cropped = torch.flip(cropped, dims=(2,))
    return cropped


def apply_cutout(image: torch.Tensor, length: int) -> torch.Tensor:
    _, height, width = image.shape
    cutout = max(1, min(int(length), height, width))
    center_y = int(torch.randint(0, height, (1,)).item())
    center_x = int(torch.randint(0, width, (1,)).item())
    half = cutout // 2
    y1 = max(0, center_y - half)
    x1 = max(0, center_x - half)
    y2 = min(height, y1 + cutout)
    x2 = min(width, x1 + cutout)
    augmented = image.clone()
    augmented[:, y1:y2, x1:x2] = 0.0
    return augmented


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0 or inputs.shape[0] < 2:
        return inputs, targets, targets, 1.0
    beta = torch.distributions.Beta(alpha, alpha)
    lam = float(beta.sample().item())
    permutation = torch.randperm(inputs.shape[0], device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    return mixed_inputs, targets, targets[permutation], lam


def mixup_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(normalize_cifar10(x_val.to(device)))
        loss = float(criterion(logits, y_val.to(device)).item())
        acc = float((logits.argmax(dim=1).cpu() == y_val).float().mean().item())
    return loss, acc


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_arch(
    arch_id: str,
    arch_index: int,
    train_loader: DataLoader,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict], dict]:
    arch = parse_arch_id(arch_id)
    torch.manual_seed(args.seed + arch_index)
    model = LemonadeCNN(arch).to(device)
    params = count_params(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=0.0)
    curve_rows: list[dict] = []
    best = {"loss": float("inf"), "accuracy": 0.0, "epoch": 0}
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            mixed_xb, target_a, target_b, lam = mixup_batch(xb, yb, args.mixup_alpha)
            loss = mixup_loss(criterion, model(mixed_xb), target_a, target_b, lam)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_loss, val_acc = evaluate(model, criterion, x_val, y_val, device)
        if val_acc > best["accuracy"] or (val_acc == best["accuracy"] and val_loss < best["loss"]):
            best = {"loss": val_loss, "accuracy": val_acc, "epoch": epoch}
        curve_rows.append(
            {
                "arch_id": arch_id,
                "arch_index": arch_index,
                "epoch": epoch,
                "eval_loss": val_loss,
                "eval_accuracy": val_acc,
                "best_eval_loss": best["loss"],
                "best_eval_accuracy": best["accuracy"],
                "params": params,
                "mixup_alpha": args.mixup_alpha,
                "cutout_length": 0 if args.no_augment else args.cutout_length,
                "lr": args.lr,
            }
        )
        print(
            f"arch {arch_index + 1}/{len(args.arch)} epoch {epoch}/{args.epochs}: "
            f"acc={val_acc:.4f} loss={val_loss:.4f} best={best['accuracy']:.4f} "
            f"params={params} arch={arch_id}",
            flush=True,
        )

    final_loss, final_acc = evaluate(model, criterion, x_val, y_val, device)
    raw_row = {
        "arch_id": arch_id,
        "arch_index": arch_index,
        "final_loss": final_loss,
        "final_accuracy": final_acc,
        "best_loss": best["loss"],
        "best_accuracy": best["accuracy"],
        "best_epoch": best["epoch"],
        "params": params,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "mixup_alpha": args.mixup_alpha,
        "cutout_length": 0 if args.no_augment else args.cutout_length,
        "augment": not args.no_augment,
        "train_samples": int(len(train_loader.dataset)),
        "eval_samples": int(y_val.shape[0]),
        "wall_seconds": time.perf_counter() - start,
    }
    return curve_rows, raw_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain selected LEMONADE CIFAR-10 architectures from scratch with paper-like final-training defaults.")
    parser.add_argument("--arch", action="append", required=True, help="Architecture id, e.g. ch56-48-200_d2-1-2_CCS_RR-")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--no-augment", action="store_true", help="Disable random crop/flip and Cutout for smoke tests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cifar-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cifar10_cnn_lemonade_retrain"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    x_train, y_train, x_val, y_val = tensors_from_cifar(args)
    dataset = CifarTrainDataset(
        x_train,
        y_train,
        augment=not args.no_augment,
        cutout_length=0 if args.no_augment else args.cutout_length,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)

    all_curves: list[dict] = []
    raw_rows: list[dict] = []
    for arch_index, arch_id in enumerate(args.arch):
        curve_rows, raw_row = train_arch(arch_id, arch_index, train_loader, x_val, y_val, args, device)
        all_curves.extend(curve_rows)
        raw_rows.append(raw_row)

    write_csv(args.output_dir / "curves.csv", all_curves, list(all_curves[0].keys()))
    write_csv(args.output_dir / "raw_runs.csv", raw_rows, list(raw_rows[0].keys()))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
