from __future__ import annotations

import argparse
from collections import Counter
import logging
import pickle
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

from .evolution import EvolutionConfig, dense_parameter_count, run_evolution
from .network import NetworkConfig, SpectralNetwork
from .metrics import binary_accuracy
from .spectral import optimize_dense_parameters, optimize_spectral_parameters
from .utils import seed_all, sync_cuda


def topology_signature(network: SpectralNetwork) -> str:
    if not network.layers:
        return "[]"
    dims = [network.layers[0].shape[1], *[layer.shape[0] for layer in network.layers]]
    return "-".join(str(dim) for dim in dims)




def binary_positive_rate(network: SpectralNetwork, inputs: np.ndarray) -> float:
    outputs = network.forward(inputs)
    return float(np.mean(np.asarray(outputs).reshape(-1) >= 0.0))


def balanced_binary_accuracy(network: SpectralNetwork, inputs: np.ndarray, targets: np.ndarray) -> float:
    outputs = network.forward(inputs)
    predicted = np.asarray(outputs).reshape(-1) >= 0.0
    truth = np.asarray(targets).reshape(-1) >= 0.5
    positives = truth
    negatives = ~truth
    true_positive_rate = float(np.mean(predicted[positives] == truth[positives])) if np.any(positives) else 0.0
    true_negative_rate = float(np.mean(predicted[negatives] == truth[negatives])) if np.any(negatives) else 0.0
    return 0.5 * (true_positive_rate + true_negative_rate)

def format_topology_counts(population: list[SpectralNetwork], top_k: int = 5) -> str:
    counts = Counter(topology_signature(network) for network in population)
    parts = [f"{topology}:{count}" for topology, count in counts.most_common(top_k)]
    hidden = len(counts) - len(parts)
    if hidden > 0:
        parts.append(f"+{hidden} more")
    return ", ".join(parts) if parts else "[]"


def print_generation_report(
    generation: int,
    total_generations: int,
    best_net: SpectralNetwork,
    loss: float,
    population: list[SpectralNetwork],
    accuracy: float,
    extra: str = "",
) -> None:
    extra_part = f" | {extra}" if extra else ""
    print(
        f"Generation {generation}/{total_generations} | "
        f"best_loss={loss:.6f} | "
        f"best_accuracy={accuracy:.2%} | "
        f"best_topology={topology_signature(best_net)} | "
        f"params={dense_parameter_count(best_net)}"
        f"{extra_part} | "
        f"population_topologies={format_topology_counts(population)}",
        flush=True,
    )

def print_topology_summary(result, top_k: int = 5) -> None:
    counts = Counter(topology_signature(network) for network in result.population)
    print("Final population topology counts:")
    for topology, count in counts.most_common(top_k):
        print(f"  {topology}: {count}")
    if counts:
        topology, count = counts.most_common(1)[0]
        print(f"Most common final topology: {topology} ({count}/{len(result.population)})")


def parse_rank_fraction(raw: str | None) -> float | None:
    if raw is None or raw.strip().lower() in {"", "none", "off"}:
        return None
    value = raw.strip()
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        fraction = float(numerator) / float(denominator)
    else:
        fraction = float(value)
    if fraction <= 0.0:
        raise argparse.ArgumentTypeError("rank fraction must be positive")
    return fraction


def refine_lamarckian_child(
    individual: SpectralNetwork,
    inputs: np.ndarray,
    targets: np.ndarray,
    loss: str,
    config: EvolutionConfig,
) -> None:
    if config.refine_steps <= 0:
        return
    if config.refine_method == "dense-gd":
        optimize_dense_parameters(
            individual,
            inputs,
            targets,
            loss=loss,
            steps=config.refine_steps,
            lr=config.refine_lr,
            optimize_bias=config.optimize_bias,
            seed=int(config.seed + 1),
            batch_size=config.refine_batch_size,
            optimizer_name=config.refine_optimizer,
        )
    elif config.refine_method == "svd-gd":
        optimize_spectral_parameters(
            individual,
            inputs,
            targets,
            loss=loss,
            steps=config.refine_steps,
            lr=config.refine_lr,
            optimize_bias=config.optimize_bias,
            optimize_uv=True,
            seed=int(config.seed + 1),
            batch_size=config.refine_batch_size,
            optimizer_name=config.refine_optimizer,
        )
    else:
        raise ValueError(f"Unsupported refine_method: {config.refine_method}")



def make_layer_delete_sample_fn(inputs: np.ndarray, batch_size: int | None):
    def sample_fn(rng: np.random.Generator) -> np.ndarray:
        effective_batch_size = len(inputs) if batch_size is None else int(batch_size)
        effective_batch_size = max(1, min(effective_batch_size, len(inputs)))
        if effective_batch_size == len(inputs):
            return inputs
        indices = rng.choice(len(inputs), size=effective_batch_size, replace=False)
        return inputs[indices]

    return sample_fn
def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    targets = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=float)
    return inputs, targets


def run_xor_experiment(config: EvolutionConfig) -> None:
    rng = seed_all(config.seed)
    inputs, targets = xor_dataset()
    population = [SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[2, 1], r_max=config.r_max, seed=int(config.seed + i))) for i in range(config.population_size)]

    start_time = time.perf_counter()

    def evaluate(individual: SpectralNetwork) -> float:
        return individual.evaluate(inputs, targets, loss="bce")

    def refine_child(individual: SpectralNetwork) -> None:
        refine_lamarckian_child(individual, inputs, targets, "bce", config)

    result = run_evolution(
        population,
        evaluate,
        config,
        refine_fn=refine_child,
        progress_fn=lambda generation, best_net, loss, population: print_generation_report(
            generation,
            config.generations,
            best_net,
            loss,
            population,
            binary_accuracy(best_net, inputs, targets),
        ),
        layer_delete_sample_fn=make_layer_delete_sample_fn(inputs, config.refine_batch_size),
    )
    print_topology_summary(result)
    best = result.best_network
    best_loss = best.evaluate(inputs, targets, loss="bce")
    acc = binary_accuracy(best, inputs, targets)
    sync_cuda()
    duration = time.perf_counter() - start_time

    logging.info("XOR experiment complete: best_loss=%.6f best_accuracy=%.2f duration=%.2fs", best_loss, acc, duration)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best accuracy: {acc:.2%}")
    print(f"Train+eval time: {duration:.2f}s")


def run_xor_gd_baseline(epochs: int = 2000, lr: float = 0.1, seed: int = 0) -> None:
    torch.manual_seed(seed)
    inputs, targets = xor_dataset()
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    start_time = time.perf_counter()
    report_interval = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % report_interval == 0 or epoch == epochs:
            with torch.no_grad():
                output = model(x)
                predictions = (output >= 0.5).float()
                current_accuracy = (predictions == y).float().mean().item()
            print(f"GD progress [{epoch}/{epochs}] best_loss={best_loss:.6f} accuracy={current_accuracy:.2%}", flush=True)

    model.load_state_dict(best_state)
    with torch.no_grad():
        output = model(x)
        predictions = (output >= 0.5).float()
        accuracy = (predictions == y).float().mean().item()
    sync_cuda()
    duration = time.perf_counter() - start_time

    logging.info(
        "GD benchmark complete: best_loss=%.6f best_accuracy=%.2f duration=%.2fs",
        best_loss,
        accuracy,
        duration,
    )
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best accuracy: {accuracy:.2%}")
    print(f"Train+eval time: {duration:.2f}s")


def run_xor_svd_gd_baseline(
    steps: int = 200,
    lr: float = 0.05,
    seed: int = 0,
    optimize_bias: bool = True,
    r_max: int = 4,
    rank_fraction: float | None = None,
    batch_size: int | None = None,
    alpha_lr: float | None = None,
    uv_lr: float | None = None,
    bias_lr: float | None = None,
) -> None:
    rng = seed_all(seed)
    inputs, targets = xor_dataset()
    network = SpectralNetwork.from_architecture(
        NetworkConfig(layer_dims=[2, 8, 1], r_max=r_max, seed=seed, rank_fraction=rank_fraction)
    )

    report_interval = max(1, steps // 10)
    train_duration = optimize_spectral_parameters(
        network,
        inputs,
        targets,
        loss="bce",
        steps=steps,
        lr=lr,
        alpha_lr=alpha_lr,
        uv_lr=uv_lr,
        bias_lr=bias_lr,
        optimize_bias=optimize_bias,
        optimize_uv=True,
        progress_interval=report_interval,
        progress_fn=lambda step, net: print(
            f"SVD-GD progress [{step}/{steps}] loss={net.evaluate(inputs, targets, loss='bce'):.6f} accuracy={binary_accuracy(net, inputs, targets):.2%}",
            flush=True,
        ),
        seed=seed,
        batch_size=batch_size,
    )
    eval_start = time.perf_counter()
    best_loss = network.evaluate(inputs, targets, loss="bce")
    acc = binary_accuracy(network, inputs, targets)
    sync_cuda()
    duration = train_duration + (time.perf_counter() - eval_start)

    logging.info(
        "SVD-GD benchmark complete: best_loss=%.6f best_accuracy=%.2f duration=%.2fs",
        best_loss,
        acc,
        duration,
    )
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best accuracy: {acc:.2%}")
    print(f"Train+eval time: {duration:.2f}s")


# -- Digits dataset helpers and benchmarks --

def load_digits_dataset(train_path: Path | None = None, test_path: Path | None = None):
    # repository root is two parents above this file
    repo_root = Path(__file__).resolve().parents[2]
    local_dir = repo_root / "datasets" / "digits"
    sibling_dir = repo_root.parent / "Neural evolution" / "data"

    train = Path(train_path) if train_path else (local_dir / "digits_train.csv")
    test = Path(test_path) if test_path else (local_dir / "digits_test.csv")

    if not train.exists() or not test.exists():
        train = Path(train_path) if train_path else (sibling_dir / "digits_train.csv")
        test = Path(test_path) if test_path else (sibling_dir / "digits_test.csv")

    tr = np.genfromtxt(str(train), delimiter=",", skip_header=1)
    te = np.genfromtxt(str(test), delimiter=",", skip_header=1)

    if tr.ndim != 2 or te.ndim != 2 or tr.shape[0] < 2 or te.shape[0] < 2:
        train = Path(train_path) if train_path else (sibling_dir / "digits_train.csv")
        test = Path(test_path) if test_path else (sibling_dir / "digits_test.csv")
        tr = np.genfromtxt(str(train), delimiter=",", skip_header=1)
        te = np.genfromtxt(str(test), delimiter=",", skip_header=1)

    x_train = tr[:, 1:] / 16.0
    y_train = tr[:, 0].astype(int)
    x_test = te[:, 1:] / 16.0
    y_test = te[:, 0].astype(int)
    return x_train, y_train, x_test, y_test


def load_sign_mnist_dataset(train_path: Path | None = None, test_path: Path | None = None):
    repo_root = Path(__file__).resolve().parents[2]
    local_dir = repo_root / "datasets" / "digits"
    sibling_dir = repo_root.parent / "Neural evolution" / "data"

    train = Path(train_path) if train_path else (local_dir / "sign_mnist_train.csv")
    test = Path(test_path) if test_path else (local_dir / "sign_mnist_test.csv")

    if not train.exists() or not test.exists():
        train = Path(train_path) if train_path else (sibling_dir / "sign_mnist_train.csv")
        test = Path(test_path) if test_path else (sibling_dir / "sign_mnist_test.csv")

    tr = np.genfromtxt(str(train), delimiter=",", skip_header=1)
    te = np.genfromtxt(str(test), delimiter=",", skip_header=1)

    x_train = tr[:, 1:] / 255.0
    y_train_raw = tr[:, 0].astype(int)
    x_test = te[:, 1:] / 255.0
    y_test_raw = te[:, 0].astype(int)
    classes = sorted(set(y_train_raw.tolist()) | set(y_test_raw.tolist()))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y_train = np.array([label_to_index[label] for label in y_train_raw], dtype=int)
    y_test = np.array([label_to_index[label] for label in y_test_raw], dtype=int)
    return x_train, y_train, x_test, y_test

def _load_cifar10_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        payload = pickle.load(f, encoding="latin1")
    data = np.asarray(payload["data"], dtype=np.float32)
    labels = np.asarray(payload.get("labels", payload.get("fine_labels")), dtype=int)
    if data.ndim != 2 or data.shape[1] != 3072:
        raise ValueError(f"Unexpected CIFAR-10 batch shape in {path}: {data.shape}")
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    return images, labels


def _prepare_cifar10_images(images: np.ndarray, image_size: int = 64, grayscale: bool = False) -> np.ndarray:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    processed: list[np.ndarray] = []
    for image in images:
        pil_image = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB")
        if grayscale:
            pil_image = pil_image.convert("L")
        if image_size != 32:
            pil_image = pil_image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        processed.append(arr.reshape(-1))
    return np.stack(processed).astype(np.float32)


def load_cifar10_dataset(
    dataset_dir: Path | None = None,
    image_size: int = 64,
    grayscale: bool = False,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    seed: int = 0,
):
    """Load CIFAR-10 from the standard cifar-10-batches-py directory.

    The default preprocessing converts images to 64x64 RGB vectors, matching
    CIFAR-style image inputs while still keeping the dense baseline explicit.
    """
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(dataset_dir) if dataset_dir else None,
        repo_root / "datasets" / "cifar-10-batches-py",
        repo_root / "datasets" / "cifar10" / "cifar-10-batches-py",
        repo_root.parent / "Neural evolution" / "data" / "cifar-10-batches-py",
    ]
    root = next((candidate for candidate in candidates if candidate and candidate.exists()), None)
    if root is None:
        raise FileNotFoundError(
            "Expected CIFAR-10 Python batches under datasets/cifar-10-batches-py. "
            "Download and extract cifar-10-python.tar.gz from the official CIFAR-10 site."
        )

    train_images: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for batch_index in range(1, 6):
        images, labels = _load_cifar10_batch(root / f"data_batch_{batch_index}")
        train_images.append(images)
        train_labels.append(labels)
    x_train_images = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0).astype(int)
    x_test_images, y_test = _load_cifar10_batch(root / "test_batch")
    y_test = y_test.astype(int)

    rng = np.random.default_rng(seed)
    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(y_train):
        indices = rng.permutation(len(y_train))[:max_train_samples]
        x_train_images = x_train_images[indices]
        y_train = y_train[indices]
    if max_test_samples is not None and max_test_samples > 0 and max_test_samples < len(y_test):
        indices = rng.permutation(len(y_test))[:max_test_samples]
        x_test_images = x_test_images[indices]
        y_test = y_test[indices]

    x_train = _prepare_cifar10_images(x_train_images, image_size=image_size, grayscale=grayscale)
    x_test = _prepare_cifar10_images(x_test_images, image_size=image_size, grayscale=grayscale)
    return x_train, y_train, x_test, y_test



def load_cats_dogs_dataset(
    dataset_dir: Path | None = None,
    image_size: int = 32,
    test_fraction: float = 0.2,
    seed: int = 0,
    max_samples: int | None = None,
):
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(dataset_dir) if dataset_dir else None,
        repo_root / "datasets" / "cats_dogs",
        repo_root / "datasets" / "dataset",
    ]
    root = next((candidate for candidate in candidates if candidate and candidate.exists()), None)
    if root is None:
        raise FileNotFoundError("Expected cats-dogs data under datasets/cats_dogs or datasets/dataset")

    cache_path = root / f"cats_dogs_{image_size}x{image_size}_gray_crops.npz"
    if cache_path.exists():
        cached = np.load(cache_path)
        x_all = cached["x"].astype(np.float32)
        y_all = cached["y"].astype(np.float32)
    else:
        annotations = root / "annotations"
        images = root / "images"
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for annotation_path in sorted(annotations.glob("*.xml")):
            tree = ET.parse(annotation_path)
            xml_root = tree.getroot()
            filename = xml_root.findtext("filename")
            label_name = xml_root.findtext("object/name")
            if filename is None or label_name not in {"cat", "dog"}:
                continue
            image_path = images / filename
            if not image_path.exists():
                continue
            box = xml_root.find("object/bndbox")
            with Image.open(image_path) as image:
                image = image.convert("L")
                if box is not None:
                    width, height = image.size
                    xmin = max(0, int(float(box.findtext("xmin", "0"))))
                    ymin = max(0, int(float(box.findtext("ymin", "0"))))
                    xmax = min(width, int(float(box.findtext("xmax", str(width)))))
                    ymax = min(height, int(float(box.findtext("ymax", str(height)))))
                    if xmax > xmin and ymax > ymin:
                        image = image.crop((xmin, ymin, xmax, ymax))
                image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
                xs.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
            ys.append(0 if label_name == "cat" else 1)
        if not xs:
            raise ValueError(f"No cat/dog records found in {root}")
        x_all = np.stack(xs).astype(np.float32)
        y_all = np.array(ys, dtype=np.float32).reshape(-1, 1)
        np.savez_compressed(cache_path, x=x_all, y=y_all)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x_all))
    if max_samples is not None and max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]
    split = max(1, int(round(len(indices) * (1.0 - test_fraction))))
    split = min(split, len(indices) - 1)
    train_indices, test_indices = indices[:split], indices[split:]
    return x_all[train_indices], y_all[train_indices], x_all[test_indices], y_all[test_indices]


def run_digits_experiment(config: EvolutionConfig, train_path: Path | None = None, test_path: Path | None = None) -> None:
    x_train, y_train, x_test, y_test = load_digits_dataset(train_path, test_path)
    population = [SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[64, 10], r_max=config.r_max, seed=int(config.seed + i))) for i in range(config.population_size)]


    start_time = time.perf_counter()

    def evaluate(individual: SpectralNetwork) -> float:
        return individual.evaluate(x_test, y_test, loss="ce")

    def refine_child(individual: SpectralNetwork) -> None:
        refine_lamarckian_child(individual, x_train, y_train, "ce", config)

    result = run_evolution(
        population,
        evaluate,
        config,
        refine_fn=refine_child,
        progress_fn=lambda generation, best_net, loss, population: print_generation_report(
            generation,
            config.generations,
            best_net,
            loss,
            population,
            best_net.accuracy(x_test, y_test),
        ),
        layer_delete_sample_fn=make_layer_delete_sample_fn(x_train, config.refine_batch_size),
    )
    print_topology_summary(result)
    best = result.best_network
    best_loss = best.evaluate(x_test, y_test, loss="ce")
    acc = best.accuracy(x_test, y_test)
    sync_cuda()
    duration = time.perf_counter() - start_time

    logging.info("Digits experiment complete: best_loss=%.6f best_accuracy=%.2f duration=%.2fs", best_loss, acc, duration)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best accuracy: {acc:.2%}")
    print(f"Train+eval time: {duration:.2f}s")


def run_digits_gd_baseline(
    train_path: Path | None = None,
    test_path: Path | None = None,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 0,
    batch_size: int = 256,
) -> None:
    x_train, y_train, x_test, y_test = load_digits_dataset(train_path, test_path)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X, Y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(batch_size, len(train_dataset))),
        shuffle=True,
        generator=generator,
    )

    model = nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    start = time.perf_counter()
    report_interval = max(1, epochs // 10)
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
        train_loss = total_loss / len(train_dataset)
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % report_interval == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out = model(X.to(device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                current_acc = float((preds == y_train).mean())
            print(f"GD progress [{ep}/{epochs}] best_loss={best_loss:.6f} accuracy={current_acc:.2%}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x_test, dtype=torch.float32, device=device))
        preds = torch.argmax(out, dim=1).cpu().numpy()
        acc = float((preds == y_test).mean())
    sync_cuda()
    duration = time.perf_counter() - start

    print(f"GD baseline - best_loss={best_loss:.6f} acc={acc:.2%} train+eval_time={duration:.2f}s")


def run_digits_svd_gd_baseline(train_path: Path | None = None, test_path: Path | None = None, steps=200, lr=0.05, seed=0, r_max=10, rank_fraction: float | None = None, optimize_bias=True, batch_size: int | None = None, alpha_lr: float | None = None, uv_lr: float | None = None, bias_lr: float | None = None) -> None:
    x_train, y_train, x_test, y_test = load_digits_dataset(train_path, test_path)
    seed_all(seed)
    network = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[64, 32, 10], r_max=r_max, seed=seed, rank_fraction=rank_fraction))
    report_interval = max(1, steps // 10)
    train_duration = optimize_spectral_parameters(
        network,
        x_train,
        y_train,
        loss="ce",
        steps=steps,
        lr=lr,
        alpha_lr=alpha_lr,
        uv_lr=uv_lr,
        bias_lr=bias_lr,
        optimize_bias=optimize_bias,
        optimize_uv=True,
        seed=seed,
        progress_interval=report_interval,
        progress_fn=lambda step, net: print(
            f"SVD-GD progress [{step}/{steps}] loss={net.evaluate(x_test, y_test, loss='ce'):.6f} accuracy={net.accuracy(x_test, y_test):.2%}",
            flush=True,
        ),
        batch_size=batch_size,
    )
    eval_start = time.perf_counter()
    loss = network.evaluate(x_test, y_test, loss="ce")
    acc = network.accuracy(x_test, y_test)
    sync_cuda()
    duration = train_duration + (time.perf_counter() - eval_start)
    print(f"SVD-GD baseline - loss={loss:.6f} acc={acc:.2%} train+eval_time={duration:.2f}s")


def run_sign_mnist_experiment(config: EvolutionConfig, train_path: Path | None = None, test_path: Path | None = None) -> None:
    x_train, y_train, x_test, y_test = load_sign_mnist_dataset(train_path, test_path)
    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    population = [SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[784, num_classes], r_max=config.r_max, seed=int(config.seed + i))) for i in range(config.population_size)]


    start_time = time.perf_counter()

    def evaluate(individual: SpectralNetwork) -> float:
        return individual.evaluate(x_test, y_test, loss="ce")

    def refine_child(individual: SpectralNetwork) -> None:
        refine_lamarckian_child(individual, x_train, y_train, "ce", config)

    result = run_evolution(
        population,
        evaluate,
        config,
        refine_fn=refine_child,
        progress_fn=lambda generation, best_net, loss, population: print_generation_report(
            generation,
            config.generations,
            best_net,
            loss,
            population,
            best_net.accuracy(x_test, y_test),
        ),
        layer_delete_sample_fn=make_layer_delete_sample_fn(x_train, config.refine_batch_size),
    )
    print_topology_summary(result)
    best = result.best_network
    best_loss = best.evaluate(x_test, y_test, loss="ce")
    acc = best.accuracy(x_test, y_test)
    sync_cuda()
    duration = time.perf_counter() - start_time

    logging.info("Sign-MNIST experiment complete: best_loss=%.6f best_accuracy=%.2f duration=%.2fs", best_loss, acc, duration)
    print(f"Best loss: {best_loss:.6f}")
    print(f"Best accuracy: {acc:.2%}")
    print(f"Train+eval time: {duration:.2f}s")


def run_sign_mnist_gd_baseline(
    train_path: Path | None = None,
    test_path: Path | None = None,
    epochs: int = 50,
    lr: float = 0.01,
    seed: int = 0,
    batch_size: int = 256,
) -> None:
    x_train, y_train, x_test, y_test = load_sign_mnist_dataset(train_path, test_path)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X, Y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=max(1, min(batch_size, len(train_dataset))),
        shuffle=True,
        generator=generator,
    )

    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, int(max(y_train.max(), y_test.max()) + 1)),
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    report_interval = max(1, epochs // 10)
    start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * xb.shape[0]
        train_loss = total_loss / len(train_dataset)
        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % report_interval == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(x_test, dtype=torch.float32, device=device))
                preds = torch.argmax(out, dim=1).cpu().numpy()
                current_acc = float((preds == y_test).mean())
            print(f"GD progress [{ep}/{epochs}] best_loss={best_loss:.6f} accuracy={current_acc:.2%}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x_test, dtype=torch.float32, device=device))
        preds = torch.argmax(out, dim=1).cpu().numpy()
        acc = float((preds == y_test).mean())
    sync_cuda()
    duration = time.perf_counter() - start

    print(f"Sign-MNIST GD baseline - best_loss={best_loss:.6f} acc={acc:.2%} train+eval_time={duration:.2f}s")


def run_sign_mnist_svd_gd_baseline(train_path: Path | None = None, test_path: Path | None = None, steps=200, lr=0.001, seed=0, r_max=10, rank_fraction: float | None = None, optimize_bias=True, batch_size: int | None = None, alpha_lr: float | None = None, uv_lr: float | None = None, bias_lr: float | None = None) -> None:
    x_train, y_train, x_test, y_test = load_sign_mnist_dataset(train_path, test_path)
    seed_all(seed)
    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    network = SpectralNetwork.from_architecture(NetworkConfig(layer_dims=[784, 512, 256, 128, num_classes], r_max=r_max, seed=seed, rank_fraction=rank_fraction))
    report_interval = max(1, steps // 10)
    train_duration = optimize_spectral_parameters(
        network,
        x_train,
        y_train,
        loss="ce",
        steps=steps,
        lr=lr,
        alpha_lr=alpha_lr,
        uv_lr=uv_lr,
        bias_lr=bias_lr,
        optimize_bias=optimize_bias,
        optimize_uv=True,
        seed=seed,
        progress_interval=report_interval,
        progress_fn=lambda step, net: print(
            f"SVD-GD progress [{step}/{steps}] loss={net.evaluate(x_test, y_test, loss='ce'):.6f} accuracy={net.accuracy(x_test, y_test):.2%}"
        ),
        batch_size=batch_size,
    )
    eval_start = time.perf_counter()
    loss = network.evaluate(x_test, y_test, loss="ce")
    acc = network.accuracy(x_test, y_test)
    sync_cuda()
    duration = train_duration + (time.perf_counter() - eval_start)
    print(f"Sign-MNIST SVD-GD baseline - loss={loss:.6f} acc={acc:.2%} train+eval_time={duration:.2f}s")




def run_cats_dogs_experiment(
    config: EvolutionConfig,
    dataset_dir: Path | None = None,
    image_size: int = 32,
    max_samples: int | None = None,
) -> None:
    x_train, y_train, x_test, y_test = load_cats_dogs_dataset(
        dataset_dir,
        image_size=image_size,
        seed=config.seed,
        max_samples=max_samples,
    )
    population = [
        SpectralNetwork.from_architecture(
            NetworkConfig(layer_dims=[image_size * image_size, 1], r_max=config.r_max, seed=int(config.seed + i))
        )
        for i in range(config.population_size)
    ]

    start_time = time.perf_counter()

    def evaluate(individual: SpectralNetwork) -> float:
        return individual.evaluate(x_test, y_test, loss="balanced_bce")

    def refine_child(individual: SpectralNetwork) -> None:
        refine_lamarckian_child(individual, x_train, y_train, "balanced_bce", config)

    result = run_evolution(
        population,
        evaluate,
        config,
        refine_fn=refine_child,
        progress_fn=lambda generation, best_net, loss, population: print_generation_report(
            generation,
            config.generations,
            best_net,
            loss,
            population,
            binary_accuracy(best_net, x_test, y_test),
            extra=f"balanced_acc={balanced_binary_accuracy(best_net, x_test, y_test):.2%}",
        ),
        layer_delete_sample_fn=make_layer_delete_sample_fn(x_train, config.refine_batch_size),
    )
    print_topology_summary(result)
    best = result.best_network
    best_fitness_loss = best.evaluate(x_test, y_test, loss="balanced_bce")
    best_bce_loss = best.evaluate(x_test, y_test, loss="bce")
    acc = binary_accuracy(best, x_test, y_test)
    balanced_acc = balanced_binary_accuracy(best, x_test, y_test)
    sync_cuda()
    duration = time.perf_counter() - start_time

    logging.info("Cats-Dogs evolution complete: best_balanced_loss=%.6f best_accuracy=%.2f duration=%.2fs", best_fitness_loss, acc, duration)
    print(f"Best balanced loss: {best_fitness_loss:.6f}")
    print(f"Best BCE loss: {best_bce_loss:.6f}")
    print(f"Best accuracy: {acc:.2%}")
    print(f"Best balanced accuracy: {balanced_acc:.2%}")
    print(f"Train+eval time: {duration:.2f}s")

def run_cats_dogs_gd_baseline(
    dataset_dir: Path | None = None,
    epochs: int = 50,
    lr: float = 0.0005,
    seed: int = 0,
    batch_size: int = 128,
    image_size: int = 32,
    max_samples: int | None = None,
) -> None:
    x_train, y_train, x_test, y_test = load_cats_dogs_dataset(dataset_dir, image_size=image_size, seed=seed, max_samples=max_samples)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(x_train, dtype=torch.float32)
    Y = torch.tensor(y_train, dtype=torch.float32)
    train_dataset = TensorDataset(X, Y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=max(1, min(batch_size, len(train_dataset))), shuffle=True, generator=generator)
    model = nn.Sequential(
        nn.Linear(image_size * image_size, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    report_interval = max(1, epochs // 10)
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        if epoch % report_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                xt = torch.tensor(x_test, dtype=torch.float32, device=device)
                yt = torch.tensor(y_test, dtype=torch.float32, device=device)
                output = model(xt)
                current_loss = float(loss_fn(output, yt).item())
                current_acc = float(((output >= 0.5).float() == yt).float().mean().item())
            print(
                f"Cats-Dogs GD progress [{epoch}/{epochs}] loss={current_loss:.6f} accuracy={current_acc:.2%}",
                flush=True,
            )
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_test, dtype=torch.float32, device=device)
        yt = torch.tensor(y_test, dtype=torch.float32, device=device)
        output = model(xt)
        loss = float(loss_fn(output, yt).item())
        acc = float(((output >= 0.5).float() == yt).float().mean().item())
    sync_cuda()
    print(f"Cats-Dogs GD baseline - loss={loss:.6f} acc={acc:.2%} train+eval_time={time.perf_counter() - start:.2f}s")


def run_cats_dogs_svd_gd_baseline(
    dataset_dir: Path | None = None,
    steps: int = 200,
    lr: float = 0.0005,
    seed: int = 0,
    r_max: int = 10,
    rank_fraction: float | None = None,
    optimize_bias: bool = True,
    batch_size: int | None = 128,
    alpha_lr: float | None = None,
    uv_lr: float | None = None,
    bias_lr: float | None = None,
    image_size: int = 32,
    max_samples: int | None = None,
) -> None:
    x_train, y_train, x_test, y_test = load_cats_dogs_dataset(dataset_dir, image_size=image_size, seed=seed, max_samples=max_samples)
    seed_all(seed)
    network = SpectralNetwork.from_architecture(
        NetworkConfig(layer_dims=[image_size * image_size, 256, 64, 16, 1], r_max=r_max, seed=seed, rank_fraction=rank_fraction)
    )
    start = time.perf_counter()
    report_interval = max(1, steps // 10)
    optimize_spectral_parameters(
        network,
        x_train,
        y_train,
        loss="bce",
        steps=steps,
        lr=lr,
        alpha_lr=alpha_lr,
        uv_lr=uv_lr,
        bias_lr=bias_lr,
        optimize_bias=optimize_bias,
        optimize_uv=True,
        seed=seed,
        progress_interval=report_interval,
        progress_fn=lambda step, net: print(
            f"Cats-Dogs SVD-GD progress [{step}/{steps}] "
            f"loss={net.evaluate(x_test, y_test, loss='bce'):.6f} "
            f"accuracy={binary_accuracy(net, x_test, y_test):.2%}",
            flush=True,
        ),
        batch_size=batch_size,
    )
    loss = network.evaluate(x_test, y_test, loss="bce")
    acc = binary_accuracy(network, x_test, y_test)
    sync_cuda()
    print(f"Cats-Dogs SVD-GD baseline - loss={loss:.6f} acc={acc:.2%} train+eval_time={time.perf_counter() - start:.2f}s")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run benchmarks (xor, digits, sign_mnist, cats_dogs).")
    parser.add_argument("--problem", choices=["xor", "digits", "sign_mnist", "cats_dogs"], default="xor")
    parser.add_argument("--mode", choices=["evolution", "gd", "svd-gd"], default="evolution")
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--r-max", type=int, default=4)
    parser.add_argument("--rank-fraction", default=None, help="Per-layer rank as a fraction of min(m,n), e.g. 1/2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--elitism", type=int, default=1, help="Number of best individuals copied unchanged to the next generation; default 1 to preserve the best individual each generation")
    parser.add_argument("--topology-mutation-rate", type=float, default=0.05, help="Probability of adding a function-preserving signed-ReLU layer to a child")
    parser.add_argument("--node-split-rate", type=float, default=0.05, help="Probability of splitting one hidden node into two function-preserving twins")
    parser.add_argument("--layer-delete-rate", type=float, default=0.05, help="Probability of deleting one hidden layer with approximate linear merge")
    parser.add_argument("--node-delete-rate", type=float, default=0.05, help="Probability of deleting one hidden node from a hidden layer")
    parser.add_argument("--layer-delete-ridge-lambda", type=float, default=1e-3, help="Ridge penalty for data-fitted layer deletion")
    parser.add_argument("--max-hidden-layers", type=int, default=3, help="Maximum hidden layers reachable by topology mutation")
    parser.add_argument("--max-hidden-width", type=int, default=64, help="Maximum width for any hidden layer after node splits")
    parser.add_argument("--conv-kernel-size", type=int, default=3, help="Kept for legacy identity-conv experiments; current image evolution uses signed-ReLU identity expansion")
    parser.add_argument("--dense-layer-addition", action="store_true", help="Use signed-ReLU dense layer insertion; kept for compatibility")
    parser.add_argument("--signed-relu-width-policy", choices=["minimal", "mean"], default="mean", help="Width for signed-ReLU layer addition: minimal uses exact 2*min(n,m), mean embeds the exact split into max(minimal,(n+m)//2); default mean for image tasks")
    parser.add_argument("--crossover-rate", type=float, default=0.5)
    parser.add_argument("--refine-steps", "--sigma-gd-steps", dest="refine_steps", type=int, default=1,
                        help="Number of spectral refinement steps applied per child each generation")
    parser.add_argument("--refine-lr", "--sigma-gd-lr", dest="refine_lr", type=float, default=0.01,
                        help="Learning rate for Lamarckian child refinement")
    parser.add_argument("--refine-method", choices=["dense-gd", "svd-gd"], default="dense-gd",
                        help="Local learning after mutation/crossover. dense-gd matches the LEMONADE-style story; svd-gd keeps the old behavior.")
    parser.add_argument("--refine-optimizer", choices=["sgd", "adam"], default="sgd",
                        help="Optimizer for Lamarckian refinement. SGD is default so duplicated/split nodes can drift apart over time.")
    parser.add_argument("--refine-batch-size", type=int, default=256,
                        help="Mini-batch size for evolution refinement. Use 0 for full-batch updates.")
    parser.add_argument("--parsimony-tolerance", type=float, default=0.01,
                        help="Loss tolerance for tie-breaking by fewer dense parameters during evolution selection")
    parser.add_argument("--optimize-bias", action="store_true", default=True,
                        help="Optimize layer biases during local refinement; enabled by default")
    parser.add_argument("--no-optimize-bias", dest="optimize_bias", action="store_false",
                        help="Disable bias optimization during local refinement")
    parser.add_argument("--svd-gd-steps", type=int, default=200,
                        help="Number of spectral-parameter gradient steps for the baseline")
    parser.add_argument("--svd-gd-lr", type=float, default=0.001,
                        help="Fallback learning rate for all SVD-GD parameter groups")
    parser.add_argument("--svd-gd-alpha-lr", type=float, default=None,
                        help="Optional learning rate for SVD-GD alpha parameters")
    parser.add_argument("--svd-gd-uv-lr", type=float, default=None,
                        help="Optional learning rate for SVD-GD u/v direction parameters")
    parser.add_argument("--svd-gd-bias-lr", type=float, default=None,
                        help="Optional learning rate for SVD-GD bias parameters")
    parser.add_argument("--svd-gd-batch-size", type=int, default=256,
                        help="Mini-batch size for SVD-GD. Use 0 for full-batch updates.")
    parser.add_argument("--gd-epochs", type=int, default=2000)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    parser.add_argument("--gd-batch-size", type=int, default=256)
    parser.add_argument("--train", type=str, default=None, help="Optional train CSV path (for digits)")
    parser.add_argument("--test", type=str, default=None, help="Optional test CSV path (for digits)")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Optional cats-dogs dataset folder")
    parser.add_argument("--image-size", type=int, default=32, help="Cats-dogs crop size, image_size x image_size grayscale")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap for image experiments")
    args = parser.parse_args()

    if args.problem == "xor":
        if args.mode == "gd":
            run_xor_gd_baseline(epochs=args.gd_epochs, lr=args.gd_lr, seed=args.seed)
        elif args.mode == "svd-gd":
            run_xor_svd_gd_baseline(
                steps=args.svd_gd_steps,
                lr=args.svd_gd_lr,
                seed=args.seed,
                optimize_bias=args.optimize_bias,
                r_max=args.r_max,
                rank_fraction=parse_rank_fraction(args.rank_fraction),
                batch_size=args.svd_gd_batch_size or None,
                alpha_lr=args.svd_gd_alpha_lr,
                uv_lr=args.svd_gd_uv_lr,
                bias_lr=args.svd_gd_bias_lr,
            )
        else:
            config = EvolutionConfig(
                population_size=args.population,
                generations=args.generations,
                r_max=args.r_max,
                elite_size=args.elitism,
                crossover_rate=args.crossover_rate,
                topology_mutation_rate=args.topology_mutation_rate,
                node_split_rate=args.node_split_rate,
                layer_delete_rate=args.layer_delete_rate,
                node_delete_rate=args.node_delete_rate,
                layer_delete_ridge_lambda=args.layer_delete_ridge_lambda,
                max_hidden_layers=args.max_hidden_layers,
                max_hidden_width=args.max_hidden_width,
                refine_steps=args.refine_steps,
                refine_lr=args.refine_lr,
                refine_method=args.refine_method,
                refine_optimizer=args.refine_optimizer,
                refine_batch_size=args.refine_batch_size or None,
                optimize_bias=args.optimize_bias,
                parsimony_tolerance=args.parsimony_tolerance,
                seed=args.seed,
            )
            run_xor_experiment(config)
    elif args.problem == "digits":
        if args.mode == "gd":
            run_digits_gd_baseline(train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None, epochs=args.gd_epochs, lr=args.gd_lr, seed=args.seed, batch_size=args.gd_batch_size)
        elif args.mode == "svd-gd":
            run_digits_svd_gd_baseline(train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None, steps=args.svd_gd_steps, lr=args.svd_gd_lr, seed=args.seed, r_max=args.r_max, rank_fraction=parse_rank_fraction(args.rank_fraction), optimize_bias=args.optimize_bias, batch_size=args.svd_gd_batch_size or None, alpha_lr=args.svd_gd_alpha_lr, uv_lr=args.svd_gd_uv_lr, bias_lr=args.svd_gd_bias_lr)
        else:
            config = EvolutionConfig(
                population_size=args.population,
                generations=args.generations,
                r_max=args.r_max,
                elite_size=args.elitism,
                crossover_rate=args.crossover_rate,
                topology_mutation_rate=args.topology_mutation_rate,
                node_split_rate=args.node_split_rate,
                layer_delete_rate=args.layer_delete_rate,
                node_delete_rate=args.node_delete_rate,
                layer_delete_ridge_lambda=args.layer_delete_ridge_lambda,
                max_hidden_layers=args.max_hidden_layers,
                max_hidden_width=args.max_hidden_width,
                convolution_image_shape=None if args.dense_layer_addition else (8, 8),
                convolution_kernel_size=args.conv_kernel_size,
                signed_relu_width_policy=args.signed_relu_width_policy,
                refine_steps=args.refine_steps,
                refine_lr=args.refine_lr,
                refine_method=args.refine_method,
                refine_optimizer=args.refine_optimizer,
                refine_batch_size=args.refine_batch_size or None,
                optimize_bias=args.optimize_bias,
                parsimony_tolerance=args.parsimony_tolerance,
                seed=args.seed,
            )
            run_digits_experiment(config, train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None)
    elif args.problem == "sign_mnist":
        if args.mode == "gd":
            run_sign_mnist_gd_baseline(train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None, epochs=args.gd_epochs, lr=args.gd_lr, seed=args.seed, batch_size=args.gd_batch_size)
        elif args.mode == "svd-gd":
            run_sign_mnist_svd_gd_baseline(train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None, steps=args.svd_gd_steps, lr=args.svd_gd_lr, seed=args.seed, r_max=args.r_max, rank_fraction=parse_rank_fraction(args.rank_fraction), optimize_bias=args.optimize_bias, batch_size=args.svd_gd_batch_size or None, alpha_lr=args.svd_gd_alpha_lr, uv_lr=args.svd_gd_uv_lr, bias_lr=args.svd_gd_bias_lr)
        else:
            config = EvolutionConfig(
                population_size=args.population,
                generations=args.generations,
                r_max=args.r_max,
                elite_size=args.elitism,
                crossover_rate=args.crossover_rate,
                topology_mutation_rate=args.topology_mutation_rate,
                node_split_rate=args.node_split_rate,
                layer_delete_rate=args.layer_delete_rate,
                node_delete_rate=args.node_delete_rate,
                layer_delete_ridge_lambda=args.layer_delete_ridge_lambda,
                max_hidden_layers=args.max_hidden_layers,
                max_hidden_width=args.max_hidden_width,
                convolution_image_shape=None if args.dense_layer_addition else (28, 28),
                convolution_kernel_size=args.conv_kernel_size,
                signed_relu_width_policy=args.signed_relu_width_policy,
                refine_steps=args.refine_steps,
                refine_lr=args.refine_lr,
                refine_method=args.refine_method,
                refine_optimizer=args.refine_optimizer,
                refine_batch_size=args.refine_batch_size or None,
                optimize_bias=args.optimize_bias,
                parsimony_tolerance=args.parsimony_tolerance,
                seed=args.seed,
            )
            run_sign_mnist_experiment(config, train_path=Path(args.train) if args.train else None, test_path=Path(args.test) if args.test else None)
    else:  # cats_dogs
        if args.mode == "gd":
            run_cats_dogs_gd_baseline(
                dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
                epochs=args.gd_epochs,
                lr=args.gd_lr,
                seed=args.seed,
                batch_size=args.gd_batch_size,
                image_size=args.image_size,
                max_samples=args.max_samples,
            )
        elif args.mode == "svd-gd":
            run_cats_dogs_svd_gd_baseline(
                dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
                steps=args.svd_gd_steps,
                lr=args.svd_gd_lr,
                seed=args.seed,
                r_max=args.r_max,
                rank_fraction=parse_rank_fraction(args.rank_fraction),
                optimize_bias=args.optimize_bias,
                batch_size=args.svd_gd_batch_size or None,
                alpha_lr=args.svd_gd_alpha_lr,
                uv_lr=args.svd_gd_uv_lr,
                bias_lr=args.svd_gd_bias_lr,
                image_size=args.image_size,
                max_samples=args.max_samples,
            )
        else:
            config = EvolutionConfig(
                population_size=args.population,
                generations=args.generations,
                r_max=args.r_max,
                elite_size=args.elitism,
                crossover_rate=args.crossover_rate,
                topology_mutation_rate=args.topology_mutation_rate,
                node_split_rate=args.node_split_rate,
                layer_delete_rate=args.layer_delete_rate,
                node_delete_rate=args.node_delete_rate,
                layer_delete_ridge_lambda=args.layer_delete_ridge_lambda,
                max_hidden_layers=args.max_hidden_layers,
                max_hidden_width=args.max_hidden_width,
                convolution_image_shape=None if args.dense_layer_addition else (args.image_size, args.image_size),
                convolution_kernel_size=args.conv_kernel_size,
                signed_relu_width_policy=args.signed_relu_width_policy,
                refine_steps=args.refine_steps,
                refine_lr=args.refine_lr,
                refine_method=args.refine_method,
                refine_optimizer=args.refine_optimizer,
                refine_batch_size=args.refine_batch_size or None,
                optimize_bias=args.optimize_bias,
                parsimony_tolerance=args.parsimony_tolerance,
                seed=args.seed,
            )
            run_cats_dogs_experiment(
                config,
                dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
                image_size=args.image_size,
                max_samples=args.max_samples,
            )


if __name__ == "__main__":
    main()









