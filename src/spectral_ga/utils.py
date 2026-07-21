from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch




def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((predictions - targets) ** 2))


def bce_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    predictions = np.clip(predictions, eps, 1.0 - eps)
    return float(-np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))


def balanced_bce_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-8) -> float:
    predictions = np.clip(predictions, eps, 1.0 - eps)
    targets = np.asarray(targets, dtype=float)
    positive_rate = float(np.mean(targets))
    negative_rate = 1.0 - positive_rate
    positive_weight = 0.5 / max(positive_rate, eps)
    negative_weight = 0.5 / max(negative_rate, eps)
    weights = np.where(targets > 0.5, positive_weight, negative_weight)
    losses = -(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    return float(np.mean(weights * losses))


def ce_loss(predictions: np.ndarray, targets: np.ndarray, eps: float = 1e-12) -> float:
    # predictions are logits (N, C) or probabilities; convert to probabilities via softmax
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        preds = preds[:, np.newaxis]
    # numerically stable softmax
    x = preds - np.max(preds, axis=1, keepdims=True)
    exp_x = np.exp(x)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    targets_arr = np.asarray(targets).reshape(-1)
    # if targets are one-hot, convert to indices
    if targets_arr.ndim > 0 and probs.shape[1] == targets_arr.size:
        pass
    if targets_arr.ndim > 1 and targets_arr.shape[1] > 1:
        targets_idx = np.argmax(targets_arr, axis=1)
    else:
        targets_idx = targets_arr.astype(int)
    # Negative log-likelihood
    nll = -np.log(np.clip(probs[np.arange(probs.shape[0]), targets_idx], eps, 1.0))
    return float(np.mean(nll))


def torch_loss(loss: str, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if loss == "mse":
        return torch.mean((predictions - targets) ** 2)
    elif loss == "bce":
        return torch.nn.functional.binary_cross_entropy(predictions, targets)
    elif loss == "balanced_bce":
        positive_rate = torch.mean(targets)
        negative_rate = 1.0 - positive_rate
        eps = torch.tensor(1e-8, dtype=targets.dtype, device=targets.device)
        positive_weight = 0.5 / torch.maximum(positive_rate, eps)
        negative_weight = 0.5 / torch.maximum(negative_rate, eps)
        weights = torch.where(targets > 0.5, positive_weight, negative_weight)
        loss_values = torch.nn.functional.binary_cross_entropy(predictions, targets, reduction="none")
        return torch.mean(weights * loss_values)
    elif loss == "ce":
        # Cross-entropy expects integer class targets (long) and logits as predictions
        if targets.dtype != torch.long:
            targets_long = targets.squeeze().long()
        else:
            targets_long = targets
        return torch.nn.functional.cross_entropy(predictions, targets_long)
    raise ValueError(f"Unsupported loss: {loss}")


def seed_all(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def sync_cuda() -> None:
    """Synchronize CUDA device if available to ensure accurate timing when using GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

