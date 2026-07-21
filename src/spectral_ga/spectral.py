from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import time
import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import torch_loss
from .innovation import LayerGene


@dataclass
class SpectralLayer:
    residual: np.ndarray
    u: np.ndarray
    alpha: np.ndarray
    v: np.ndarray
    gene: LayerGene | None = None

    def copy(self) -> "SpectralLayer":
        return SpectralLayer(
            residual=self.residual.copy(),
            u=self.u.copy(),
            alpha=self.alpha.copy(),
            v=self.v.copy(),
            gene=None if self.gene is None else self.gene.copy(),
        )

    def to_dict(self) -> dict:
        return {
            "residual": self.residual.tolist(),
            "u": self.u.tolist(),
            "alpha": self.alpha.tolist(),
            "v": self.v.tolist(),
            "gene": None if self.gene is None else self.gene.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpectralLayer":
        return cls(
            residual=np.asarray(data["residual"], dtype=float),
            u=np.asarray(data["u"], dtype=float),
            alpha=np.asarray(data["alpha"], dtype=float),
            v=np.asarray(data["v"], dtype=float),
            gene=LayerGene.from_dict(data.get("gene")),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.residual.shape

    @property
    def rank(self) -> int:
        return self.alpha.shape[0]


def decompose_dense(weight: np.ndarray, r_max: int, keep_residual: bool = True) -> SpectralLayer:
    output_dim, input_dim = weight.shape
    u, s, vt = np.linalg.svd(weight, full_matrices=False)
    r = min(r_max, output_dim, input_dim, s.shape[0])
    u_r = u[:, :r].astype(float)
    alpha = s[:r].astype(float)
    v_r = vt[:r, :].T.astype(float)
    if keep_residual:
        reconstructed = (u_r * alpha[np.newaxis, :]) @ v_r.T
        residual = weight - reconstructed
    else:
        residual = np.zeros_like(weight, dtype=float)
    return SpectralLayer(residual=residual, u=u_r, alpha=alpha, v=v_r)


def reconstruct_layer(layer: SpectralLayer) -> np.ndarray:
    if layer.rank == 0:
        return layer.residual.copy()
    weighted_u = layer.u * layer.alpha[np.newaxis, :]
    return layer.residual + weighted_u @ layer.v.T


def _forward_spectral_torch(
    x: torch.Tensor,
    network: "SpectralNetwork",
    alpha_parameters: list[torch.nn.Parameter],
    u_parameters: list[torch.Tensor],
    v_parameters: list[torch.Tensor],
    bias_parameters: list[torch.Tensor],
    residual_tensors: list[torch.Tensor],
    loss: str,
) -> torch.Tensor:
    for idx, _layer in enumerate(network.layers):
        u_t = u_parameters[idx]
        v_t = v_parameters[idx]
        alpha_t = alpha_parameters[idx]
        weight_t = residual_tensors[idx] + (u_t * alpha_t.unsqueeze(0)) @ v_t.T
        bias_t = bias_parameters[idx]
        x = x @ weight_t.T + bias_t
        if idx < len(network.layers) - 1:
            x = torch.maximum(x, torch.zeros_like(x))
        elif loss in {"bce", "balanced_bce"}:
            x = torch.sigmoid(x)
    return x


def optimize_spectral_parameters(
    network: "SpectralNetwork",
    inputs: np.ndarray,
    targets: np.ndarray,
    loss: str = "mse",
    steps: int = 20,
    lr: float = 0.05,
    alpha_lr: float | None = None,
    uv_lr: float | None = None,
    bias_lr: float | None = None,
    optimize_bias: bool = False,
    optimize_uv: bool = False,
    seed: int = 0,
    progress_interval: int = 0,
    progress_fn: Callable[[int, "SpectralNetwork"], None] | None = None,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    debug: bool = False,
    batch_size: int | None = None,
    optimizer_name: str = "adam",
) -> float:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    alpha_parameters: list[torch.nn.Parameter] = []
    u_parameters: list[torch.Tensor] = []
    v_parameters: list[torch.Tensor] = []
    bias_parameters: list[torch.Tensor] = []
    uv_trainable_parameters: list[torch.nn.Parameter] = []
    bias_trainable_parameters: list[torch.nn.Parameter] = []
    residual_tensors: list[torch.Tensor] = []

    params: list[torch.nn.Parameter] = []
    for layer in network.layers:
        residual_tensors.append(torch.tensor(layer.residual.astype(float), dtype=dtype, device=device))
        alpha_param = torch.nn.Parameter(torch.tensor(layer.alpha.astype(float), dtype=dtype, device=device))
        params.append(alpha_param)
        alpha_parameters.append(alpha_param)

        if optimize_uv:
            u_param = torch.nn.Parameter(torch.tensor(layer.u.astype(float), dtype=dtype, device=device))
            v_param = torch.nn.Parameter(torch.tensor(layer.v.astype(float), dtype=dtype, device=device))
            params.extend([u_param, v_param])
            uv_trainable_parameters.extend([u_param, v_param])
        else:
            u_param = torch.tensor(layer.u.astype(float), dtype=dtype, device=device)
            v_param = torch.tensor(layer.v.astype(float), dtype=dtype, device=device)

        u_parameters.append(u_param)
        v_parameters.append(v_param)

    for bias in network.biases:
        if optimize_bias:
            bias_param = torch.nn.Parameter(torch.tensor(bias.astype(float), dtype=dtype, device=device))
            params.append(bias_param)
            bias_trainable_parameters.append(bias_param)
            bias_parameters.append(bias_param)
        else:
            bias_parameters.append(torch.tensor(bias.astype(float), dtype=dtype, device=device))

    param_groups = [{"params": alpha_parameters, "lr": lr if alpha_lr is None else alpha_lr}]
    if uv_trainable_parameters:
        param_groups.append({"params": uv_trainable_parameters, "lr": lr if uv_lr is None else uv_lr})
    if bias_trainable_parameters:
        param_groups.append({"params": bias_trainable_parameters, "lr": lr if bias_lr is None else bias_lr})
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")
    inputs_t = torch.tensor(inputs.astype(float), dtype=dtype, device=device)
    # targets: keep as long for cross-entropy, else float for regression/bce
    if loss == "ce":
        targets_t = torch.tensor(targets.reshape(-1), dtype=torch.long, device=device)
    else:
        targets_t = torch.tensor(targets.astype(float), dtype=dtype, device=device)
    if batch_size is not None and batch_size > 0 and batch_size < inputs_t.shape[0]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset = TensorDataset(inputs_t.detach().cpu(), targets_t.detach().cpu())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        batches = iter(loader)
    else:
        batches = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    for step in range(1, steps + 1):
        if batches is None:
            batch_inputs = inputs_t
            batch_targets = targets_t
        else:
            try:
                batch_inputs, batch_targets = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch_inputs, batch_targets = next(batches)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions = _forward_spectral_torch(
            batch_inputs,
            network,
            alpha_parameters,
            u_parameters,
            v_parameters,
            bias_parameters,
            residual_tensors,
            loss,
        )
        loss_val = torch_loss(loss, predictions, batch_targets)
        loss_val.backward()
        # optional gradient clipping to stabilize updates
        if grad_clip is not None and grad_clip > 0:
            try:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            except Exception:
                pass
        optimizer.step()

        if optimize_uv:
            with torch.no_grad():
                for idx, layer in enumerate(network.layers):
                    for mode_idx in range(layer.rank):
                        u_vec = u_parameters[idx][:, mode_idx]
                        v_vec = v_parameters[idx][:, mode_idx]
                        u_norm = torch.norm(u_vec)
                        v_norm = torch.norm(v_vec)
                        if u_norm > 0 and v_norm > 0:
                            scale = u_norm * v_norm
                            u_parameters[idx][:, mode_idx] = u_vec / u_norm
                            v_parameters[idx][:, mode_idx] = v_vec / v_norm
                            alpha_parameters[idx][mode_idx] = alpha_parameters[idx][mode_idx] * scale

        if progress_fn is not None and progress_interval > 0 and step % progress_interval == 0:
            # materialize current params back to network for evaluation
            for idx, layer in enumerate(network.layers):
                layer.alpha = alpha_parameters[idx].detach().cpu().numpy().copy()
                if optimize_uv:
                    layer.u = u_parameters[idx].detach().cpu().numpy().copy()
                    layer.v = v_parameters[idx].detach().cpu().numpy().copy()
            if optimize_bias:
                for idx, bias_param in enumerate(bias_parameters):
                    if getattr(bias_param, "requires_grad", False):
                        network.biases[idx] = bias_param.detach().cpu().numpy().copy()

            if debug:
                try:
                    grad_norms = []
                    param_norms = []
                    for p in params:
                        if isinstance(p, torch.nn.Parameter):
                            g = p.grad
                            grad_norms.append(0.0 if g is None else float(g.norm().item()))
                            param_norms.append(float(p.data.norm().item()))
                    if grad_norms:
                        avg_grad = sum(grad_norms) / len(grad_norms)
                        avg_param = sum(param_norms) / len(param_norms)
                        print(f"[debug] step={step} avg_grad_norm={avg_grad:.6e} avg_param_norm={avg_param:.6e}", flush=True)
                except Exception:
                    pass

            progress_fn(step, network)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_duration = time.perf_counter() - train_start

    for idx, layer in enumerate(network.layers):
        layer.alpha = alpha_parameters[idx].detach().cpu().numpy().copy()
        if optimize_uv:
            layer.u = u_parameters[idx].detach().cpu().numpy().copy()
            layer.v = v_parameters[idx].detach().cpu().numpy().copy()
    if optimize_bias:
        for idx, bias_param in enumerate(bias_parameters):
            if bias_param.requires_grad:
                network.biases[idx] = bias_param.detach().cpu().numpy().copy()

    return train_duration


def _forward_dense_torch(
    x: torch.Tensor,
    weight_parameters: list[torch.nn.Parameter],
    bias_parameters: list[torch.nn.Parameter],
    loss: str,
) -> torch.Tensor:
    for idx, (weight_t, bias_t) in enumerate(zip(weight_parameters, bias_parameters)):
        x = x @ weight_t.T + bias_t
        if idx < len(weight_parameters) - 1:
            x = torch.maximum(x, torch.zeros_like(x))
        elif loss in {"bce", "balanced_bce"}:
            x = torch.sigmoid(x)
    return x


def optimize_dense_parameters(
    network: "SpectralNetwork",
    inputs: np.ndarray,
    targets: np.ndarray,
    loss: str = "mse",
    steps: int = 20,
    lr: float = 0.01,
    optimize_bias: bool = True,
    seed: int = 0,
    progress_interval: int = 0,
    progress_fn: Callable[[int, "SpectralNetwork"], None] | None = None,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    batch_size: int | None = None,
    optimizer_name: str = "adam",
) -> float:
    """Lamarckian dense-weight GD refinement for a spectral network.

    The network is optimized as ordinary dense weight matrices, then each learned
    matrix is decomposed back into the existing spectral rank. This keeps the
    evolution loop LEMONADE-like: children inherit learned weights, local search
    is ordinary GD, and SVD coordinates are still available for crossover.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    ranks = [layer.rank for layer in network.layers]
    weight_parameters = [
        torch.nn.Parameter(torch.tensor(reconstruct_layer(layer).astype(float), dtype=dtype, device=device))
        for layer in network.layers
    ]
    bias_parameters: list[torch.nn.Parameter] = []
    for bias in network.biases:
        parameter = torch.nn.Parameter(torch.tensor(bias.astype(float), dtype=dtype, device=device), requires_grad=optimize_bias)
        bias_parameters.append(parameter)

    params: list[torch.nn.Parameter] = [*weight_parameters]
    if optimize_bias:
        params.extend(bias_parameters)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")

    inputs_t = torch.tensor(inputs.astype(float), dtype=dtype, device=device)
    if loss == "ce":
        targets_t = torch.tensor(targets.reshape(-1), dtype=torch.long, device=device)
    else:
        targets_t = torch.tensor(targets.astype(float), dtype=dtype, device=device)

    if batch_size is not None and batch_size > 0 and batch_size < inputs_t.shape[0]:
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset = TensorDataset(inputs_t.detach().cpu(), targets_t.detach().cpu())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
        batches = iter(loader)
    else:
        batches = None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    def materialize() -> None:
        for idx, weight_param in enumerate(weight_parameters):
            network.layers[idx] = decompose_dense(
                weight_param.detach().cpu().numpy().copy(),
                ranks[idx],
                keep_residual=False,
            )
        if optimize_bias:
            for idx, bias_param in enumerate(bias_parameters):
                network.biases[idx] = bias_param.detach().cpu().numpy().copy()

    for step in range(1, steps + 1):
        if batches is None:
            batch_inputs = inputs_t
            batch_targets = targets_t
        else:
            try:
                batch_inputs, batch_targets = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch_inputs, batch_targets = next(batches)
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

        optimizer.zero_grad()
        predictions = _forward_dense_torch(batch_inputs, weight_parameters, bias_parameters, loss)
        loss_val = torch_loss(loss, predictions, batch_targets)
        loss_val.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()

        if progress_fn is not None and progress_interval > 0 and step % progress_interval == 0:
            materialize()
            progress_fn(step, network)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_duration = time.perf_counter() - train_start
    materialize()
    return train_duration


def optimize_alphas(*args, **kwargs) -> float:
    """Backward-compatible alias for alpha-only callers.

    Prefer optimize_spectral_parameters when optimize_uv=True is used.
    """
    return optimize_spectral_parameters(*args, **kwargs)
