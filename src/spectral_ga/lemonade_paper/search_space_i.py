from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PaperBlock:
    block_id: int
    op: str
    out_channels: int
    skip_from: int | None = None


@dataclass(frozen=True)
class PaperCNNArch:
    stages: tuple[tuple[PaperBlock, ...], tuple[PaperBlock, ...], tuple[PaperBlock, ...]]
    next_block_id: int


def arch_id(arch: PaperCNNArch) -> str:
    parts = []
    for stage in arch.stages:
        blocks = []
        for block in stage:
            op = "S" if block.op == "sepconv" else "C"
            skip = f"r{block.skip_from}" if block.skip_from is not None else "-"
            blocks.append(f"{block.block_id}{op}{block.out_channels}{skip}")
        parts.append(".".join(blocks))
    return "ss1_" + "__".join(parts)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, op: str) -> None:
        super().__init__()
        if op == "sepconv":
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
            )
        elif op == "conv":
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        else:
            raise ValueError(f"Unsupported block op: {op}")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvexMerge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(-20.0))

    def forward(self, current: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.logit)
        return (1.0 - lam) * current + lam * skip


class PaperLemonadeCNN(nn.Module):
    def __init__(self, arch: PaperCNNArch, num_classes: int = 10) -> None:
        super().__init__()
        self.arch = arch
        self.blocks = nn.ModuleDict()
        self.merges = nn.ModuleDict()
        in_channels = 3
        for stage in arch.stages:
            for block in stage:
                self.blocks[str(block.block_id)] = ConvBNReLU(in_channels, block.out_channels, block.op)
                if block.skip_from is not None:
                    self.merges[str(block.block_id)] = ConvexMerge()
                in_channels = block.out_channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activations: dict[int, torch.Tensor] = {}
        for stage_index, stage in enumerate(self.arch.stages):
            for block in stage:
                out = self.blocks[str(block.block_id)](x)
                if block.skip_from is not None and block.skip_from in activations:
                    skip = activations[block.skip_from]
                    if skip.shape == out.shape:
                        out = self.merges[str(block.block_id)](out, skip)
                activations[block.block_id] = out
                x = out
            if stage_index < 2:
                x = nn.functional.max_pool2d(x, 2)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


MIN_PARAMETER_LOWER_BOUND = 10_000


def initial_arches() -> list[PaperCNNArch]:
    specs = [
        ("sepconv", (48, 96, 96)),
        ("sepconv", (88, 176, 176)),
        ("conv", (32, 64, 128)),
        ("conv", (64, 128, 256)),
    ]
    arches = []
    for op, channels in specs:
        block_id = 0
        stages = []
        for out_channels in channels:
            stages.append((PaperBlock(block_id, op, out_channels),))
            block_id += 1
        arches.append(PaperCNNArch(tuple(stages), block_id))  # type: ignore[arg-type]
    return arches


def cheap_objectives(arch: PaperCNNArch) -> tuple[float]:
    model = PaperLemonadeCNN(arch)
    return (math.log(float(count_params(model))),)


def satisfies_parameter_lower_bound(arch: PaperCNNArch) -> bool:
    return count_params(PaperLemonadeCNN(arch)) >= MIN_PARAMETER_LOWER_BOUND


def all_blocks(arch: PaperCNNArch) -> list[tuple[int, int, PaperBlock]]:
    return [(stage_index, block_index, block) for stage_index, stage in enumerate(arch.stages) for block_index, block in enumerate(stage)]


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
    target.zero_()
    target[slices].copy_(source[slices].to(device=target.device, dtype=target.dtype))
    return int(math.prod(part.stop for part in slices))


def _initialize_new_state_value(name: str, target: torch.Tensor) -> None:
    if name.endswith("num_batches_tracked"):
        target.zero_()
    elif name.endswith("running_var"):
        target.fill_(1.0)
    elif name.endswith("running_mean") or name.endswith(".bias"):
        target.zero_()
    elif name.endswith(".bn.weight"):
        target.fill_(1.0)
    elif ".conv" in name and name.endswith("weight"):
        target.zero_()
        if target.ndim == 4 and target.shape[2] == target.shape[3]:
            center = target.shape[2] // 2
            if target.shape[1] == 1:
                target[:, 0, center, center] = 1.0
            else:
                torch.nn.init.dirac_(target)
    elif name.endswith("logit"):
        target.fill_(-20.0)


def warmstart_model(model: nn.Module, parent_state: dict[str, torch.Tensor] | None) -> int:
    if parent_state is None:
        return 0
    child_state = model.state_dict()
    copied_values = 0
    with torch.no_grad():
        for name, target in child_state.items():
            source = parent_state.get(name)
            if source is not None:
                copied_values += copy_tensor_overlap(target, source)
            else:
                _initialize_new_state_value(name, target)
    model.load_state_dict(child_state)
    return copied_values


def snapshot_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _replace_stage(arch: PaperCNNArch, stage_index: int, blocks: Iterable[PaperBlock]) -> PaperCNNArch:
    stages = [tuple(stage) for stage in arch.stages]
    stages[stage_index] = tuple(blocks)
    return PaperCNNArch((stages[0], stages[1], stages[2]), arch.next_block_id)


def insert_conv_bn_relu(arch: PaperCNNArch, rng: random.Random) -> PaperCNNArch:
    stage_index = rng.randrange(3)
    stage = list(arch.stages[stage_index])
    insert_after = rng.randrange(len(stage))
    prev_channels = stage[insert_after].out_channels
    op = stage[insert_after].op
    new_block = PaperBlock(arch.next_block_id, op, prev_channels)
    stage.insert(insert_after + 1, new_block)
    updated = _replace_stage(arch, stage_index, stage)
    return PaperCNNArch(updated.stages, arch.next_block_id + 1)


def increase_filters(arch: PaperCNNArch, rng: random.Random) -> PaperCNNArch:
    blocks = all_blocks(arch)
    stage_index, block_index, block = rng.choice(blocks)
    factor = rng.choice([1.25, 1.5, 2.0])
    out_channels = int(math.ceil(block.out_channels * factor / 8.0)) * 8
    stage = list(arch.stages[stage_index])
    stage[block_index] = PaperBlock(block.block_id, block.op, out_channels, block.skip_from)
    return _replace_stage(arch, stage_index, stage)


def add_skip_connection(arch: PaperCNNArch, rng: random.Random) -> PaperCNNArch:
    stages = [list(stage) for stage in arch.stages]
    candidates: list[tuple[int, int, int]] = []
    for stage_index, stage in enumerate(stages):
        for target_index, target in enumerate(stage):
            for source in stage[:target_index]:
                if source.out_channels == target.out_channels and target.skip_from is None:
                    candidates.append((stage_index, target_index, source.block_id))
    if not candidates:
        return arch
    stage_index, target_index, source_id = rng.choice(candidates)
    block = stages[stage_index][target_index]
    stages[stage_index][target_index] = PaperBlock(block.block_id, block.op, block.out_channels, source_id)
    return PaperCNNArch((tuple(stages[0]), tuple(stages[1]), tuple(stages[2])), arch.next_block_id)


def remove_layer_or_skip(arch: PaperCNNArch, rng: random.Random) -> PaperCNNArch:
    skip_candidates = [(s, b, block) for s, b, block in all_blocks(arch) if block.skip_from is not None]
    removable = [(s, b, block) for s, b, block in all_blocks(arch) if len(arch.stages[s]) > 1]
    if skip_candidates and (not removable or rng.random() < 0.5):
        stage_index, block_index, block = rng.choice(skip_candidates)
        stage = list(arch.stages[stage_index])
        stage[block_index] = PaperBlock(block.block_id, block.op, block.out_channels, None)
        return _replace_stage(arch, stage_index, stage)
    if not removable:
        return arch
    stage_index, block_index, block = rng.choice(removable)
    stage = [item for index, item in enumerate(arch.stages[stage_index]) if index != block_index]
    stage = [PaperBlock(item.block_id, item.op, item.out_channels, None if item.skip_from == block.block_id else item.skip_from) for item in stage]
    return _replace_stage(arch, stage_index, stage)


def prune_filters(arch: PaperCNNArch, rng: random.Random, min_filters: int = 16) -> PaperCNNArch:
    candidates = [(s, b, block) for s, b, block in all_blocks(arch) if block.out_channels > min_filters]
    if not candidates:
        return arch
    stage_index, block_index, block = rng.choice(candidates)
    divisor = rng.choice([2, 4])
    out_channels = max(min_filters, int(math.ceil((block.out_channels / divisor) / 8.0)) * 8)
    stage = list(arch.stages[stage_index])
    stage[block_index] = PaperBlock(block.block_id, block.op, out_channels, None)
    return _replace_stage(arch, stage_index, stage)


def replace_conv_with_sepconv(arch: PaperCNNArch, rng: random.Random) -> PaperCNNArch:
    candidates = [(s, b, block) for s, b, block in all_blocks(arch) if block.op == "conv"]
    if not candidates:
        return arch
    stage_index, block_index, block = rng.choice(candidates)
    stage = list(arch.stages[stage_index])
    stage[block_index] = PaperBlock(block.block_id, "sepconv", block.out_channels, block.skip_from)
    return _replace_stage(arch, stage_index, stage)


NETWORK_MORPHISMS = (insert_conv_bn_relu, increase_filters, add_skip_connection)
APPROXIMATE_NETWORK_MORPHISMS = (remove_layer_or_skip, prune_filters, replace_conv_with_sepconv)


def mutate_arch(arch: PaperCNNArch, rng: random.Random) -> tuple[PaperCNNArch, list[str], bool]:
    child = copy.deepcopy(arch)
    applied: list[str] = []
    needs_distillation = False
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(NETWORK_MORPHISMS + APPROXIMATE_NETWORK_MORPHISMS)
        before = arch_id(child)
        candidate = op(child, rng)
        if arch_id(candidate) == before or not satisfies_parameter_lower_bound(candidate):
            continue
        child = candidate
        applied.append(op.__name__)
        if op in APPROXIMATE_NETWORK_MORPHISMS:
            needs_distillation = True
    if not applied:
        child = insert_conv_bn_relu(child, rng)
        applied.append("insert_conv_bn_relu")
    return child, applied, needs_distillation




