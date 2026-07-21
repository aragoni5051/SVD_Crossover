import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.retrain_cifar10_lemonade_arches import (
    CifarTrainDataset,
    apply_cutout,
    mixup_batch,
    parse_arch_id,
    random_crop_flip,
)


def test_parse_arch_id_round_trips_stage_fields() -> None:
    arch = parse_arch_id("ch56-48-200_d2-1-2_CCS_RR-")

    assert arch.channels == (56, 48, 200)
    assert arch.depths == (2, 1, 2)
    assert arch.separable == (False, False, True)
    assert arch.skips == (True, True, False)


def test_random_crop_flip_preserves_cifar_shape() -> None:
    image = torch.rand(3, 32, 32)

    augmented = random_crop_flip(image)

    assert augmented.shape == image.shape


def test_cutout_masks_pixels_without_changing_shape() -> None:
    torch.manual_seed(0)
    image = torch.ones(3, 32, 32)

    augmented = apply_cutout(image, 16)

    assert augmented.shape == image.shape
    assert torch.count_nonzero(augmented == 0.0) > 0


def test_mixup_batch_returns_mixed_targets_and_inputs() -> None:
    torch.manual_seed(1)
    inputs = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).reshape(4, 3, 2, 2)
    labels = torch.tensor([0, 1, 2, 3])

    mixed, target_a, target_b, lam = mixup_batch(inputs, labels, alpha=1.0)

    assert mixed.shape == inputs.shape
    assert torch.equal(target_a, labels)
    assert target_b.shape == labels.shape
    assert 0.0 <= lam <= 1.0


def test_cifar_train_dataset_applies_optional_augmentation() -> None:
    torch.manual_seed(2)
    images = torch.ones(2, 3, 32, 32)
    labels = torch.tensor([1, 2])
    dataset = CifarTrainDataset(images, labels, augment=True, cutout_length=8)

    image, label = dataset[0]

    assert image.shape == (3, 32, 32)
    assert int(label) == 1
