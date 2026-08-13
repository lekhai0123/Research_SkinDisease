import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import datasets

from .transforms import build_augmentation_transforms, build_eval_transform


class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, exclude_paths=None):
        super().__init__(root=root, transform=transform)
        exclude_paths = {str(Path(p)) for p in (exclude_paths or [])}

        filtered_samples = []
        filtered_targets = []

        for path, target in self.samples:
            if str(Path(path)) not in exclude_paths:
                filtered_samples.append((path, target))
                filtered_targets.append(target)

        self.samples = filtered_samples
        self.imgs = filtered_samples
        self.targets = filtered_targets


@dataclass
class Skin31Data:
    train_loader: DataLoader
    test_loader: DataLoader
    class_names: list
    num_classes: int


def save_class_meta(path, dataset_name, num_classes, class_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": dataset_name,
            "num_classes": num_classes,
            "class_names": class_names
        }, f, ensure_ascii=False, indent=2)


def build_skin31_dataloaders(
    train_dir,
    test_dir,
    img_size=224,
    batch_size=32,
    num_workers=0,
    exclude_paths=None,
):
    """Build the balanced Skin31 train/test dataloaders.

    The training set is a concatenation of the same images passed through each
    named augmentation (original, center-zoom, rotation, brightness, shear,
    vertical flip, horizontal flip), then class-balanced via a weighted sampler
    so that every disease category is seen with roughly equal frequency despite
    the long-tailed class distribution.
    """
    eval_tf = build_eval_transform(img_size)
    aug_transforms = build_augmentation_transforms(img_size)

    base_train = FilteredImageFolder(train_dir, transform=None, exclude_paths=exclude_paths)
    class_names = base_train.classes
    num_classes = len(class_names)
    base_targets = np.array(base_train.targets)

    train_sets = [
        FilteredImageFolder(train_dir, transform=tf, exclude_paths=exclude_paths)
        for tf in aug_transforms.values()
    ]
    train_dataset = ConcatDataset(train_sets)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_counts = np.bincount(base_targets, minlength=num_classes)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask]
    sample_weights = class_weights[base_targets]
    sample_weights = np.tile(sample_weights, len(train_sets))

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Skin31: {len(train_dataset)} train / {len(test_dataset)} test, {num_classes} classes")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Skin31Data(
        train_loader=train_loader,
        test_loader=test_loader,
        class_names=class_names,
        num_classes=num_classes
    )
