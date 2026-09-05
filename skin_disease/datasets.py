import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
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
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list
    num_classes: int
    data_mode: str


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
    data_mode="final",
    val_fraction=0.125,
    seed=42,
):
    """Build the Skin31 dataloaders under the paper's two-phase evaluation protocol.

    data_mode="dev": carves a stratified validation split out of the public 80%
    training partition (val_fraction=0.125 of it, i.e. 10% of the whole dataset,
    giving an overall 70/10/20 train/val/test partition) for hyperparameter
    selection. The public 20% test partition is untouched but is not evaluated
    here - selection must never see it.

    data_mode="final": once hyperparameters are frozen, trains on the full
    original 80% partition (no validation split) and evaluates on the public
    20% test partition, reproducing the original 80/20 split.

    In both modes the training set is a concatenation of the same images passed
    through each named augmentation (original, center-zoom, rotation, brightness,
    shear, vertical flip, horizontal flip), then class-balanced via a weighted
    sampler so that every disease category is seen with roughly equal frequency
    despite the long-tailed class distribution.
    """
    if data_mode not in ("dev", "final"):
        raise ValueError(f"data_mode must be 'dev' or 'final', got: {data_mode}")

    eval_tf = build_eval_transform(img_size)
    aug_transforms = build_augmentation_transforms(img_size)

    full_train_base = datasets.ImageFolder(train_dir)
    class_names = full_train_base.classes
    num_classes = len(class_names)

    if data_mode == "dev":
        all_paths = np.array([path for path, _ in full_train_base.samples])
        all_targets = np.array(full_train_base.targets)

        train_paths, val_paths = train_test_split(
            all_paths,
            test_size=val_fraction,
            random_state=seed,
            stratify=all_targets,
        )
        train_paths = set(map(str, train_paths))
        val_paths = set(map(str, val_paths))

        train_excluded_paths = val_paths
        val_dataset = FilteredImageFolder(train_dir, transform=eval_tf, exclude_paths=train_paths)
        base_train_refined = FilteredImageFolder(train_dir, transform=None, exclude_paths=val_paths)
    else:
        train_excluded_paths = set()
        val_dataset = None
        base_train_refined = FilteredImageFolder(train_dir, transform=None, exclude_paths=[])

    train_sets = [
        FilteredImageFolder(train_dir, transform=tf, exclude_paths=train_excluded_paths)
        for tf in aug_transforms.values()
    ]
    train_dataset = ConcatDataset(train_sets)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)

    base_targets = np.array(base_train_refined.targets)
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

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    n_val = len(val_dataset) if val_dataset is not None else 0
    print(
        f"Skin31 [{data_mode}]: {len(train_dataset)} train (augmented) / "
        f"{n_val} val / {len(test_dataset)} test, {num_classes} classes"
    )

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Skin31Data(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        num_classes=num_classes,
        data_mode=data_mode,
    )
