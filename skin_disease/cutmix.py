import numpy as np
import torch


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_cutmix(images, labels, alpha=1.0):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(images.size(0), device=images.device)

    labels_a = labels
    labels_b = labels[rand_index]

    x1, y1, x2, y2 = rand_bbox(images.size(), lam)
    images = images.clone()
    images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))

    return images, labels_a, labels_b, lam


def prepare_batch(images, labels, use_cutmix=False, cutmix_alpha=1.0):
    if use_cutmix:
        images, labels_a, labels_b, lam = apply_cutmix(images, labels, alpha=cutmix_alpha)
        return images, labels_a, labels_b, lam
    return images, labels, labels, 1.0
