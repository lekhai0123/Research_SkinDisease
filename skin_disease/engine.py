import numpy as np
import torch
from tqdm.auto import tqdm

from .cutmix import prepare_batch
from .distillation import kd_kl_loss


def forward_loss(
    model,
    images,
    labels,
    criterion,
    labels_a=None,
    labels_b=None,
    lam=1.0,
    use_amp=True,
    teacher_model=None,
    kd_alpha=0.0,
    kd_temperature=4.0,
):
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.amp.autocast(amp_device, enabled=use_amp and torch.cuda.is_available()):
        outputs = model(images)

        if labels_a is not None and labels_b is not None and lam != 1.0:
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
        else:
            loss = criterion(outputs, labels)

        if teacher_model is not None and kd_alpha > 0.0:
            with torch.no_grad():
                t_out = teacher_model(images)
            kd = kd_kl_loss(outputs, t_out, temperature=kd_temperature)
            loss = (1.0 - kd_alpha) * loss + kd_alpha * kd

    return outputs, loss


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_cutmix=False,
    cutmix_alpha=1.0,
    use_sam=False,
    teacher_model=None,
    kd_alpha=0.0,
    kd_temperature=4.0,
):
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0

    pbar = tqdm(loader, leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, labels_a, labels_b, lam = prepare_batch(
            images=images,
            labels=labels,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha
        )

        optimizer.zero_grad(set_to_none=True)

        if not use_sam:
            outputs, loss = forward_loss(
                model=model,
                images=images,
                labels=labels,
                criterion=criterion,
                labels_a=labels_a if use_cutmix else None,
                labels_b=labels_b if use_cutmix else None,
                lam=lam,
                use_amp=True,
                teacher_model=teacher_model,
                kd_alpha=kd_alpha,
                kd_temperature=kd_temperature,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            if use_cutmix:
                correct += lam * (preds == labels_a).sum().item() + (1.0 - lam) * (preds == labels_b).sum().item()
            else:
                correct += (preds == labels).sum().item()

            total += labels.size(0)
            pbar.set_postfix(loss=running_loss / total, acc=correct / total)
            continue

        outputs_1, loss_1 = forward_loss(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
            labels_a=labels_a if use_cutmix else None,
            labels_b=labels_b if use_cutmix else None,
            lam=lam,
            use_amp=False,
            teacher_model=teacher_model,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )

        if not torch.isfinite(loss_1):
            raise RuntimeError(f"Non-finite loss detected at SAM first pass: {loss_1.item()}")

        loss_1.backward()
        optimizer.first_step(zero_grad=True)

        _, loss_2 = forward_loss(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
            labels_a=labels_a if use_cutmix else None,
            labels_b=labels_b if use_cutmix else None,
            lam=lam,
            use_amp=False,
            teacher_model=teacher_model,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )

        if not torch.isfinite(loss_2):
            raise RuntimeError(f"Non-finite loss detected at SAM second pass: {loss_2.item()}")

        loss_2.backward()
        optimizer.second_step(zero_grad=False)
        optimizer.base_optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += loss_1.item() * images.size(0)
        preds = outputs_1.argmax(dim=1)

        if use_cutmix:
            correct += lam * (preds == labels_a).sum().item() + (1.0 - lam) * (preds == labels_b).sum().item()
        else:
            correct += (preds == labels).sum().item()

        total += labels.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        pbar = tqdm(loader, leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(amp_device, enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite eval loss detected: {loss.item()}")

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total, np.array(all_labels), np.array(all_preds)
