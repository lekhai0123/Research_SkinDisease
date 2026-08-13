import gc
import json
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn

from .engine import evaluate, train_one_epoch
from .models import efficientnet_b2_dual_level, efficientnet_b2_original
from .sam import SAM


def build_model(model_name, model_variant, num_classes, drop_rate=0.0, pretrained=True, fusion_channels=512, device=None):
    if model_name != "efficientnet_b2":
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate
        )
    elif model_variant == "original":
        model = efficientnet_b2_original(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate
        )
    elif model_variant == "dual_level":
        model = efficientnet_b2_dual_level(
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate,
            fusion_channels=fusion_channels
        )
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    return model.to(device) if device is not None else model


def build_optimizer(model, optimizer_name, lr, weight_decay, use_sam=False, sam_rho=0.05, sam_adaptive=False):
    if not use_sam:
        if optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    if optimizer_name == "adam":
        return SAM(model.parameters(), torch.optim.Adam, rho=sam_rho, adaptive=sam_adaptive, lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adamw":
        return SAM(model.parameters(), torch.optim.AdamW, rho=sam_rho, adaptive=sam_adaptive, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer, scheduler_name, epochs):
    target_optimizer = optimizer.base_optimizer if hasattr(optimizer, "base_optimizer") else optimizer

    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(target_optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def slugify_exp_name(exp_name):
    return exp_name.replace(" ", "_").replace("+", "plus").lower()


def save_history_files(exp_name, history, history_dir):
    history_dir = Path(history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    exp_slug = slugify_exp_name(exp_name)
    hist_df = pd.DataFrame(history)
    csv_path = history_dir / f"{exp_slug}_history.csv"
    json_path = history_dir / f"{exp_slug}_history.json"
    hist_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def save_summary_files(all_results, work_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(all_results)
    csv_path = work_dir / "experiment_summary.csv"
    json_path = work_dir / "experiment_summary.json"
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return summary_df, csv_path, json_path


def append_summary_result(result, work_dir):
    """Append one run_experiment() result onto work_dir/experiment_summary.csv.

    Useful for accumulating results across separate CLI invocations, e.g. the
    three knowledge-distillation seeds (42, 123, 2024) reported in the paper.
    """
    work_dir = Path(work_dir)
    summary_path = work_dir / "experiment_summary.csv"
    existing = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    existing.append(result)
    summary_df, _, _ = save_summary_files(existing, work_dir)
    return summary_df


def run_experiment(
    exp_name,
    model_name,
    model_variant,
    num_classes,
    train_loader,
    test_loader,
    device,
    work_dir,
    history_dir,
    epochs=15,
    optimizer_name="adam",
    scheduler_name="cosine",
    lr=5e-4,
    weight_decay=1e-6,
    label_smoothing=0.0,
    use_cutmix=False,
    cutmix_alpha=1.0,
    use_sam=False,
    sam_rho=0.05,
    sam_adaptive=False,
    drop_rate=0.0,
    fusion_channels=512,
    use_kd=False,
    kd_teacher_path=None,
    kd_teacher_fusion_channels=512,
    kd_alpha=0.7,
    kd_temperature=4.0,
):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Running: {exp_name} | model={model_name} | variant={model_variant} | lr={lr} | sam={use_sam} | kd={use_kd}")
    print("=" * 80)

    exp_slug = slugify_exp_name(exp_name)
    save_path = work_dir / f"{exp_slug}_best.pth"

    model = build_model(
        model_name=model_name,
        model_variant=model_variant,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained=True,
        fusion_channels=fusion_channels,
        device=device,
    )

    teacher_model = None
    if use_kd and kd_teacher_path is not None:
        state = torch.load(kd_teacher_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]

        teacher_model = efficientnet_b2_dual_level(
            num_classes=num_classes,
            pretrained=False,
            fusion_channels=kd_teacher_fusion_channels,
        )
        missing, unexpected = teacher_model.load_state_dict(state, strict=False)
        print(f"[KD] Teacher loaded. missing={len(missing)} unexpected={len(unexpected)}")
        teacher_model = teacher_model.to(device).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        use_sam=use_sam,
        sam_rho=sam_rho,
        sam_adaptive=sam_adaptive
    )
    scheduler = build_scheduler(optimizer, scheduler_name, epochs)

    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=torch.cuda.is_available())

    best_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_cutmix=use_cutmix,
            cutmix_alpha=cutmix_alpha,
            use_sam=use_sam,
            teacher_model=teacher_model,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
        )

        val_loss, val_acc, y_true, y_pred = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "experiment": exp_name,
            "model_name": model_name,
            "model_variant": model_variant,
            "optimizer": optimizer_name,
            "scheduler": scheduler_name,
            "lr": float(lr),
            "label_smoothing": float(label_smoothing),
            "use_cutmix": bool(use_cutmix),
            "cutmix_alpha": float(cutmix_alpha),
            "use_sam": bool(use_sam),
            "sam_rho": float(sam_rho),
            "sam_adaptive": bool(sam_adaptive),
            "drop_rate": float(drop_rate),
            "fusion_channels": int(fusion_channels),
            "use_kd": bool(use_kd),
            "kd_teacher_path": str(kd_teacher_path) if kd_teacher_path is not None else None,
            "kd_alpha": float(kd_alpha),
            "kd_temperature": float(kd_temperature),
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "test_loss": float(val_loss),
            "test_acc": float(val_acc),
        }
        history.append(row)

        save_history_files(exp_name, history, history_dir)

        print(
            f"[{exp_name}] "
            f"Epoch {epoch:02d}/{epochs} | "
            f"lr={lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={val_loss:.4f} | "
            f"test_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = float(val_acc)
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

    print(f"[{exp_name}] Best test accuracy: {best_acc:.4f} at epoch {best_epoch}")
    print(f"[{exp_name}] Saved best model to: {save_path}")

    del model, criterion, optimizer, scheduler, scaler
    if teacher_model is not None:
        del teacher_model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "experiment": exp_name,
        "model_name": model_name,
        "model_variant": model_variant,
        "optimizer": optimizer_name,
        "scheduler": scheduler_name,
        "lr": float(lr),
        "label_smoothing": float(label_smoothing),
        "use_cutmix": bool(use_cutmix),
        "cutmix_alpha": float(cutmix_alpha),
        "use_sam": bool(use_sam),
        "sam_rho": float(sam_rho),
        "sam_adaptive": bool(sam_adaptive),
        "drop_rate": float(drop_rate),
        "fusion_channels": int(fusion_channels),
        "use_kd": bool(use_kd),
        "kd_teacher_path": str(kd_teacher_path) if kd_teacher_path is not None else None,
        "kd_alpha": float(kd_alpha),
        "kd_temperature": float(kd_temperature),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "save_path": str(save_path),
        "status": "ok"
    }
