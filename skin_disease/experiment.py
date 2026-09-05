import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from .engine import evaluate, train_one_epoch
from .metrics import compute_full_metrics
from .models import efficientnet_b2_dual_level, efficientnet_b2_original
from .sam import SAM

SUMMARY_LEAD_COLS = [
    "experiment", "model_name", "model_variant", "data_mode", "status",
    "best_epoch", "final_epoch",
    "best_val_acc", "best_val_balanced_acc", "best_val_macro_f1",
    "test_acc", "test_balanced_acc", "test_macro_f1",
    "test_weighted_f1", "test_macro_precision", "test_macro_recall",
    "test_loss",
]


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


def _order_summary_columns(df):
    lead = [c for c in SUMMARY_LEAD_COLS if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def save_summary_files(all_results, work_dir):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_df = _order_summary_columns(pd.DataFrame(all_results))
    csv_path = work_dir / "experiment_summary.csv"
    json_path = work_dir / "experiment_summary.json"
    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
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
    class_names,
    train_loader,
    test_loader,
    device,
    work_dir,
    history_dir,
    val_loader=None,
    data_mode="final",
    select_metric="val_acc",
    log_test_each_epoch=True,
    epochs=15,
    optimizer_name="adamw",
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
    kd_teacher_variant="dual_level",
    kd_teacher_fusion_channels=512,
    kd_alpha=0.7,
    kd_temperature=4.0,
):
    """Train one configuration under the paper's two-phase protocol.

    data_mode="dev": model selection is done on `val_loader` only (never on
    `test_loader`) - the checkpoint with the best `select_metric` on the
    validation split is kept, and the test set is evaluated exactly once at
    the end, purely to report a number that played no part in selection.

    data_mode="final": hyperparameters are already frozen, so there is no
    validation split and no checkpoint selection - training runs for the
    fixed `epochs` budget and the final-epoch weights are evaluated once on
    the test set. `test_loader` may still be evaluated every epoch (when
    log_test_each_epoch=True) for transparency in the history file, but that
    per-epoch number never influences which weights are kept.
    """
    if data_mode not in ("dev", "final"):
        raise ValueError(f"data_mode must be 'dev' or 'final', got: {data_mode}")
    if data_mode == "dev" and val_loader is None:
        raise ValueError("data_mode='dev' requires a val_loader")

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    is_dev_mode = data_mode == "dev"
    exp_slug = slugify_exp_name(exp_name)
    save_path = work_dir / f"{exp_slug}_{'best' if is_dev_mode else 'final'}.pth"

    print("=" * 90)
    print(f"Running: {exp_name} | model={model_name} | variant={model_variant} | data_mode={data_mode}")
    print(f"  lr={lr} | sam={use_sam}({sam_rho}) | cutmix={use_cutmix} | ls={label_smoothing} | kd={use_kd}")
    print("=" * 90)

    config = {
        "experiment": exp_name,
        "model_name": model_name,
        "model_variant": model_variant,
        "data_mode": data_mode,
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
        "kd_teacher_variant": kd_teacher_variant if use_kd else None,
        "kd_alpha": float(kd_alpha),
        "kd_temperature": float(kd_temperature),
        "select_metric": select_metric if is_dev_mode else None,
        "epochs": int(epochs),
    }

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

        teacher_model = build_model(
            model_name="efficientnet_b2",
            model_variant=kd_teacher_variant,
            num_classes=num_classes,
            pretrained=False,
            fusion_channels=kd_teacher_fusion_channels,
        )
        missing, unexpected = teacher_model.load_state_dict(state, strict=False)
        print(f"[KD] Teacher ({kd_teacher_variant}) loaded. missing={len(missing)} unexpected={len(unexpected)}")
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

    best_score = -float("inf")
    best_epoch = 0
    best_val_metrics = {}
    history = []

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

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

        if is_dev_mode:
            val_loss, _, y_true_val, y_pred_val = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_full_metrics(y_true_val, y_pred_val, num_classes, prefix="val_")
            val_metrics["val_loss"] = float(val_loss)

            row = {**config, "epoch": epoch, "lr_epoch": float(current_lr),
                   "train_loss": float(train_loss), "train_acc": float(train_acc), **val_metrics}

            score = val_metrics.get(select_metric, val_metrics["val_acc"])
            if score > best_score:
                best_score = float(score)
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                torch.save(model.state_dict(), save_path)
                print(f"    -> new best ({select_metric}={best_score:.4f})")

            print(
                f"[{exp_name}] Epoch {epoch:02d}/{epochs} | lr={current_lr:.2e} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} || "
                f"val_loss={val_metrics['val_loss']:.4f} val_acc={val_metrics['val_acc']:.4f} "
                f"val_bal={val_metrics['val_balanced_acc']:.4f} val_f1={val_metrics['val_macro_f1']:.4f}"
            )
        else:
            epoch_test_metrics = {}
            if log_test_each_epoch:
                test_loss_epoch, _, y_true_test, y_pred_test = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
                epoch_test_metrics = compute_full_metrics(y_true_test, y_pred_test, num_classes, prefix="test_")
                epoch_test_metrics["test_loss"] = float(test_loss_epoch)

            row = {**config, "epoch": epoch, "lr_epoch": float(current_lr),
                   "train_loss": float(train_loss), "train_acc": float(train_acc), **epoch_test_metrics}

            if log_test_each_epoch:
                print(
                    f"[{exp_name}] Epoch {epoch:02d}/{epochs} | lr={current_lr:.2e} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} || "
                    f"test_loss={epoch_test_metrics['test_loss']:.4f} test_acc={epoch_test_metrics['test_acc']:.4f} "
                    f"test_bal={epoch_test_metrics['test_balanced_acc']:.4f} test_f1={epoch_test_metrics['test_macro_f1']:.4f}"
                )
            else:
                print(f"[{exp_name}] Epoch {epoch:02d}/{epochs} | lr={current_lr:.2e} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

        history.append(row)
        save_history_files(exp_name, history, history_dir)

    if is_dev_mode:
        print(f"[{exp_name}] Best VAL {select_metric} = {best_score:.4f} @ epoch {best_epoch}")
    else:
        best_epoch = epochs
        torch.save(model.state_dict(), save_path)
        print(f"[{exp_name}] FINAL MODEL SAVED @ fixed epoch {epochs} (no checkpoint selection)")

    del model, criterion, optimizer, scheduler, scaler
    if teacher_model is not None:
        del teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Single, final test-set evaluation: the best-validation checkpoint in dev
    # mode, or the fixed final-epoch checkpoint in final mode. Either way this
    # is the only time test_loader is used for anything but transparency logging.
    test_loss = None
    test_metrics = {f"test_{k}": None for k in
                     ["acc", "balanced_acc", "macro_f1", "macro_precision", "macro_recall", "weighted_f1"]}
    y_true_test = y_pred_test = None

    if is_dev_mode and best_epoch == 0:
        print(f"[{exp_name}] WARNING: no best-validation checkpoint was saved, skipping final test evaluation.")
    else:
        eval_model = build_model(
            model_name=model_name, model_variant=model_variant, num_classes=num_classes,
            drop_rate=drop_rate, pretrained=False, fusion_channels=fusion_channels,
        )
        eval_model.load_state_dict(torch.load(save_path, map_location=device), strict=True)
        eval_model = eval_model.to(device)

        test_criterion = nn.CrossEntropyLoss()
        test_loss, _, y_true_test, y_pred_test = evaluate(eval_model, test_loader, test_criterion, device)
        test_loss = float(test_loss)
        test_metrics = compute_full_metrics(y_true_test, y_pred_test, num_classes, prefix="test_")

        label = f"best-validation (epoch {best_epoch})" if is_dev_mode else f"fixed epoch {epochs}"
        print("-" * 90)
        print(f"[{exp_name}] FINAL TEST ({label})")
        print(f"  acc={test_metrics['test_acc']:.4f}  balanced_acc={test_metrics['test_balanced_acc']:.4f}  "
              f"macro_f1={test_metrics['test_macro_f1']:.4f}  loss={test_loss:.4f}")
        print("-" * 90)
        del eval_model, test_criterion

    test_report_path = None
    test_confusion_path = None
    if y_true_test is not None and y_pred_test is not None:
        labels_idx = list(range(num_classes))
        report = classification_report(
            y_true_test, y_pred_test, labels=labels_idx, target_names=class_names,
            digits=4, zero_division=0, output_dict=True
        )
        test_report_path = work_dir / f"{exp_slug}_test_report.json"
        with open(test_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        test_confusion_path = work_dir / f"{exp_slug}_test_confusion.npy"
        np.save(test_confusion_path, confusion_matrix(y_true_test, y_pred_test, labels=labels_idx))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        **config,
        "best_epoch": int(best_epoch) if is_dev_mode and best_epoch > 0 else None,
        "final_epoch": int(epochs) if not is_dev_mode else None,
        "best_val_score": float(best_score) if is_dev_mode and best_epoch > 0 else None,
        "best_val_acc": best_val_metrics.get("val_acc") if is_dev_mode else None,
        "best_val_balanced_acc": best_val_metrics.get("val_balanced_acc") if is_dev_mode else None,
        "best_val_macro_f1": best_val_metrics.get("val_macro_f1") if is_dev_mode else None,
        "best_val_weighted_f1": best_val_metrics.get("val_weighted_f1") if is_dev_mode else None,
        "best_val_loss": best_val_metrics.get("val_loss") if is_dev_mode else None,
        **test_metrics,
        "test_loss": test_loss,
        "model_selection": f"best_{select_metric}" if is_dev_mode else "fixed_last_epoch",
        "test_report_path": str(test_report_path) if test_report_path else None,
        "test_confusion_path": str(test_confusion_path) if test_confusion_path else None,
        "save_path": str(save_path),
        "status": "ok",
    }
