import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from download_dataset import download_dataset
from skin_disease import config
from skin_disease.datasets import build_skin31_dataloaders, save_class_meta
from skin_disease.experiment import append_summary_result, run_experiment
from skin_disease.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train an EffSkinNet EfficientNet-B2/B0 model on Skin31.")

    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None, help="Skin31 root (containing train/ and test/). Auto-downloaded into data/ via kagglehub if omitted.")
    parser.add_argument("--work-dir", type=Path, default=config.DEFAULT_WORK_DIR)
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    parser.add_argument("--img-size", type=int, default=config.DEFAULT_IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.DEFAULT_NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)

    parser.add_argument("--model-name", choices=["efficientnet_b2", "efficientnet_b0"], default="efficientnet_b2")
    parser.add_argument("--model-variant", choices=["original", "dual_level"], default="original")
    parser.add_argument("--fusion-channels", type=int, default=512)
    parser.add_argument("--drop-rate", type=float, default=0.0)

    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=config.DEFAULT_WEIGHT_DECAY)

    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--use-cutmix", action="store_true")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)

    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--sam-adaptive", action="store_true")

    parser.add_argument("--use-kd", action="store_true")
    parser.add_argument("--kd-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--kd-teacher-fusion-channels", type=int, default=512)
    parser.add_argument("--kd-alpha", type=float, default=0.7)
    parser.add_argument("--kd-temperature", type=float, default=4.0)

    return parser.parse_args()


def resolve_dataset_root(dataset_root_arg):
    if dataset_root_arg is not None:
        return dataset_root_arg
    print("--dataset-root not set, downloading via kagglehub into data/ (cached after the first run)...")
    return download_dataset()


def main():
    args = parse_args()

    if args.use_kd and args.kd_teacher_checkpoint is None:
        raise ValueError("--kd-teacher-checkpoint is required when --use-kd is set")

    device = set_seed(args.seed)

    dataset_root = resolve_dataset_root(args.dataset_root)
    print(f"Using dataset root: {dataset_root}")

    data = build_skin31_dataloaders(
        train_dir=dataset_root / "train",
        test_dir=dataset_root / "test",
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    save_class_meta(
        args.work_dir / "class_names_skin31.json",
        dataset_name="skin31",
        num_classes=data.num_classes,
        class_names=data.class_names,
    )

    result = run_experiment(
        exp_name=args.exp_name,
        model_name=args.model_name,
        model_variant=args.model_variant,
        num_classes=data.num_classes,
        train_loader=data.train_loader,
        test_loader=data.test_loader,
        device=device,
        work_dir=args.work_dir,
        history_dir=args.work_dir / "histories",
        epochs=args.epochs,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        use_cutmix=args.use_cutmix,
        cutmix_alpha=args.cutmix_alpha,
        use_sam=args.use_sam,
        sam_rho=args.sam_rho,
        sam_adaptive=args.sam_adaptive,
        drop_rate=args.drop_rate,
        fusion_channels=args.fusion_channels,
        use_kd=args.use_kd,
        kd_teacher_path=args.kd_teacher_checkpoint,
        kd_teacher_fusion_channels=args.kd_teacher_fusion_channels,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
    )

    append_summary_result(result, work_dir=args.work_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
