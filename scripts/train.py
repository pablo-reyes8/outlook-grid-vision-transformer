import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Model_A_OutGridNet import OutlookerFrontGridNet
from src.Model_B_OutGridNet import MaxOutNet
from src.data.load_cifrar100 import get_cifar100_dataloaders
from src.model.downsampling import DownsampleConfig
from src.stage_config import StageCfg
from src.training.autocast import seed_everything
from src.training.train_full_model import train_model


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_stages(stage_cfgs: list[dict]) -> list[StageCfg]:
    return [StageCfg(**cfg) for cfg in stage_cfgs]


def build_model(model_cfg: dict) -> torch.nn.Module:
    model_type = str(model_cfg.get("type", "model_b")).lower()
    stages = build_stages(model_cfg.get("stages", []))
    if not stages:
        raise ValueError("model.stages must have at least one stage config")

    down_cfg = DownsampleConfig(**model_cfg.get("downsample", {}))

    common = dict(
        num_classes=int(model_cfg.get("num_classes", 100)),
        stages=stages,
        in_ch=int(model_cfg.get("in_ch", 3)),
        stem_dim=int(model_cfg.get("stem_dim", 64)),
        dpr_max=float(model_cfg.get("dpr_max", 0.1)),
        down_cfg=down_cfg,
    )

    if model_type in ("a", "model_a", "outlooker_front"):
        return OutlookerFrontGridNet(
            outlooker_front_depth=int(model_cfg.get("outlooker_front_depth", 2)),
            **common,
        )
    if model_type in ("b", "model_b", "outgrid"):
        return MaxOutNet(**common)

    raise ValueError(f"Unknown model.type '{model_type}'. Use 'model_a' or 'model_b'.")


def build_dataloaders(data_cfg: dict, num_classes: int):
    dataset = str(data_cfg.get("dataset", "cifar100")).lower()
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 2))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    if dataset == "cifar100":
        return get_cifar100_dataloaders(
            batch_size=batch_size,
            data_dir=str(data_cfg.get("data_dir", "./data")),
            num_workers=num_workers,
            val_split=float(data_cfg.get("val_split", 0.0)),
            pin_memory=pin_memory,
            ra_num_ops=int(data_cfg.get("ra_num_ops", 2)),
            ra_magnitude=int(data_cfg.get("ra_magnitude", 7)),
            random_erasing_p=float(data_cfg.get("random_erasing_p", 0.25)),
            img_size=int(data_cfg.get("img_size", 32)),
        )

    if dataset == "synthetic":
        num_samples = int(data_cfg.get("num_samples", 256))
        img_size = int(data_cfg.get("img_size", 32))
        images = torch.randn(num_samples, 3, img_size, img_size)
        labels = torch.randint(0, num_classes, (num_samples,))
        ds = torch.utils.data.TensorDataset(images, labels)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        return loader, None, None

    raise ValueError("data.dataset must be 'cifar100' or 'synthetic'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Outlook-Grid models")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML config")
    parser.add_argument("--model", choices=["a", "b", "model_a", "model_b"], help="Override model type")
    parser.add_argument("--device", help="Override runtime device")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--data-dir", help="Override dataset root")
    parser.add_argument("--num-workers", type=int, help="Override dataloader workers")
    parser.add_argument("--img-size", type=int, help="Override input image size")
    parser.add_argument("--val-split", type=float, help="Override val split (0..1)")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--resume", help="Path to resume checkpoint")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--seed", type=int, help="Override random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})

    if args.model:
        model_cfg["type"] = args.model
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.data_dir is not None:
        data_cfg["data_dir"] = args.data_dir
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.img_size is not None:
        data_cfg["img_size"] = args.img_size
    if args.val_split is not None:
        data_cfg["val_split"] = args.val_split
    if args.device is not None:
        runtime_cfg["device"] = args.device
    if args.output_dir is not None:
        runtime_cfg["output_dir"] = args.output_dir
    if args.resume is not None:
        train_cfg["resume_path"] = args.resume
    if args.no_amp:
        train_cfg["use_amp"] = False
    if args.seed is not None:
        runtime_cfg["seed"] = args.seed

    seed_everything(int(runtime_cfg.get("seed", 7)), deterministic=False)

    device = str(runtime_cfg.get("device", "cuda")).lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    output_dir = Path(runtime_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_cfg)
    num_classes = int(model_cfg.get("num_classes", 100))

    train_loader, val_loader, _ = build_dataloaders(data_cfg, num_classes)

    save_path = Path(train_cfg.get("save_path", "best_model.pt"))
    last_path = Path(train_cfg.get("last_path", "last_model.pt"))
    if not save_path.is_absolute():
        save_path = output_dir / save_path
    if not last_path.is_absolute():
        last_path = output_dir / last_path

    history, _ = train_model(
        model=model,
        train_loader=train_loader,
        epochs=int(train_cfg.get("epochs", 1)),
        val_loader=val_loader,
        device=device,
        lr=float(train_cfg.get("lr", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
        autocast_dtype=str(train_cfg.get("autocast_dtype", "fp16")),
        use_amp=bool(train_cfg.get("use_amp", True)),
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        min_lr=float(train_cfg.get("min_lr", 0.0)),
        label_smoothing=float(train_cfg.get("label_smoothing", 0.1)),
        print_every=int(train_cfg.get("print_every", 100)),
        save_path=str(save_path),
        last_path=str(last_path),
        resume_path=train_cfg.get("resume_path", None),
        mixup_alpha=float(train_cfg.get("mixup_alpha", 0.0)),
        cutmix_alpha=float(train_cfg.get("cutmix_alpha", 0.0)),
        mix_prob=float(train_cfg.get("mix_prob", 1.0)),
        num_classes=num_classes,
        channels_last=bool(train_cfg.get("channels_last", False)),
        early_stop=bool(train_cfg.get("early_stop", True)),
        early_stop_metric=str(train_cfg.get("early_stop_metric", "top1")),
        early_stop_patience=int(train_cfg.get("early_stop_patience", 10)),
        early_stop_min_delta=float(train_cfg.get("early_stop_min_delta", 0.0)),
        early_stop_require_monotonic=bool(train_cfg.get("early_stop_require_monotonic", False)),
    )

    print("Training complete. History keys:", sorted(history.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
