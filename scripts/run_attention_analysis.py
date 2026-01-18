import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_dataloaders, build_model, load_yaml
from src.experiments.heat_map_att_grid import plot_grid_attention_random
from src.experiments.heat_map_att_outlooker import plot_outlooker_locality_random
from src.experiments.mad_metrics import compute_grid_and_outlooker_mad_by_stage
from src.training.chekpoints import load_checkpoint


DATA_STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "svhn": ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    "tinyimagenet200": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def _parse_stages(value: str, max_stage: int) -> tuple[int, ...]:
    v = value.strip().lower()
    if v == "all":
        return tuple(range(max_stage))
    items = [int(x) for x in value.split(",") if x.strip()]
    return tuple(i for i in items if 0 <= i < max_stage)


def _pick_loader(split: str, train_loader, val_loader, test_loader):
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader or train_loader
    if split == "test":
        return test_loader or val_loader or train_loader
    raise ValueError("split must be train, val, or test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention visualizations + MAD metrics")
    parser.add_argument("--config", default="configs/cifar100_model_a.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default="analysis_outputs")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--stages", default="0,1,2,3")
    parser.add_argument("--block-idx", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--skip-outlooker", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--skip-mad", action="store_true")
    parser.add_argument("--show", action="store_true")

    parser.add_argument("--gy", type=int, default=0)
    parser.add_argument("--gx", type=int, default=0)
    parser.add_argument("--no-normalize-grid", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    device = str(args.device).lower()
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    model = build_model(model_cfg).to(device).eval()

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, optimizer=None, scheduler=None, scaler=None, map_location=device)

    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, int(model_cfg.get("num_classes", 100)))
    loader = _pick_loader(args.split, train_loader, val_loader, test_loader)

    stages = _parse_stages(args.stages, max_stage=len(model_cfg.get("stages", [])))
    stage_depths = [int(s.get("depth", 0)) for s in model_cfg.get("stages", [])]

    dataset_key = str(data_cfg.get("dataset", "cifar100")).lower()
    mean, std = DATA_STATS.get(dataset_key, DATA_STATS["cifar100"])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_outlooker:
        outlooker_dir = out_dir / "outlooker"
        plot_outlooker_locality_random(
            model=model,
            loader=loader,
            device=device,
            save_dir=str(outlooker_dir),
            n_images=args.n_images,
            stages=stages,
            block_idx=args.block_idx,
            mean=mean,
            std=std,
            seed=args.seed,
            show=args.show,
        )

    if not args.skip_grid:
        grid_dir = out_dir / "grid"
        plot_grid_attention_random(
            model=model,
            loader=loader,
            device=device,
            save_dir=str(grid_dir),
            n_images=args.n_images,
            stages=stages,
            block_idx=args.block_idx,
            stage_depths=stage_depths,
            mean=mean,
            std=std,
            seed=args.seed,
            show=args.show,
        )

    if not args.skip_mad:
        results = compute_grid_and_outlooker_mad_by_stage(
            model=model,
            loader=loader,
            block_idx=args.block_idx,
            stages=stages,
            n_images=args.n_images,
            seed=args.seed,
            device=device,
            gy=args.gy,
            gx=args.gx,
            normalize_grid=not args.no_normalize_grid,
        )

        json_path = out_dir / "mad_metrics.json"
        csv_path = out_dir / "mad_metrics.csv"

        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        if results:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)

        print(f"MAD saved to: {json_path}")
        print(f"MAD CSV saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
