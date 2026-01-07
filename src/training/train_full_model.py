import time
from typing import Dict, Tuple

import torch 
import torch.nn as nn 

from src.training.one_epoch_train import *
from src.training.chekpoints import * 


def train_model(
    model: nn.Module,
    train_loader,
    epochs: int,
    val_loader=None,
    device: str = "cuda",
    lr: float = 5e-4,
    weight_decay: float = 0.05,
    autocast_dtype: str = "fp16",
    use_amp: bool = True,
    grad_clip_norm: float | None = 1.0,
    warmup_ratio: float = 0.05,
    min_lr: float = 0.0,
    label_smoothing: float = 0.1,
    print_every: int = 100,
    save_path: str = "best_model.pt",
    last_path: str = "last_model.pt",
    resume_path: str | None = None,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    channels_last: bool = False,
    early_stop: bool = True,
    early_stop_metric: str = "top1",          # "top1" or "loss"
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.0,
    early_stop_require_monotonic: bool = False,) -> Tuple[Dict[str, list], nn.Module]:

    """
    Single-process trainer (no DDP, no EMA).

    Expects helpers already defined in your file:
      - build_param_groups_no_wd
      - WarmupCosineLR
      - make_grad_scaler
      - save_checkpoint / load_checkpoint
      - train_one_epoch (the one above)
      - evaluate_one_epoch
    """
    model.to(device)

    # Optimizer
    param_groups = build_param_groups_no_wd(model, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler warmup + cosine (step-based)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = WarmupCosineLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr)

    # AMP scaler
    scaler = None
    if use_amp and autocast_dtype.lower() in ("fp16", "float16"):
        scaler = make_grad_scaler(device=device, enabled=True)

    # Resume
    start_epoch = 0
    best_val_top1 = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    if resume_path is not None:
        ckpt = load_checkpoint(
            resume_path, model,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            map_location=device,
            strict=True,)

        start_epoch = int(ckpt.get("epoch", 0))
        best_val_top1 = float(ckpt.get("best_top1", best_val_top1))
        extra = ckpt.get("extra", {}) or {}
        best_val_loss = float(extra.get("best_val_loss", best_val_loss))
        best_epoch = int(extra.get("best_epoch", best_epoch))
        print(f"Resumed from {resume_path} at epoch {start_epoch} | best_top1 {best_val_top1:.2f}% | best_loss {best_val_loss:.4f}")

    history = {
        "train_loss": [], "train_top1": [], "train_top3": [], "train_top5": [],
        "val_loss": [], "val_top1": [], "val_top3": [], "val_top5": [],
        "lr": [],}

    metric = early_stop_metric.lower()
    assert metric in ("top1", "loss")
    mode = "max" if metric == "top1" else "min"
    best_metric = best_val_top1 if metric == "top1" else best_val_loss
    patience = int(early_stop_patience)
    bad_epochs = 0
    last_vals: list[float] = []

    def _is_improvement(curr: float, best: float) -> bool:
        d = float(early_stop_min_delta)
        return (curr > (best + d)) if mode == "max" else (curr < (best - d))

    def _degradation_monotonic(vals: list[float]) -> bool:
        if not early_stop_require_monotonic or len(vals) < 2:
            return True
        if mode == "max":
            return all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
        else:
            return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        t_epoch = time.time()

        # If a sampler supports set_epoch, reshuffle deterministically per epoch (works even without DDP)
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        if val_loader is not None and hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(epoch)

        # --- Train ---
        tr_loss, tr_m = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
            num_classes=num_classes,
            channels_last=channels_last,
            print_every=print_every,)

        history["train_loss"].append(tr_loss)
        history["train_top1"].append(tr_m["top1"])
        history["train_top3"].append(tr_m["top3"])
        history["train_top5"].append(tr_m["top5"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"[Train] loss {tr_loss:.4f} | top1 {tr_m['top1']:.2f}% | top3 {tr_m['top3']:.2f}% | "
            f"top5 {tr_m['top5']:.2f}% | lr {optimizer.param_groups[0]['lr']:.2e}")

        # Save "last" checkpoint every epoch
        save_checkpoint(
            last_path, model, optimizer, scheduler, scaler,
            epoch=epoch, best_top1=best_val_top1,
            extra={
                "autocast_dtype": autocast_dtype,
                "use_amp": use_amp,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "early_stop_metric": metric,
                "early_stop_patience": patience,
                "early_stop_min_delta": float(early_stop_min_delta),},)

        stop_now = False

        # --- Val ---
        if val_loader is not None:
            va_loss, va_m = evaluate_one_epoch(
                model=model,
                dataloader=val_loader,
                device=device,
                autocast_dtype=autocast_dtype,
                use_amp=use_amp,
                label_smoothing=0.0,
                channels_last=channels_last,)

            history["val_loss"].append(va_loss)
            history["val_top1"].append(va_m["top1"])
            history["val_top3"].append(va_m["top3"])
            history["val_top5"].append(va_m["top5"])

            print(
                f"[Val]   loss {va_loss:.4f} | top1 {va_m['top1']:.2f}% | top3 {va_m['top3']:.2f}% | top5 {va_m['top5']:.2f}%")

            # Best checkpoint by val_top1
            if va_m["top1"] > best_val_top1:
                best_val_top1 = float(va_m["top1"])
                if va_loss < best_val_loss:
                    best_val_loss = float(va_loss)
                    best_epoch = int(epoch)

                save_checkpoint(
                    save_path, model, optimizer, scheduler, scaler,
                    epoch=epoch, best_top1=best_val_top1,
                    extra={
                        "autocast_dtype": autocast_dtype,
                        "use_amp": use_amp,
                        "best_val_loss": best_val_loss,
                        "best_epoch": best_epoch,},)

                print(f"Best saved to {save_path} (val top1 {best_val_top1:.2f}%)")

            # Early stop on chosen metric
            if early_stop:
                curr_metric = float(va_m["top1"]) if metric == "top1" else float(va_loss)

                last_vals.append(curr_metric)
                if len(last_vals) > patience:
                    last_vals = last_vals[-patience:]

                if _is_improvement(curr_metric, best_metric):
                    best_metric = curr_metric
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience and _degradation_monotonic(last_vals):
                    print(f"Early-stop: no improvement on val_{metric} for {patience} epochs.")
                    stop_now = True

        if stop_now:
            break

        dt = time.time() - t_epoch
        print(f"Epoch time: {dt/60:.2f} min")

    return history, model
