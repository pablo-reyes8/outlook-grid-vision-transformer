import time 
from typing import Optional, Dict, Tuple, Any

import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.training.autocast import * 
from src.training.cutmix_mixup_aug import * 
from src.training.warmup import * 
from src.training.metrics import *

def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cuda",
    scaler=None,
    autocast_dtype: str = "bf16",
    use_amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    mix_prob: float = 1.0,
    num_classes: int = 100,
    channels_last: bool = False,
    print_every: int = 100,) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Single-process train loop (no DDP, no EMA) + instrumentation.

    Expects helpers already defined in your file:
      - autocast_ctx
      - apply_mixup_cutmix
      - soft_target_cross_entropy
      - accuracy_topk

    Returns:
      avg_loss, metrics(top1/top3/top5), extra(stats for logging)
    """
    model.train()

    # Only use GradScaler for fp16 
    use_scaler = (scaler is not None) and use_amp and autocast_dtype.lower() in ("fp16", "float16")

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    # instrumentation meters 
    grad_norm_sum = 0.0
    grad_norm_count = 0
    clip_steps = 0
    overflow_steps = 0
    nonfinite_loss_steps = 0

    data_time_sum = 0.0
    iter_time_sum = 0.0

    # timing
    t_epoch0 = time.time()
    t_data = time.time()

    for step, (images, targets) in enumerate(dataloader, start=1):
        t0 = time.time()
        data_time_sum += (t0 - t_data)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        B = targets.size(0)

        # mixup/cutmix => soft targets
        images_aug, targets_soft = apply_mixup_cutmix(
            images, targets,
            num_classes=num_classes,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mix_prob,)

        optimizer.zero_grad(set_to_none=True)

        # forward under autocast
        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images_aug)  # [B, K]

        # loss in fp32
        if (mixup_alpha > 0.0) or (cutmix_alpha > 0.0):
            # With mixup/cutmix, label smoothing is usually redundant.
            loss = soft_target_cross_entropy(logits.float(), targets_soft)
        else:
            loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

        # guard non-finite loss
        if not torch.isfinite(loss):
            nonfinite_loss_steps += 1
            optimizer.zero_grad(set_to_none=True)
            # if fp16 scaler is used, count as overflow-like
            if use_scaler:
                overflow_steps += 1

            iter_time_sum += (time.time() - t0)
            t_data = time.time()
            continue

        # backward + step (with grad norm + clipping + overflow detection)
        if use_scaler:
            scale_before = float(scaler.get_scale())

            scaler.scale(loss).backward()

            # unscale before grad operations
            scaler.unscale_(optimizer)

            # compute grad norm + clip
            if grad_clip_norm is not None:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if float(gnorm) > float(grad_clip_norm):
                    clip_steps += 1
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            grad_norm_sum += float(gnorm)
            grad_norm_count += 1

            scaler.step(optimizer)
            scaler.update()

            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                overflow_steps += 1

        else:
            loss.backward()

            if grad_clip_norm is not None:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if float(gnorm) > float(grad_clip_norm):
                    clip_steps += 1
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

            grad_norm_sum += float(gnorm)
            grad_norm_count += 1

            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # metrics
        running_loss += float(loss.item()) * B
        total += B

        accs = accuracy_topk(
            logits.detach(),
            targets_soft if targets_soft.ndim == 2 else targets,
            ks=(1, 3, 5),)

        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

        # iter time
        iter_time_sum += (time.time() - t0)

        # logging
        if print_every and (step % print_every == 0 or step == len(dataloader)):
            dt = time.time() - t_epoch0
            imgs_sec = total / max(dt, 1e-9)

            gnorm_avg = grad_norm_sum / max(1, grad_norm_count)
            clip_pct = 100.0 * clip_steps / max(1, grad_norm_count)
            scale_now = float(scaler.get_scale()) if use_scaler else 1.0

            print(
                f"[train step {step}/{len(dataloader)}] "
                f"loss {running_loss/total:.4f} | "
                f"top1 {100*c1/total:.2f}% | top3 {100*c3/total:.2f}% | top5 {100*c5/total:.2f}% | "
                f"{imgs_sec:.1f} img/s | lr {optimizer.param_groups[0]['lr']:.2e} | "
                f"gnorm {gnorm_avg:.3f} | clip {clip_pct:.1f}% | "
                f"oflow {overflow_steps} | nonfinite {nonfinite_loss_steps} | scale {scale_now:.1f}")

        # reset for next loop
        t_data = time.time()

    avg_loss = running_loss / max(1, total)
    metrics = {
        "top1": 100.0 * c1 / max(1, total),
        "top3": 100.0 * c3 / max(1, total),
        "top5": 100.0 * c5 / max(1, total),}

    extra = {
        "grad_norm_avg": float(grad_norm_sum / max(1, grad_norm_count)),
        "clip_frac": float(clip_steps / max(1, grad_norm_count)),
        "amp_overflow_steps": float(overflow_steps),
        "nonfinite_loss_steps": float(nonfinite_loss_steps),
        "scaler_scale": float(scaler.get_scale()) if use_scaler else 1.0,
        "data_time_s_per_step": float(data_time_sum / max(1, len(dataloader))),
        "iter_time_s_per_step": float(iter_time_sum / max(1, len(dataloader))),}

    return avg_loss, metrics, extra


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "bf16",
    use_amp: bool = True,
    label_smoothing: float = 0.0,
    channels_last: bool = False) -> Tuple[float, Dict[str, float]]:
    """
    Single-process evaluation loop (no DDP, no EMA).

    Expects helpers already defined in your file:
      - autocast_ctx
      - accuracy_topk
    """
    model.eval().to(device)

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        B = targets.size(0)

        with autocast_ctx(device=device, enabled=use_amp, dtype=autocast_dtype, cache_enabled=True):
            logits = model(images)

        loss = F.cross_entropy(logits.float(), targets, label_smoothing=label_smoothing)

        running_loss += loss.item() * B
        total += B

        accs = accuracy_topk(logits, targets, ks=(1, 3, 5))
        c1 += accs[1] * B / 100.0
        c3 += accs[3] * B / 100.0
        c5 += accs[5] * B / 100.0

    avg_loss = running_loss / max(1, total)
    metrics = {
        "top1": 100.0 * c1 / max(1, total),
        "top3": 100.0 * c3 / max(1, total),
        "top5": 100.0 * c5 / max(1, total),}

    return avg_loss, metrics
