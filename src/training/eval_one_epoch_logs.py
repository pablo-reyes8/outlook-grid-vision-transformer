import time
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.metrics import * 
from src.training.autocast import *

import copy

def format_ops(x: float, kind: str = "FLOPs") -> str:
    if not (x > 0):
        return "N/A"

    if x >= 1e12:
        return f"{x/1e12:.2f} T{kind}"
    elif x >= 1e9:
        return f"{x/1e9:.2f} G{kind}"
    elif x >= 1e6:
        return f"{x/1e6:.2f} M{kind}"
    else:
        return f"{x:.2e} {kind}"



def _count_params_and_size_mib(model: nn.Module) -> Tuple[int, float]:
    n_params = sum(p.numel() for p in model.parameters())
    # parámetros únicamente (sin buffers). Asume p.element_size() correcto.
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return n_params, n_bytes / (1024**2)

def _try_estimate_flops(model: nn.Module, example_inputs: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            flops = float(FlopCountAnalysis(model, example_inputs).total())
        # Convención común: FLOPs ≈ 2 * MACs
        macs = flops / 2.0
        return {"flops": flops, "macs": macs}
    except Exception:
        pass

    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            macs, _params = profile(model, inputs=(example_inputs,), verbose=False)
        return {"flops": float(macs) * 2.0, "macs": float(macs)}
    except Exception:
        pass

    return {"flops": None, "macs": None}

@torch.no_grad()
def evaluate_one_epoch_logs(
    model: nn.Module,
    dataloader,
    device: str = "cuda",
    autocast_dtype: str = "bf16",
    use_amp: bool = True,
    label_smoothing: float = 0.0,
    channels_last: bool = False,
    measure_flops: bool = True,
    flops_warmup_batches: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """
    Single-process evaluation loop (no DDP, no EMA).

    Expects helpers already defined in your file:
      - autocast_ctx
      - accuracy_topk

    Adds performance metrics:
      - imgs_per_sec
      - ms_per_batch
      - gpu_mem_allocated_mib, gpu_mem_reserved_mib, gpu_mem_peak_allocated_mib
      - model_params, model_param_size_mib
      - flops_per_forward (optional), macs_per_forward (optional)
    """
    model.eval().to(device)

    # --- Model size stats ---
    model_params, model_param_size_mib = _count_params_and_size_mib(model)

    # --- GPU memory baseline / peak ---
    is_cuda = (device.startswith("cuda") and torch.cuda.is_available())
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_alloc_start = torch.cuda.memory_allocated()
        mem_reserved_start = torch.cuda.memory_reserved()
    else:
        mem_alloc_start = mem_reserved_start = 0

    running_loss = 0.0
    total = 0
    c1 = c3 = c5 = 0.0

    # --- Timing stats ---
    n_batches = 0
    total_batch_time = 0.0  # seconds, measured with sync for CUDA

    # --- FLOPs estimate (optional; computed once on first real batch) ---
    flops_info = {"flops": None, "macs": None}
    flops_done = False

    # global start (for imgs/sec too)
    if is_cuda:
        torch.cuda.synchronize()
    t_epoch0 = time.perf_counter()

    for b, (images, targets) in enumerate(dataloader):
        n_batches += 1

        # Warmup timing (optional): skip a few batches to avoid first-iteration overhead
        do_time = True
        if b < flops_warmup_batches:
            do_time = False

        if is_cuda and do_time:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

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

        # FLOPs/MACs estimate using this batch (once)
        if measure_flops and (not flops_done):
            try:
                model_cpu = copy.deepcopy(model).eval().cpu()
                ex = images[:1].detach().float().cpu()
                flops_info = _try_estimate_flops(model_cpu, ex)
            finally:
                flops_done = True

        if is_cuda and do_time:
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if do_time:
            total_batch_time += (t1 - t0)

    if is_cuda:
        torch.cuda.synchronize()
    t_epoch1 = time.perf_counter()

    avg_loss = running_loss / max(1, total)
    metrics = {
        "top1": 100.0 * c1 / max(1, total),
        "top3": 100.0 * c3 / max(1, total),
        "top5": 100.0 * c5 / max(1, total),}

    # --- Throughput ---
    # imgs/sec: usa tiempo total del epoch 
    epoch_time = max(1e-12, (t_epoch1 - t_epoch0))
    imgs_per_sec = float(total) / epoch_time

    timed_batches = max(0, n_batches - flops_warmup_batches)
    if timed_batches > 0:
        ms_per_batch = 1000.0 * (total_batch_time / timed_batches)
    else:
        ms_per_batch = None

    if is_cuda:
        mem_alloc_end = torch.cuda.memory_allocated()
        mem_reserved_end = torch.cuda.memory_reserved()
        mem_peak = torch.cuda.max_memory_allocated()
        gpu_mem_allocated_mib = mem_alloc_end / (1024**2)
        gpu_mem_reserved_mib = mem_reserved_end / (1024**2)
        gpu_mem_peak_allocated_mib = mem_peak / (1024**2)
        gpu_mem_alloc_delta_mib = (mem_alloc_end - mem_alloc_start) / (1024**2)
        gpu_mem_reserved_delta_mib = (mem_reserved_end - mem_reserved_start) / (1024**2)
    else:
        gpu_mem_allocated_mib = gpu_mem_reserved_mib = gpu_mem_peak_allocated_mib = 0.0
        gpu_mem_alloc_delta_mib = gpu_mem_reserved_delta_mib = 0.0

    metrics.update({
        "imgs_per_sec": imgs_per_sec,
        "epoch_time_sec": float(epoch_time),
        "ms_per_batch": float(ms_per_batch) if ms_per_batch is not None else float("nan"),

        "model_params": float(model_params),
        "model_param_size_mib": float(model_param_size_mib),

        "gpu_mem_allocated_mib": float(gpu_mem_allocated_mib),
        "gpu_mem_reserved_mib": float(gpu_mem_reserved_mib),
        "gpu_mem_peak_allocated_mib": float(gpu_mem_peak_allocated_mib),
        "gpu_mem_alloc_delta_mib": float(gpu_mem_alloc_delta_mib),
        "gpu_mem_reserved_delta_mib": float(gpu_mem_reserved_delta_mib),

        "flops_per_forward": float(flops_info["flops"]) if flops_info["flops"] is not None else float("nan"),
        "macs_per_forward": float(flops_info["macs"]) if flops_info["macs"] is not None else float("nan"),
    })

    return avg_loss, metrics
