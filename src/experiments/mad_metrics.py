import os, re
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.heat_map_att_outlooker import *
from src.experiments.heat_map_att_grid import *

def _pick_q_indices_from_attn(attn_mean: torch.Tensor, Hg: int, Wg: int):
    """
    attn_mean: [N, N] (promedio sobre heads)
    Returns:
      q_center: token central
      q_max: query con mayor pico (max de su fila)
      q_min: query con menor pico
    """
    N = Hg * Wg
    assert attn_mean.shape == (N, N), f"Expected {(N,N)} got {tuple(attn_mean.shape)}"

    q_center = (Hg // 2) * Wg + (Wg // 2)

    row_peak = attn_mean.max(dim=-1).values
    q_max = int(row_peak.argmax().item())
    q_min = int(row_peak.argmin().item())
    return q_center, q_max, q_min


def grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_idx, head_reduce="mean"):
    """
    attn: [Bgrp, heads, N, N]
    meta: (B, Hf, Wf, C, g)
    q_idx: índice del query dentro del grid token space (0..N-1), N=Hg*Wg
    gy,gx: cuál grupo (interleaving) visualizas
    Retorna MAD L1 en coords del featuremap completo (Hf,Wf).
    """
    B, Hf, Wf, C, g_meta = meta
    assert g_meta == g
    N = Hg * Wg

    grp = b * (g*g) + gy * g + gx
    A = attn[grp]

    if head_reduce == "mean":
        A = A.mean(0)
    elif head_reduce == "max":
        A = A.max(0).values
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'")

    w = A[q_idx]
    w = w / (w.sum() + 1e-12)

    # coords del query y keys en grid-local coords
    qy = q_idx // Wg
    qx = q_idx %  Wg

    ky = torch.arange(Hg, device=w.device).repeat_interleave(Wg)
    kx = torch.arange(Wg, device=w.device).repeat(Hg)

    # mapeo a featuremap completo por interleaving
    yq_full = qy * g + gy
    xq_full = qx * g + gx
    yk_full = ky * g + gy
    xk_full = kx * g + gx

    dist_l1 = (yk_full - yq_full).abs() + (xk_full - xq_full).abs()   # [N]
    mad = (w * dist_l1).sum().item()
    return mad


def grid_attn_mad_summary(attn, meta, Hg, Wg, g, b=0, gy=0, gx=0, q_idxs=(None,)):
    """
    Promedia MAD sobre múltiples queries (por ejemplo center/maxE/minE).
    Si q_idxs incluye None, usa el center por default.
    """
    N = Hg * Wg
    out = []
    for q in q_idxs:
        if q is None:
            q = (Hg//2)*Wg + (Wg//2)
        out.append(grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q))
    return float(sum(out) / len(out))


def kernel3x3_mad_l1(K):
    """
    K: [3,3] (puede tener negativos si es 'kernel' tipo conv).
    Si K es mezcla/atención, debería ser >=0 y sumar 1.
    Si puede tener negativos, usa softmax(|K|) o softmax(K) según tu definición.
    """
    K = torch.tensor(K)
    w = K.reshape(-1)
    w = torch.softmax(w, dim=0)
    coords = torch.tensor([(-1,-1),(-1,0),(-1,1),
                           ( 0,-1),( 0,0),( 0,1),
                           ( 1,-1),( 1,0),( 1,1)], dtype=torch.float32)
    dist = coords.abs().sum(dim=1)  # L1
    return float((w * dist).sum().item())



def pick_q_idxs_for_image(attn, Hg, Wg, g, b, gy=0, gx=0):
    """
    attn: [Bgrp, heads, N, N]
    Retorna (q_center, q_max, q_min) para la imagen b y grupo (gy,gx).
    """
    grp = b * (g*g) + gy * g + gx
    A = attn[grp].mean(0)  # [N,N] mean over heads
    q_center, q_max, q_min = _pick_q_indices_from_attn(A, Hg, Wg)
    return q_center, q_max, q_min


def outlooker_kernel_mad_norm(k3x3: torch.Tensor, eps=1e-12):
    """
    k3x3: [3,3] weights NO-NEG (idealmente ya suman 1).
    Retorna MAD normalizado a [0,1] dividiendo por 2.
    """
    k = torch.clamp(k3x3, min=0.0)
    k = k / (k.sum() + eps)

    dist = torch.tensor([[2,1,2],
                         [1,0,1],
                         [2,1,2]], device=k.device, dtype=k.dtype)
    mad = (k * dist).sum()         # en [0,2]
    return (mad / 2.0).item()


def _kernel_at(weights_5d, y, x):
    # weights_5d: (B, heads, 9, H, W)
    kern = weights_5d[:, :, :, y, x].mean(dim=1)
    return kern.view(-1, 3, 3)

def _softmax_local(attn_logits, k2=9):
    B, C, H, W = attn_logits.shape
    assert C % k2 == 0, f"Esperaba C múltiplo de {k2}. Got C={C}."
    heads = C // k2
    w = attn_logits.view(B, heads, k2, H, W)
    w = torch.softmax(w, dim=2)
    return w  # (B, heads, 9, H, W)

def _center_and_spread(weights_5d):
    # center index para 3x3 = 4
    center = weights_5d[:, :, 4, :, :].mean(dim=1)
    maxw   = weights_5d.max(dim=2).values.mean(dim=1)
    spread = 1.0 - maxw
    return center, spread


def _pick_positions_from_map(m, border=1):
    H, W = m.shape
    yc, xc = H // 2, W // 2

    m2 = m.clone()
    if border > 0:
        m2[:border, :] = -1e9
        m2[-border:, :] = -1e9
        m2[:, :border] = -1e9
        m2[:, -border:] = -1e9

    idx_max = int(torch.argmax(m2).item())
    y_max, x_max = divmod(idx_max, W)

    m3 = m.clone()
    if border > 0:
        m3[:border, :] = +1e9
        m3[-border:, :] = +1e9
        m3[:, :border] = +1e9
        m3[:, -border:] = +1e9

    idx_min = int(torch.argmin(m3).item())
    y_min, x_min = divmod(idx_min, W)

    return (yc, xc), (y_max, x_max), (y_min, x_min)

def outlooker_mad_for_image(attn_logits_b: torch.Tensor):
    """
    attn_logits_b: logits del Outlooker para UNA imagen b
                  (lo sacamos de attn_logits[b:b+1] para reutilizar helpers).
    Retorna MAD promedio (center, maxCW, minCW) normalizado en [0,1].
    """
    w = _softmax_local(attn_logits_b, k2=9)

    center_map, _ = _center_and_spread(w)
    cm = center_map[0]

    (yc, xc), (yM, xM), (ym, xm) = _pick_positions_from_map(cm)

    k_center = _kernel_at(w, yc, xc)[0]   # [3,3]
    k_max    = _kernel_at(w, yM, xM)[0]
    k_min    = _kernel_at(w, ym, xm)[0]

    mad_c = outlooker_kernel_mad_norm(k_center)
    mad_M = outlooker_kernel_mad_norm(k_max)
    mad_m = outlooker_kernel_mad_norm(k_min)

    return (mad_c + mad_M + mad_m) / 3.0, mad_c, mad_M, mad_m




def _get_random_batch(loader, device, seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if hasattr(loader, "__len__"):
        n_batches = len(loader)
        j = random.randrange(n_batches)
        it = iter(loader)
        for _ in range(j):
            next(it)
        batch = next(it)
    else:
        batch = next(iter(loader))

    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(device)

def _choose_random_indices(B, n_images, seed=None):
    if seed is not None:
        random.seed(seed + 12345)
    n_images = min(n_images, B)
    return random.sample(range(B), k=n_images)


@torch.no_grad()
def compute_grid_and_outlooker_mad_by_stage(
    model,
    loader,
    block_idx=0,
    stages=(0,1,2,3),
    n_images=16,
    seed=10,
    device="cuda",
    gy=0, gx=0,
    normalize_grid=True):

    model = model.to(device).eval()

    x_all = _get_random_batch(loader, device=device, seed=seed)
    B = x_all.shape[0]
    idxs = _choose_random_indices(B, n_images=n_images, seed=seed)
    x = x_all[idxs]
    n = x.shape[0]

    enable_mhsa_capture(model, True)
    cap_grid = GridAttnCapturer(model)
    cap_out  = OutlookAttnCapturer(model)

    _ = model(x)

    results = []

    for s in stages:
        # defaults
        Hf = Wf = None
        grid_denom = None
        grid_abs_mean = grid_abs_center = grid_abs_max = grid_abs_min = None

        out_abs_mean = out_abs_center = out_abs_max = out_abs_min = None

        # ------------------
        # GRID
        # ------------------
        pack_g = cap_grid.get(stage=s, block=block_idx)
        grid_ok = (pack_g is not None and pack_g.get("attn", None) is not None and pack_g.get("meta", None) is not None)

        grid_mean = grid_std = grid_c = grid_M = grid_m = None
        if grid_ok:
            attn = pack_g["attn"]
            meta = pack_g["meta"]
            Hg, Wg = pack_g["grid_hw"]
            g = pack_g["g"]

            Bm, Hf, Wf, C, _ = meta
            assert Bm == n

            grid_denom = float((Hf - 1) + (Wf - 1)) if normalize_grid else 1.0
            denom = grid_denom

            mad_list, mad_c_list, mad_M_list, mad_m_list = [], [], [], []
            for b in range(n):
                q_center, q_max, q_min = pick_q_idxs_for_image(attn, Hg, Wg, g, b, gy=gy, gx=gx)

                mad_avg = grid_attn_mad_summary(attn, meta, Hg, Wg, g, b=b, gy=gy, gx=gx,
                                               q_idxs=(q_center, q_max, q_min)) / denom
                mad_c   = grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_center) / denom
                mad_M   = grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_max)    / denom
                mad_m   = grid_attn_mad_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_min)    / denom

                mad_list.append(mad_avg)
                mad_c_list.append(mad_c); mad_M_list.append(mad_M); mad_m_list.append(mad_m)

            grid_mean = float(np.mean(mad_list))
            grid_std  = float(np.std(mad_list))
            grid_c    = float(np.mean(mad_c_list))
            grid_M    = float(np.mean(mad_M_list))
            grid_m    = float(np.mean(mad_m_list))

            # ABS (featuremap pixels)
            if normalize_grid:
                grid_abs_mean   = grid_mean * grid_denom
                grid_abs_center = grid_c    * grid_denom
                grid_abs_max    = grid_M    * grid_denom
                grid_abs_min    = grid_m    * grid_denom
            else:
                grid_abs_mean, grid_abs_center, grid_abs_max, grid_abs_min = grid_mean, grid_c, grid_M, grid_m

        # ------------------
        # OUTLOOKER
        # ------------------
        attn_logits = cap_out.get(stage=s, block=block_idx)
        out_ok = (attn_logits is not None)

        out_mean = out_std = out_c = out_M = out_m = None
        if out_ok:
            out_list, oc_list, oM_list, om_list = [], [], [], []
            for b in range(n):
                mad_avg, mad_c, mad_M, mad_m = outlooker_mad_for_image(attn_logits[b:b+1])
                out_list.append(mad_avg)
                oc_list.append(mad_c); oM_list.append(mad_M); om_list.append(mad_m)

            out_mean = float(np.mean(out_list))
            out_std  = float(np.std(out_list))
            out_c    = float(np.mean(oc_list))
            out_M    = float(np.mean(oM_list))
            out_m    = float(np.mean(om_list))

            # ABS in 3x3 neighborhood (max=2)
            out_abs_mean   = out_mean * 2.0
            out_abs_center = out_c    * 2.0
            out_abs_max    = out_M    * 2.0
            out_abs_min    = out_m    * 2.0

        if (not grid_ok) and (not out_ok):
            print(f"[WARN] No capturas (grid/outlooker) en stage={s}, block={block_idx}")
            continue

        results.append({
            "stage": s,
            "block": block_idx,
            "seed": seed,
            "n_images": n,
            "gy": gy, "gx": gx,

            # GRID norm
            "MAD_grid_mean": grid_mean,
            "MAD_grid_std":  grid_std,
            "MAD_grid_center_mean": grid_c,
            "MAD_grid_max_mean":    grid_M,
            "MAD_grid_min_mean":    grid_m,

            # GRID meta + abs
            "grid_Hf": Hf if grid_ok else None,
            "grid_Wf": Wf if grid_ok else None,
            "grid_denom": grid_denom,
            "MAD_grid_abs_mean": grid_abs_mean,
            "MAD_grid_abs_center_mean": grid_abs_center,
            "MAD_grid_abs_max_mean":    grid_abs_max,
            "MAD_grid_abs_min_mean":    grid_abs_min,

            # OUT norm
            "MAD_outlook_mean": out_mean,
            "MAD_outlook_std":  out_std,
            "MAD_outlook_center_mean": out_c,
            "MAD_outlook_max_mean":    out_M,
            "MAD_outlook_min_mean":    out_m,

            # OUT abs
            "MAD_outlook_abs_mean": out_abs_mean,
            "MAD_outlook_abs_center_mean": out_abs_center,
            "MAD_outlook_abs_max_mean":    out_abs_max,
            "MAD_outlook_abs_min_mean":    out_abs_min,})

    cap_grid.close()
    cap_out.close()
    enable_mhsa_capture(model, False)
    return results
