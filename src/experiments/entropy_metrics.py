import os, re
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.experiments.heat_map_att_outlooker import *
from src.experiments.heat_map_att_grid import *

def _softmax_local(attn_logits, k2=9):
    B, C, H, W = attn_logits.shape
    assert C % k2 == 0, f"Esperaba C múltiplo de {k2}. Got C={C}."
    heads = C // k2
    w = attn_logits.view(B, heads, k2, H, W)
    w = torch.softmax(w, dim=2)
    return w  # (B, heads, 9, H, W)


def _kernel_at(weights_5d, y, x):
    # weights_5d: (B, heads, 9, H, W)
    kern = weights_5d[:, :, :, y, x].mean(dim=1)
    return kern.view(-1, 3, 3)


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


def entropy_from_probs(p: torch.Tensor, eps=1e-12):
    """
    p: probabilities summing to 1 (any shape).
    Returns Shannon entropy (nats).
    """
    p = torch.clamp(p, min=eps)
    return float(-(p * torch.log(p)).sum().item())


def entropy_normalized_nats(H, K):
    """
    Normalize entropy by log(K) so it lies in [0,1] (approximately).
    """
    return H / (np.log(K) + 1e-12)

def sample_q_indices(Hg, Wg, n_q=32, seed=0, exclude_border=1, device="cpu"):
    ys = torch.arange(Hg, device=device)
    xs = torch.arange(Wg, device=device)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    if exclude_border > 0:
        mask = (Y >= exclude_border) & (Y < Hg - exclude_border) & \
               (X >= exclude_border) & (X < Wg - exclude_border)
        valid = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    else:
        valid = torch.arange(Hg * Wg, device=device)

    rng = np.random.default_rng(seed)
    if len(valid) <= n_q:
        return valid.tolist()
    idx = rng.choice(len(valid), size=n_q, replace=False)
    return valid[idx].tolist()

def sample_xy(H, W, n_xy=64, seed=0, exclude_border=1):
    rng = np.random.default_rng(seed)
    ys = np.arange(exclude_border, H - exclude_border)
    xs = np.arange(exclude_border, W - exclude_border)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    coords = np.stack([Y.reshape(-1), X.reshape(-1)], axis=1)
    if len(coords) == 0:
        return []
    if len(coords) <= n_xy:
        return coords.tolist()
    idx = rng.choice(len(coords), size=n_xy, replace=False)
    return coords[idx].tolist()

def grid_attn_mad_entropy_for_query(attn, meta, Hg, Wg, g, b, gy, gx, q_idx, head_reduce="mean", eps=1e-12):
    """
    Returns:
      mad_abs (featuremap L1),
      H (entropy in nats),
      Hn (entropy normalized by log(N))
    """
    B, Hf, Wf, C, g_meta = meta
    assert g_meta == g
    N = Hg * Wg

    grp = b * (g*g) + gy * g + gx
    A = attn[grp]  # [heads, N, N]

    if head_reduce == "mean":
        A = A.mean(0)
    elif head_reduce == "max":
        A = A.max(0).values
    else:
        raise ValueError("head_reduce must be 'mean' or 'max'")

    w = A[q_idx]
    w = w / (w.sum() + eps)

    # --- MAD ---
    qy = q_idx // Wg
    qx = q_idx %  Wg

    ky = torch.arange(Hg, device=w.device).repeat_interleave(Wg)
    kx = torch.arange(Wg, device=w.device).repeat(Hg)

    yq_full = qy * g + gy
    xq_full = qx * g + gx
    yk_full = ky * g + gy
    xk_full = kx * g + gx

    dist_l1 = (yk_full - yq_full).abs() + (xk_full - xq_full).abs()
    mad_abs = float((w * dist_l1).sum().item())

    # --- Entropy ---
    H = entropy_from_probs(w, eps=eps)             # nats
    Hn = entropy_normalized_nats(H, K=N)           # [0,1] approx
    return mad_abs, H, Hn

def grid_attn_mad_entropy_summary(attn, meta, Hg, Wg, g, b, gy, gx, q_idxs, head_reduce="mean"):
    """
    Average over many queries.
    Returns mean MAD_abs, mean H, mean Hn.
    """
    mads, Hs, Hns = [], [], []
    for q in q_idxs:
        mad_abs, H, Hn = grid_attn_mad_entropy_for_query(
            attn, meta, Hg, Wg, g, b, gy, gx, q, head_reduce=head_reduce
        )
        mads.append(mad_abs); Hs.append(H); Hns.append(Hn)
    return float(np.mean(mads)), float(np.mean(Hs)), float(np.mean(Hns))

def outlooker_kernel_mad_norm(k3x3: torch.Tensor, eps=1e-12):
    k = torch.clamp(k3x3, min=0.0)
    k = k / (k.sum() + eps)
    dist = torch.tensor([[2,1,2],
                         [1,0,1],
                         [2,1,2]], device=k.device, dtype=k.dtype)
    mad = (k * dist).sum()   # [0,2]
    return (mad / 2.0).item()

def outlooker_kernel_entropy(k3x3: torch.Tensor, eps=1e-12):
    """
    Entropy of a 3x3 kernel distribution.
    Returns H (nats) and Hn = H/log(9).
    """
    k = torch.clamp(k3x3, min=0.0)
    k = k / (k.sum() + eps)
    p = k.reshape(-1)
    H = entropy_from_probs(p, eps=eps)
    Hn = entropy_normalized_nats(H, K=9)
    return H, Hn

def outlooker_mad_entropy_for_image_sampled(attn_logits_b: torch.Tensor, n_xy=64, seed=0, exclude_border=1):
    """
    Returns per-image:
      MAD_norm_mean, MAD_norm_std,
      H_mean (nats), H_std,
      Hn_mean, Hn_std
    """
    w = _softmax_local(attn_logits_b, k2=9)
    Hmap, Wmap = int(w.shape[1]), int(w.shape[2])

    # adaptive border (avoid empty coords)
    eb = int(exclude_border)
    if Hmap - 2*eb <= 0 or Wmap - 2*eb <= 0:
        eb = 0

    coords = sample_xy(Hmap, Wmap, n_xy=n_xy, seed=seed, exclude_border=eb)
    if len(coords) == 0:
        coords = sample_xy(Hmap, Wmap, n_xy=n_xy, seed=seed, exclude_border=0)
        if len(coords) == 0:
            return None

    mads, Hs, Hns = [], [], []
    for (y, x) in coords:
        k = _kernel_at(w, y, x)[0]   # [3,3]
        mads.append(outlooker_kernel_mad_norm(k))
        H, Hn = outlooker_kernel_entropy(k)
        Hs.append(H); Hns.append(Hn)

    return {
        "MAD_norm_mean": float(np.mean(mads)),
        "MAD_norm_std":  float(np.std(mads)),
        "H_mean":        float(np.mean(Hs)),
        "H_std":         float(np.std(Hs)),
        "Hn_mean":       float(np.mean(Hns)),
        "Hn_std":        float(np.std(Hns)),
    }


@torch.no_grad()
def compute_grid_and_outlooker_mad_entropy_by_stage(
    model,
    loader,
    block_idx=0,
    stages=(0,1,2,3),
    n_images=64,
    seed=0,
    device="cuda",
    normalize_grid=True,
    # sampling controls
    grid_n_q=32,
    grid_exclude_border=1,
    grid_avg_over_groups=True,
    out_n_xy=64,
    out_exclude_border=1,
):
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
        # ------------------
        # GRID
        # ------------------
        pack_g = cap_grid.get(stage=s, block=block_idx)
        grid_ok = (pack_g is not None and pack_g.get("attn", None) is not None and pack_g.get("meta", None) is not None)

        grid_mean = grid_std = None
        grid_abs_mean = None
        grid_H_mean = grid_H_std = None
        grid_Hn_mean = grid_Hn_std = None

        Hf = Wf = None
        grid_denom = None

        if grid_ok:
            attn = pack_g["attn"]
            meta = pack_g["meta"]
            Hg, Wg = pack_g["grid_hw"]
            g = pack_g["g"]

            Bm, Hf, Wf, C, _ = meta
            assert Bm == n

            grid_denom = float((Hf - 1) + (Wf - 1)) if normalize_grid else 1.0
            Nkeys = Hg * Wg

            per_image_mad = []
            per_image_H   = []
            per_image_Hn  = []

            for b in range(n):
                # average across all interleaving groups if requested
                group_mad = []
                group_H   = []
                group_Hn  = []

                gy_range = range(g) if grid_avg_over_groups else [0]
                gx_range = range(g) if grid_avg_over_groups else [0]

                for gy in gy_range:
                    for gx in gx_range:
                        q_seed = seed + 100000*s + 1000*b + 97*gy + 131*gx + 17*block_idx
                        q_idxs = sample_q_indices(
                            Hg, Wg,
                            n_q=grid_n_q,
                            seed=q_seed,
                            exclude_border=grid_exclude_border,
                            device=attn.device
                        )
                        if len(q_idxs) == 0:
                            continue

                        mad_abs, H, Hn = grid_attn_mad_entropy_summary(
                            attn, meta, Hg, Wg, g, b=b, gy=gy, gx=gx, q_idxs=q_idxs, head_reduce="mean"
                        )

                        mad = mad_abs / grid_denom if normalize_grid else mad_abs
                        group_mad.append(mad)
                        group_H.append(H)
                        group_Hn.append(Hn)

                if len(group_mad):
                    per_image_mad.append(float(np.mean(group_mad)))
                    per_image_H.append(float(np.mean(group_H)))
                    per_image_Hn.append(float(np.mean(group_Hn)))

            if len(per_image_mad):
                grid_mean = float(np.mean(per_image_mad))
                grid_std  = float(np.std(per_image_mad))
                grid_abs_mean = grid_mean * grid_denom if normalize_grid else grid_mean

                grid_H_mean  = float(np.mean(per_image_H))
                grid_H_std   = float(np.std(per_image_H))
                grid_Hn_mean = float(np.mean(per_image_Hn))
                grid_Hn_std  = float(np.std(per_image_Hn))

        # ------------------
        # OUTLOOKER
        # ------------------
        attn_logits = cap_out.get(stage=s, block=block_idx)
        out_ok = (attn_logits is not None)

        out_mean = out_std = None
        out_abs_mean = None
        out_H_mean = out_H_std = None
        out_Hn_mean = out_Hn_std = None

        if out_ok:
            per_image_mad = []
            per_image_H   = []
            per_image_Hn  = []

            for b in range(n):
                o_seed = seed + 200000*s + 1000*b + 19*block_idx
                stats = outlooker_mad_entropy_for_image_sampled(
                    attn_logits[b:b+1],
                    n_xy=out_n_xy,
                    seed=o_seed,
                    exclude_border=out_exclude_border
                )
                if stats is None:
                    continue

                per_image_mad.append(stats["MAD_norm_mean"])
                per_image_H.append(stats["H_mean"])
                per_image_Hn.append(stats["Hn_mean"])

            if len(per_image_mad):
                out_mean = float(np.mean(per_image_mad))    # norm in [0,1]
                out_std  = float(np.std(per_image_mad))
                out_abs_mean = out_mean * 2.0               # abs in [0,2]

                out_H_mean  = float(np.mean(per_image_H))
                out_H_std   = float(np.std(per_image_H))
                out_Hn_mean = float(np.mean(per_image_Hn))
                out_Hn_std  = float(np.std(per_image_Hn))

        if (not grid_ok) and (not out_ok):
            print(f"[WARN] No captures (grid/outlooker) in stage={s}, block={block_idx}")
            continue

        results.append({
            "stage": s,
            "block": block_idx,
            "seed": seed,
            "n_images": int(n),

            # sampling config
            "grid_n_q": grid_n_q,
            "grid_exclude_border": grid_exclude_border,
            "grid_avg_over_groups": bool(grid_avg_over_groups),
            "out_n_xy": out_n_xy,
            "out_exclude_border": out_exclude_border,

            # GRID
            "MAD_grid_mean": grid_mean,
            "MAD_grid_std":  grid_std,
            "MAD_grid_abs_mean": grid_abs_mean,
            "H_grid_mean": grid_H_mean,
            "H_grid_std":  grid_H_std,
            "Hn_grid_mean": grid_Hn_mean,
            "Hn_grid_std":  grid_Hn_std,

            "grid_Hf": Hf if grid_ok else None,
            "grid_Wf": Wf if grid_ok else None,
            "grid_denom": grid_denom,

            # OUTLOOKER
            "MAD_outlook_mean": out_mean,
            "MAD_outlook_std":  out_std,
            "MAD_outlook_abs_mean": out_abs_mean,
            "H_out_mean": out_H_mean,
            "H_out_std":  out_H_std,
            "Hn_out_mean": out_Hn_mean,
            "Hn_out_std":  out_Hn_std,
        })

    cap_grid.close()
    cap_out.close()
    enable_mhsa_capture(model, False)
    return results


def run_mad_entropy_pipeline(
    model,
    test_loader,
    stage_depths,
    seeds=(0,1,2),
    n_images=64,
    grid_n_q=32,
    out_n_xy=64,
    device="cuda"
):
    all_res = []
    for seed in seeds:
        for s in [0,1,2,3]:
            for b in range(stage_depths[s]):
                all_res += compute_grid_and_outlooker_mad_entropy_by_stage(
                    model=model,
                    loader=test_loader,
                    block_idx=b,
                    stages=(s,),
                    n_images=n_images,
                    seed=seed,
                    device=device,
                    normalize_grid=True,
                    grid_n_q=grid_n_q,
                    grid_exclude_border=1,
                    grid_avg_over_groups=True,
                    out_n_xy=out_n_xy,
                    out_exclude_border=1,
                )
    return all_res