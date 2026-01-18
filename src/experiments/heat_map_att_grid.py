import os, re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def enable_mhsa_capture(model: torch.nn.Module, enabled: bool = True):
    for m in model.modules():
        if m.__class__.__name__ == "MultiHeadSelfAttention":
            setattr(m, "capture_attn", enabled)
    return model

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


class GridAttnCapturer:
    """
    Captura input BHWC y attn [Bgrp, heads, N, N] para GridAttention2D.
    Compatible con nombres: stages.<stage>.<block>.grid_attn
    """
    def __init__(self, model: torch.nn.Module):
        self.handles = []
        self.cache = {}   # (stage, block) -> dict
        self.found = {}   # (stage, block) -> module_name

        for name, m in model.named_modules():
            if m.__class__.__name__ not in ("GridAttention2D", "LocalAttention2D"):
                continue

            # parse: stages.<s>.<b>.grid_attn
            mm = re.search(r"stages\.(\d+)\.(\d+)\.grid_attn$", name)
            if mm is None:
                continue

            s = int(mm.group(1))
            b = int(mm.group(2))
            key = (s, b)
            self.found[key] = name

            def _hook(mod, inp, out, key=key):
                x_in = inp[0]  # BHWC
                attn = getattr(mod.mhsa, "last_attn", None)
                if attn is None:
                    attn = getattr(mod.mhsa, "last_attn_postdrop", None)

                meta = getattr(mod, "_last_meta", None)
                grid_hw = getattr(mod, "_last_grid_hw", None)
                g = getattr(mod, "_last_g", None)

                self.cache[key] = {
                    "x_in": x_in.detach(),
                    "attn": None if attn is None else attn.detach(),
                    "meta": meta,
                    "grid_hw": grid_hw,
                    "g": g}

            self.handles.append(m.register_forward_hook(_hook))

    def get(self, stage: int, block: int):
        return self.cache.get((stage, block), None)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _to_img_uint8(x_chw, mean, std):
    # x_chw: [3,H,W] torch
    x = x_chw.detach().cpu()
    mean_t = torch.tensor(mean)[:, None, None]
    std_t  = torch.tensor(std)[:, None, None]
    x = x * std_t + mean_t
    x = x.clamp(0, 1)
    x = (x * 255).byte().permute(1, 2, 0).numpy()
    return x

def _pos_center(H, W):
    return (H // 2, W // 2)

def _pos_max_energy(x_bhwc):
    # x_bhwc: [H,W,C]
    e = (x_bhwc ** 2).sum(dim=-1)  # [H,W]
    idx = torch.argmax(e).item()
    H, W = e.shape
    return (idx // W, idx % W)

def _pos_min_energy(x_bhwc):
    e = (x_bhwc ** 2).sum(dim=-1)  # [H,W]
    idx = torch.argmin(e).item()
    H, W = e.shape
    return (idx // W, idx % W)


def _gridattn_query_heatmap(attn_bgrp_h_n_n, meta, grid_hw, q_pos, head_agg="mean"):
    """
    attn: [Bgrp, heads, N, N]
    meta: (B, H, W, C, g)
    grid_hw: (Hg, Wg)
    q_pos: (hq, wq) en el featuremap (H,W) del grid-attn (NO la imagen original)
    retorna heatmap [B, H, W] en coords del featuremap
    """
    B, H, W, C, g = meta
    Hg, Wg = grid_hw
    N = Hg * Wg

    hq, wq = q_pos
    h_mod, w_mod = hq % g, wq % g
    h_loc, w_loc = hq // g, wq // g
    q_idx = h_loc * Wg + w_loc

    # group offset dentro del batch: offset = h_mod*g + w_mod  (coincide con permute en grid_partition)
    offset = h_mod * g + w_mod

    # heat: [B,H,W] (sparse)
    heat = torch.zeros((B, H, W), device=attn_bgrp_h_n_n.device, dtype=attn_bgrp_h_n_n.dtype)

    for b in range(B):
        grp = b * (g * g) + offset
        a = attn_bgrp_h_n_n[grp]  # [heads, N, N]
        a_q = a[:, q_idx, :]      # [heads, N]

        if head_agg == "mean":
            w = a_q.mean(dim=0)   # [N]
        elif head_agg == "max":
            w = a_q.max(dim=0).values
        else:
            h = int(head_agg)
            w = a_q[h]

        w2 = w.view(Hg, Wg)       # [Hg,Wg]

        # scatter a coords originales (solo posiciones con mismo mod)
        # (i,j) -> (i*g + h_mod, j*g + w_mod)
        for i in range(Hg):
            hi = i * g + h_mod
            for j in range(Wg):
                wj = j * g + w_mod
                heat[b, hi, wj] = w2[i, j]

    return heat

def _smooth_heatmap(heat_bhw, g, mode="box"):
    """
    Suaviza heatmap sparse para overlay.
    - box: conv con kernel g×g de unos
    """
    x = heat_bhw[:, None, :, :]  # [B,1,H,W]
    if mode == "box":
        k = g
        ker = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
        pad = k // 2
        y = F.conv2d(x, ker, padding=pad)
        return y[:, 0]
    return heat_bhw

@torch.no_grad()
def plot_grid_attention_random(
    model: torch.nn.Module,
    loader,
    device="cuda",
    save_dir="./figures/grid_attention",
    n_images=4,
    stages=(0, 1, 2, 3),
    block_idx=0,
    stage_depths=None,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    dpi=220,
    seed=None,
    head_agg="mean",        # "mean" o "max" o int
    smooth=True,
    show=True,
    overlay_alpha=0.45,     # como en Outlooker
):
    """
    Visualización cualitativa de Grid Attention:
      [Input | query center | query max-energy | query min-energy]

    Requiere que existan:
      - _get_random_batch, _choose_random_indices
      - enable_mhsa_capture, GridAttnCapturer
      - _to_img_uint8
      - _pos_center, _pos_max_energy, _pos_min_energy
      - _gridattn_query_heatmap
      - _smooth_heatmap
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device).eval()

    # filtra stages válidos para este block_idx
    if stage_depths is not None:
        valid_stages = tuple([s for s in stages if 0 <= s < len(stage_depths) and block_idx < stage_depths[s]])
    else:
        valid_stages = tuple(stages)

    # ---- batch aleatorio + imágenes aleatorias ----
    x_all = _get_random_batch(loader, device=device, seed=seed)  # [B,3,H,W]
    B = x_all.shape[0]
    idxs = _choose_random_indices(B, n_images=n_images, seed=seed)
    x = x_all[idxs]  # [n,3,H,W]

    # ---- captura ----
    enable_mhsa_capture(model, True)
    cap = GridAttnCapturer(model)

    _ = model(x)

    imgs_np = [_to_img_uint8(x[i], mean, std) for i in range(x.shape[0])]
    saved = []

    for s in valid_stages:
        pack = cap.get(stage=s, block=block_idx)
        if pack is None or pack.get("attn", None) is None or pack.get("meta", None) is None:
            print(f"[WARN] No grid-attn capturada en stage={s}, block={block_idx}.")
            continue

        x_in = pack["x_in"]        # [n, Hf, Wf, C] (BHWC)
        attn = pack["attn"]        # [...]
        meta = pack["meta"]        # (...)
        grid_hw = pack["grid_hw"]  # (Hg,Wg)
        g = pack["g"]

        n, Hf, Wf, Cf = x_in.shape

        q_pos = []
        for i in range(n):
            xi = x_in[i]
            q_pos.append({
                "center": _pos_center(Hf, Wf),
                "maxE":   _pos_max_energy(xi),
                "minE":   _pos_min_energy(xi),})

        heats_center_list, heats_maxE_list, heats_minE_list = [], [], []
        for i in range(n):
            hc = _gridattn_query_heatmap(attn, meta, grid_hw, q_pos[i]["center"], head_agg=head_agg)
            hM = _gridattn_query_heatmap(attn, meta, grid_hw, q_pos[i]["maxE"],   head_agg=head_agg)
            hm = _gridattn_query_heatmap(attn, meta, grid_hw, q_pos[i]["minE"],   head_agg=head_agg)

            # hc/hM/hm suelen venir como [n,Hf,Wf]; nos quedamos con la i
            heats_center_list.append(hc[i])
            heats_maxE_list.append(hM[i])
            heats_minE_list.append(hm[i])

        heats_center = torch.stack(heats_center_list, dim=0)  # [n,Hf,Wf]
        heats_maxE   = torch.stack(heats_maxE_list,   dim=0)
        heats_minE   = torch.stack(heats_minE_list,   dim=0)

        if smooth:
            heats_center = _smooth_heatmap(heats_center, g=g, mode="box")
            heats_maxE   = _smooth_heatmap(heats_maxE,   g=g, mode="box")
            heats_minE   = _smooth_heatmap(heats_minE,   g=g, mode="box")

        # rangos consistentes por figura
        vmin = float(torch.min(torch.stack([heats_center, heats_maxE, heats_minE])).item())
        vmax = float(torch.max(torch.stack([heats_center, heats_maxE, heats_minE])).item())

        # 4 columnas, n filas
        fig, axes = plt.subplots(
            n, 4,
            figsize=(4 * 3.9, n * 3.4),
            dpi=dpi)

        if n == 1:
            axes = axes[None, :]

        mappable = None

        def _overlay(ax, img_np, heat_2d, title, q, Hf, Wf):
          Himg, Wimg = img_np.shape[0], img_np.shape[1]

          up = F.interpolate(
              heat_2d[None, None], size=(Himg, Wimg),
              mode="nearest" )[0, 0]

          ax.imshow(img_np)
          mm = ax.imshow(up.detach().cpu().numpy(), alpha=overlay_alpha, vmin=vmin, vmax=vmax)

          yy = (q[0] + 0.5) / Hf * Himg
          xx = (q[1] + 0.5) / Wf * Wimg
          ax.scatter([xx], [yy], s=70, c="white", marker="o",
                    linewidths=1.2, edgecolors="black")

          ax.set_title(title)
          ax.axis("off")
          return mm

        for i in range(n):
            # Input
            axes[i, 0].imshow(imgs_np[i])
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            q_c = q_pos[i]["center"]
            q_M = q_pos[i]["maxE"]
            q_m = q_pos[i]["minE"]

            m1 = _overlay(axes[i, 1], imgs_np[i], heats_center[i], "GridAttn query: center",     q_c, Hf, Wf)
            m2 = _overlay(axes[i, 2], imgs_np[i], heats_maxE[i],   "GridAttn query: max-energy", q_M, Hf, Wf)
            m3 = _overlay(axes[i, 3], imgs_np[i], heats_minE[i],   "GridAttn query: min-energy", q_m, Hf, Wf)

            if mappable is None:
                mappable = m1

        fig.suptitle(f"Grid Attention (random batch) — stage={s} block={block_idx}", y=0.995, fontsize=14)
        fig.tight_layout()

        if mappable is not None:
            fig.colorbar(mappable, ax=axes[:, 1:].ravel().tolist(), fraction=0.015, pad=0.01)

        out_path = os.path.join(save_dir, f"gridattn_stage{s}_block{block_idx}_seed{seed}.png")
        fig.savefig(out_path, bbox_inches="tight")
        saved.append(out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    cap.close()
    enable_mhsa_capture(model, False)

    print("Saved:")
    for p in saved:
        print(" -", p)
    return saved
