import os
import re
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

# ---------------------------
# 1) Hook para capturar logits de OutlookAttention2d
# ---------------------------
class OutlookAttnCapturer:
    """
    Captura la salida de la conv 'attn' dentro de cada módulo OutlookAttention2d.
    Guarda tensores de forma: (B, heads*k*k, H, W) con k=3 => heads*9 canales.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles = []
        self.cache = {}  # key -> tensor
        self.key_by_stage_block = {}  # (stage, block) -> key

        pat = re.compile(r"stages\.(\d+)\.(\d+)\.")

        for name, m in model.named_modules():
            if m.__class__.__name__ == "OutlookAttention2d":
                conv = getattr(m, "attn", None)
                if isinstance(conv, torch.nn.Conv2d):
                    key = name
                    mm = pat.search(name + ".")
                    if mm:
                        stage = int(mm.group(1))
                        block = int(mm.group(2))
                        self.key_by_stage_block[(stage, block)] = key

                    h = conv.register_forward_hook(self._make_hook(key))
                    self.handles.append(h)

    def _make_hook(self, key):
        def hook(mod, inp, out):
            self.cache[key] = out.detach()
        return hook

    def get(self, stage: int, block: int):
        key = self.key_by_stage_block.get((stage, block), None)
        if key is None:
            return None
        return self.cache.get(key, None)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------
# 2) Utilidades de plotting
# ---------------------------
def _to_img_uint8(x, mean, std):
    x = x.detach().cpu()
    mean = torch.tensor(mean).view(3,1,1)
    std  = torch.tensor(std).view(3,1,1)
    x = x * std + mean
    x = x.clamp(0, 1)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1,2,0).numpy()

def _upsample_map(m, out_hw):
    t = m[None, None, ...]
    t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    return t[0,0]

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

def _kernel_at(weights_5d, y, x):
    # weights_5d: (B, heads, 9, H, W)
    kern = weights_5d[:, :, :, y, x].mean(dim=1)
    return kern.view(-1, 3, 3)

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


def _fm_to_img_xy(y_f, x_f, H_f, W_f, H_img, W_img):
    """
    Mapea coordenadas (y_f, x_f) en feature map (H_f, W_f)
    a coordenadas aproximadas en la imagen (H_img, W_img).

    Usamos el centro de la celda y escalamos.
    """
    yy = (y_f + 0.5) / H_f
    xx = (x_f + 0.5) / W_f
    y_img = yy * H_img
    x_img = xx * W_img
    return x_img, y_img


@torch.no_grad()
def plot_outlooker_locality_random(
    model: torch.nn.Module,
    loader,
    device="cuda",
    save_dir="./figures/outlooker_locality",
    n_images=4,
    stages=(0, 1, 2, 3),
    block_idx=0,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    dpi=220,
    seed=None,
    show=True,
    kernel_cap=0.15,
    kernel_cmap="RdBu_r",
    overlay_alpha=0.45):

    """
    Visualización cualitativa del Outlooker:
      - Input
      - center-weight overlay
      - spread overlay
      - kernels 3x3 (dev respecto a uniforme 1/9) en: center / maxCW / minCW

    Requiere que ya existan en tu entorno:
      - _get_random_batch, _choose_random_indices
      - OutlookAttnCapturer
      - _to_img_uint8, _upsample_map
      - _softmax_local, _center_and_spread
      - _pick_positions_from_map, _kernel_at
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device).eval()

    # ---- batch aleatorio + imágenes aleatorias ----
    x_all = _get_random_batch(loader, device=device, seed=seed)  # (B,3,H,W)
    B = x_all.shape[0]
    idxs = _choose_random_indices(B, n_images=n_images, seed=seed)
    x = x_all[idxs]  # (n,3,H,W)

    # ---- capturar logits ----
    cap = OutlookAttnCapturer(model)
    _ = model(x)

    H_img, W_img = x.shape[-2], x.shape[-1]
    imgs_np = [_to_img_uint8(x[i], mean=mean, std=std) for i in range(x.shape[0])]

    saved_paths = []

    for s in stages:
        attn_logits = cap.get(stage=s, block=block_idx)
        if attn_logits is None:
            print(f"[WARN] No se capturó Outlooker attn en stage={s}, block={block_idx}.")
            continue

        w = _softmax_local(attn_logits, k2=9)
        center_map, spread_map = _center_and_spread(w)

        center_up = torch.stack([_upsample_map(center_map[i], (H_img, W_img)) for i in range(x.shape[0])], dim=0)
        spread_up = torch.stack([_upsample_map(spread_map[i], (H_img, W_img)) for i in range(x.shape[0])], dim=0)

        cmin, cmax = float(center_up.min().item()), float(center_up.max().item())
        smin, smax = float(spread_up.min().item()), float(spread_up.max().item())

        u = 1.0 / 9.0
        # promediamos heads para obtener distribución 3x3 más estable
        w_avg = w.mean(dim=1)         # (n, 9, Hf, Wf)
        kdev_all = w_avg - u          # (n, 9, Hf, Wf)
        maxabs_stage = float(kdev_all.abs().max().item())
        v_stage = min(maxabs_stage, kernel_cap)
        if v_stage < 1e-6:
            v_stage = 1e-6

        # ---- figura ----
        # [Input | center-weight | spread | k@center | k@maxCW | k@minCW]
        fig, axes = plt.subplots(
            x.shape[0], 6,
            figsize=(6 * 3.9, x.shape[0] * 3.4),
            dpi=dpi)

        if x.shape[0] == 1:
            axes = axes[None, :]

        mappable_center = None
        mappable_spread = None
        mappable_k = None

        for i in range(x.shape[0]):
            # Input
            axes[i, 0].imshow(imgs_np[i])
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            # center-weight overlay
            axes[i, 1].imshow(imgs_np[i])
            m1 = axes[i, 1].imshow(
                center_up[i].detach().cpu().numpy(),
                alpha=overlay_alpha,
                vmin=cmin, vmax=cmax)

            axes[i, 1].set_title("Outlooker: center-weight")
            axes[i, 1].axis("off")
            if mappable_center is None:
                mappable_center = m1

            # spread overlay
            axes[i, 2].imshow(imgs_np[i])
            m2 = axes[i, 2].imshow(
                spread_up[i].detach().cpu().numpy(),
                alpha=overlay_alpha,
                vmin=smin, vmax=smax )

            axes[i, 2].set_title("Outlooker: spread (1-max)")
            axes[i, 2].axis("off")
            if mappable_spread is None:
                mappable_spread = m2

            # posiciones en el featuremap (antes del upsample)
            cm = center_map[i]  # (Hf, Wf)
            (yc, xc), (yM, xM), (ym, xm) = _pick_positions_from_map(cm)

            Hf, Wf = cm.shape
            # mapear a coords de imagen
            xC, yC = _fm_to_img_xy(yc, xc, Hf, Wf, H_img, W_img)
            xM_, yM_ = _fm_to_img_xy(yM, xM, Hf, Wf, H_img, W_img)
            xm_, ym_ = _fm_to_img_xy(ym, xm, Hf, Wf, H_img, W_img)

            # dibujar sobre center-weight overlay (col 1)
            # (si quieres también en spread overlay, repite con axes[i,2])
            axes[i, 1].scatter([xC], [yC], s=70, c="white", marker="o", linewidths=1.2, edgecolors="black")
            axes[i, 1].scatter([xM_], [yM_], s=80, c="red",   marker="x", linewidths=2.0)
            axes[i, 1].scatter([xm_], [ym_], s=90, c="blue",  marker="+", linewidths=2.0)

            k_center = _kernel_at(w[i:i+1], yc, xc)[0].detach().cpu().numpy()
            k_max    = _kernel_at(w[i:i+1], yM, xM)[0].detach().cpu().numpy()
            k_min    = _kernel_at(w[i:i+1], ym, xm)[0].detach().cpu().numpy()

            # kernel dev respecto a uniforme
            k_center_dev = k_center - u
            k_max_dev    = k_max    - u
            k_min_dev    = k_min    - u

            axes[i, 3].imshow(k_center_dev, vmin=-v_stage, vmax=v_stage, cmap=kernel_cmap, interpolation="nearest")
            axes[i, 3].set_title("Kernel 3×3 @ center")
            axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])

            axes[i, 4].imshow(k_max_dev, vmin=-v_stage, vmax=v_stage, cmap=kernel_cmap, interpolation="nearest")
            axes[i, 4].set_title("Kernel 3×3 @ max CW")
            axes[i, 4].set_xticks([]); axes[i, 4].set_yticks([])

            imk = axes[i, 5].imshow(k_min_dev, vmin=-v_stage, vmax=v_stage, cmap=kernel_cmap, interpolation="nearest")
            axes[i, 5].set_title("Kernel 3×3 @ min CW")

            axes[i, 5].set_xticks([]); axes[i, 5].set_yticks([])
            axes[i, 2].scatter([xC], [yC], s=70, c="white", marker="o", linewidths=1.2, edgecolors="black")
            axes[i, 2].scatter([xM_], [yM_], s=80, c="red",   marker="x", linewidths=2.0)
            axes[i, 2].scatter([xm_], [ym_], s=90, c="blue",  marker="+", linewidths=2.0)

            if mappable_k is None:
                mappable_k = imk

        fig.suptitle(f"Outlooker Locality (random batch) — stage={s} block={block_idx}", y=0.995, fontsize=14)
        fig.tight_layout()

        fig.colorbar(mappable_center, ax=axes[:, 1].ravel().tolist(), fraction=0.015, pad=0.01)
        fig.colorbar(mappable_spread, ax=axes[:, 2].ravel().tolist(), fraction=0.015, pad=0.01)
        fig.colorbar(mappable_k,      ax=axes[:, 3:].ravel().tolist(), fraction=0.015, pad=0.01)

        out_path = os.path.join(save_dir, f"outlooker_locality_stage{s}_block{block_idx}_random.png")
        fig.savefig(out_path, bbox_inches="tight")
        saved_paths.append(out_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

    cap.close()

    print("Saved:")
    for p in saved_paths:
        print(" -", p)
    return saved_paths