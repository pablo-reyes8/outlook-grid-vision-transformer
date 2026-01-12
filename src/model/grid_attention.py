from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.grid_partition import *

AttnMode = Literal["grid"]

@dataclass(frozen=True)
class AttentionConfig:
    dim: int
    num_heads: int
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0


@dataclass(frozen=True)
class GridAttention2DConfig:
    mode: AttnMode
    dim: int
    num_heads: int
    grid_size: int
    window_size: int = 1
    qkv_bias: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    """
    Standard MHSA for token sequences.

    Input:  x [B, N, C]
    Output: y [B, N, C]

    Works for both window and grid partitions because both can be flattened to [Bgrp, N, C].
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        if cfg.dim <= 0:
            raise ValueError("cfg.dim must be > 0")
        if cfg.num_heads <= 0:
            raise ValueError("cfg.num_heads must be > 0")
        if cfg.dim % cfg.num_heads != 0:
            raise ValueError(f"dim ({cfg.dim}) must be divisible by num_heads ({cfg.num_heads})")

        self.dim = cfg.dim
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.dim // cfg.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=cfg.qkv_bias)
        self.attn_drop = nn.Dropout(cfg.attn_drop)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=True)
        self.proj_drop = nn.Dropout(cfg.proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x.ndim==3 with shape [B, N, C]. Got {tuple(x.shape)}")
        B, N, C = x.shape
        if C != self.dim:
            raise ValueError(f"Expected last dim C={self.dim}. Got C={C}")

        # qkv: [B, N, 3C] -> [B, N, 3, heads, head_dim] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention: [B, heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if getattr(self, "capture_attn", False):
          self.last_attn = attn.detach() 

        attn = self.attn_drop(attn)

        if getattr(self, "capture_attn", False):
          self.last_attn_postdrop = attn.detach()

        # out: [B, heads, N, head_dim] -> [B, N, C]
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class GridAttention2D(nn.Module):
    """
    Grid attention wrapper.

    Input/Output: x BHWC [B,H,W,C] -> [B,H,W,C]
    """
    def __init__(self, cfg: GridAttention2DConfig):
        super().__init__()
        if cfg.mode != "grid":
            raise ValueError("This minimal version only supports mode='grid'")
        self.cfg = cfg
        self.mhsa = MultiHeadSelfAttention(
            AttentionConfig(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                qkv_bias=cfg.qkv_bias,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x.ndim==4 (BHWC). Got {tuple(x.shape)}")
        B, H, W, C = x.shape
        if C != self.cfg.dim:
            raise ValueError(f"Expected C=={self.cfg.dim}. Got C={C}")

        g = self.cfg.grid_size
        grids, meta = grid_partition(x, g)         # [B*g*g, Hg, Wg, C]

        self._last_meta = meta
        self._last_grid_hw = (grids.shape[1], grids.shape[2])
        self._last_g = g

        Bgrp, Hg, Wg, _ = grids.shape
        tokens = grids.view(Bgrp, Hg * Wg, C)      # [Bgrp, N, C]
        tokens = self.mhsa(tokens)
        grids = tokens.view(Bgrp, Hg, Wg, C)
        out = grid_unpartition(grids, meta)
        return out