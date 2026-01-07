import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.outlook_attention import *

class DropPath(nn.Module):
    """
    DropPath / Stochastic Depth. Works for any tensor shape with batch in dim 0.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or (not self.training):
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: [B, 1, 1, 1, ...]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep_prob)
        return x * mask / keep_prob



class OutlookerBlock2d(nn.Module):
    """
    x (NCHW) -> LN2d -> OutlookAttention2d -> DropPath + res
             -> LN2d -> MLP2d            -> DropPath + res
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int = 3,
        stride: int = 1,
        mlp_ratio: float = 2.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        mlp_drop: float = 0.0,
        act: str = "gelu",
        norm_eps: float = 1e-6):

        super().__init__()
        self.norm1 = LayerNorm2d(dim, eps=norm_eps)
        self.attn = OutlookAttention2d(
            dim=dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            attn_drop=attn_drop,
            proj_drop=proj_drop)

        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = LayerNorm2d(dim, eps=norm_eps)
        self.mlp = MLP2d(dim=dim, mlp_ratio=mlp_ratio, drop=mlp_drop, act=act)
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.attn(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x