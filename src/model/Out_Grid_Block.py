import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.Outlook_Block import * 
from src.model.grid_attention import *
from src.model.mbc_conv import *


class MLP(nn.Module):
    """
    MLP para BHWC: aplica sobre el último dim C.
    x: [..., C] -> [..., C]
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, act: str = "gelu"):
        super().__init__()
        hidden = max(1, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden)
        self.act = make_activation(act)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.fc1.in_features:
            raise ValueError(f"MLP expected last dim={self.fc1.in_features}, got {x.shape[-1]}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class OutGridBlock(nn.Module):
    """
    Híbrido: Outlooker (local dinámico) -> MBConv -> Grid-MHSA -> MLP
    Input/Output: [B, C, H, W]
    """
    def __init__(self, cfg):
        super().__init__()
        C = cfg.dim

        # Outlooker en NCHW
        self.outlook = OutlookerBlock2d(
            dim=C,
            num_heads=cfg.outlook_heads,          # nuevo hyperparam
            kernel_size=cfg.outlook_kernel,       # nuevo hyperparam
            stride=1,
            mlp_ratio=cfg.outlook_mlp_ratio,      # opcional, puedes fijar 0 o 2
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            mlp_drop=cfg.ffn_drop,
            drop_path=cfg.drop_path,
            act=cfg.mlp_act,)

        # MBConv NCHW
        self.mbconv = MBConv(
            in_ch=C, out_ch=C, stride=1,
            cfg=MBConvConfig(
                expand_ratio=cfg.mbconv_expand_ratio,
                se_ratio=cfg.mbconv_se_ratio,
                act=cfg.mbconv_act,
                use_bn=cfg.use_bn,
                drop_path=0.0,
            ),)

        # Grid attention BHWC
        self.norm2 = nn.LayerNorm(C)
        self.grid_attn = GridAttention2D(
            GridAttention2DConfig(
                mode="grid",
                dim=C,
                num_heads=cfg.num_heads,
                window_size=cfg.window_size,
                grid_size=cfg.grid_size,
                qkv_bias=True,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
            ))
        self.dp2 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

        # MLP BHWC
        self.norm3 = nn.LayerNorm(C)
        self.mlp = MLP(dim=C, mlp_ratio=cfg.mlp_ratio, drop=cfg.ffn_drop, act=cfg.mlp_act)
        self.dp3 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # Outlooker + MBConv (NCHW)
        x = self.outlook(x)
        x = self.mbconv(x)

        # to BHWC for grid + mlp
        x_bhwc = x.permute(0, 2, 3, 1).contiguous()

        y = self.norm2(x_bhwc)
        y = self.grid_attn(y)
        x_bhwc = x_bhwc + self.dp2(y)

        y = self.norm3(x_bhwc)
        y = self.mlp(y)
        x_bhwc = x_bhwc + self.dp3(y)

        # back to NCHW
        return x_bhwc.permute(0, 3, 1, 2).contiguous()