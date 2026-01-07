import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.Outlook_Block import * 
from src.model.grid_attention import *
from src.model.mbc_conv import *
from src.model.Out_Grid_Block import *
from src.model.grid_attention import *

class MaxOutStage(nn.Module):
    def __init__(self, block_cfg, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([OutGridBlock(block_cfg) for _ in range(depth)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

class GridOnlyBlock(nn.Module):
    """
    MBConv -> Grid-MHSA -> MLP (sin window attn).
    Input/Output: [B,C,H,W]
    """
    def __init__(self, cfg):
        super().__init__()
        C = cfg.dim

        self.mbconv = MBConv(
            in_ch=C, out_ch=C, stride=1,
            cfg=MBConvConfig(
                expand_ratio=cfg.mbconv_expand_ratio,
                se_ratio=cfg.mbconv_se_ratio,
                act=cfg.mbconv_act,
                use_bn=cfg.use_bn,
                drop_path=0.0,
            ))

        self.norm2 = nn.LayerNorm(C)
        self.grid_attn = GridAttention2D(
            GridAttention2DConfig(
                mode="grid",
                dim=C,
                num_heads=cfg.num_heads,
                window_size=getattr(cfg, "window_size", 1),
                grid_size=cfg.grid_size,
                qkv_bias=True,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
            ))
        
        self.dp2 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

        self.norm3 = nn.LayerNorm(C)
        self.mlp = MLP(dim=C, mlp_ratio=cfg.mlp_ratio, drop=cfg.ffn_drop, act=cfg.mlp_act)
        self.dp3 = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.mbconv(x)

        x_bhwc = x.permute(0, 2, 3, 1).contiguous()

        y = self.norm2(x_bhwc)
        y = self.grid_attn(y)
        x_bhwc = x_bhwc + self.dp2(y)

        y = self.norm3(x_bhwc)
        y = self.mlp(y)
        x_bhwc = x_bhwc + self.dp3(y)

        return x_bhwc.permute(0, 3, 1, 2).contiguous()
    

class StageOutThenGrid(nn.Module):
    """
    Outlooker una vez al inicio del stage, luego varios GridOnlyBlock.
    """
    def __init__(self, cfg, depth: int, out_depth: int = 1):
        super().__init__()
        self.outlookers = nn.ModuleList([
            OutlookerBlock2d(
                dim=cfg.dim,
                num_heads=cfg.outlook_heads,
                kernel_size=cfg.outlook_kernel,
                stride=1,
                mlp_ratio=cfg.outlook_mlp_ratio,
                attn_drop=cfg.attn_drop,
                proj_drop=cfg.proj_drop,
                mlp_drop=cfg.ffn_drop,
                drop_path=cfg.drop_path,
                act=cfg.mlp_act,)

            for _ in range(out_depth)])

        self.blocks = nn.ModuleList([GridOnlyBlock(cfg) for _ in range(depth)])

    def forward(self, x):
        for o in self.outlookers:
            x = o(x)
        for b in self.blocks:
            x = b(x)
        return x