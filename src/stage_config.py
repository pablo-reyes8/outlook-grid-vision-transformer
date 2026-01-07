from dataclasses import dataclass
from typing import List


@dataclass
class StageCfg:
    # core dims
    dim: int
    depth: int

    # grid attention
    num_heads: int
    grid_size: int
    window_size: int = 8  # no se usa en grid-only, pero lo mantenemos compatible

    # outlooker
    outlook_heads: int = 6
    outlook_kernel: int = 3
    outlook_mlp_ratio: float = 2.0

    # MBConv
    mbconv_expand_ratio: float = 4.0
    mbconv_se_ratio: float = 0.25
    mbconv_act: str = "silu"
    use_bn: bool = True

    # drops
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    ffn_drop: float = 0.0
    drop_path: float = 0.0

    # MLP (BHWC)
    mlp_ratio: float = 4.0
    mlp_act: str = "gelu"