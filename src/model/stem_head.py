from dataclasses import dataclass
from typing import List
import torch.nn as nn


def _make_activation(act) -> nn.Module:
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{act}'. Use one of: silu|gelu|relu")


def make_dpr(total_blocks: int, dpr_max: float) -> List[float]:
    if total_blocks <= 1:
        return [dpr_max]
    return [dpr_max * i / (total_blocks - 1) for i in range(total_blocks)]


class ConvStem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, act: str = "silu", use_bn: bool = True):
        super().__init__()
        bn = (lambda c: nn.BatchNorm2d(c)) if use_bn else (lambda c: nn.Identity())
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=not use_bn),
            bn(out_ch), _make_activation(act),)

    def forward(self, x):
        return self.stem(x)