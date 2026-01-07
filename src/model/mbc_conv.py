from typing import Literal
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.outlook_block import *

class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, se_ratio: float = 0.25, act: str = "silu"):
        super().__init__()
        if not (0.0 < se_ratio <= 1.0):
            raise ValueError("se_ratio must be in (0, 1].")

        hidden = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = make_activation(act)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        return x * self.gate(s)


ActType = Literal["silu", "gelu", "relu"]

@dataclass(frozen=True)
class MBConvConfig:
    expand_ratio: float = 4.0
    se_ratio: float = 0.25
    act: ActType = "silu"
    use_bn: bool = True
    drop_path: float = 0.0





class MBConv(nn.Module):
    """
    MBConv block (NCHW):
      Expand 1x1 -> Depthwise 3x3 -> SE -> Project 1x1
      Residual if stride=1 and in_ch==out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, cfg: MBConvConfig = MBConvConfig()):
        super().__init__()
        if in_ch <= 0 or out_ch <= 0:
            raise ValueError("in_ch and out_ch must be > 0")
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        bn = (lambda c: nn.BatchNorm2d(c)) if cfg.use_bn else (lambda c: nn.Identity())
        act = make_activation(cfg.act)

        mid_ch = max(1, int(round(in_ch * cfg.expand_ratio)))

        if mid_ch != in_ch:
            self.expand = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=not cfg.use_bn),
                bn(mid_ch),
                act,)

        else:
            self.expand = nn.Identity()

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1,
                      groups=mid_ch, bias=not cfg.use_bn),
            bn(mid_ch),
            act,)

        self.se = SqueezeExcite(mid_ch, se_ratio=cfg.se_ratio, act=cfg.act) if cfg.se_ratio > 0 else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=not cfg.use_bn),
            bn(out_ch),)

        self.use_res = (stride == 1 and in_ch == out_ch)
        self.drop_path = DropPath(cfg.drop_path) if (cfg.drop_path and cfg.drop_path > 0) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_res:
            out = x + self.drop_path(out)
        return out