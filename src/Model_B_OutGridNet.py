
from dataclasses import dataclass
from typing import List
import torch.nn as nn

from src.model.stem_head import * 
from src.model.Grid_Only_Block import *
from src.model.downsampling import * 
from src.stage_config import *

class OutlookerFrontGridNet(nn.Module):
    """
    Modelo A:
      Stem -> OutlookerFront (L bloques) -> (Stage: GridOnlyBlock x depth + Downsample) -> Head
    """
    def __init__(
        self,
        num_classes: int,
        stages: List[StageCfg],
        in_ch: int = 3,
        stem_dim: int = 64,
        outlooker_front_depth: int = 2,   # <- varios outlookers "tipo VOLO"
        dpr_max: float = 0.1,
        down_cfg: DownsampleConfig = DownsampleConfig(kind="conv", act="silu", use_bn=True),):

        super().__init__()
        assert len(stages) >= 1
        self.stem = ConvStem(in_ch, stem_dim, act="silu", use_bn=True)

        # proyección para entrar a dim del stage1 si stem_dim != stage1.dim
        self.proj_in = nn.Identity()
        if stem_dim != stages[0].dim:
            self.proj_in = nn.Conv2d(stem_dim, stages[0].dim, kernel_size=1, bias=True)

        # schedule global de drop_path por bloque (front + sum(stage.depth))
        total_blocks = outlooker_front_depth + sum(s.depth for s in stages)
        dprs = make_dpr(total_blocks, dpr_max)
        idx = 0

        # Outlooker front (NCHW) con residual + DropPath interno
        front_cfg = stages[0]
        self.front = nn.ModuleList()
        for _ in range(outlooker_front_depth):
            c = front_cfg
            self.front.append(
                OutlookerBlock2d(
                    dim=c.dim,
                    num_heads=c.outlook_heads,
                    kernel_size=c.outlook_kernel,
                    stride=1,
                    mlp_ratio=c.outlook_mlp_ratio,
                    attn_drop=c.attn_drop,
                    proj_drop=c.proj_drop,
                    mlp_drop=c.ffn_drop,
                    drop_path=dprs[idx],
                    act=c.mlp_act,))

            idx += 1

        # stages: GridOnlyBlock stacks + downsample between stages
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for si, scfg in enumerate(stages):
            blocks = nn.ModuleList()
            for _ in range(scfg.depth):
                # clonar cfg pero con drop_path asignado por bloque
                bcfg = StageCfg(**{**scfg.__dict__, "drop_path": dprs[idx]})
                blocks.append(GridOnlyBlock(bcfg))
                idx += 1
            self.stages.append(blocks)

            # downsample (except after last stage)
            if si < len(stages) - 1:
                self.downs.append(Downsample(scfg.dim, stages[si+1].dim, cfg=down_cfg))

        # head
        self.head_norm = nn.BatchNorm2d(stages[-1].dim)
        self.classifier = nn.Linear(stages[-1].dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.proj_in(x)

        # front outlooker
        for blk in self.front:
            x = blk(x)

        # grid-only stages
        for si, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if si < len(self.downs):
                x = self.downs[si](x)

        # global pool + cls
        x = self.head_norm(x)
        x = x.mean(dim=(2, 3))

        return self.classifier(x)