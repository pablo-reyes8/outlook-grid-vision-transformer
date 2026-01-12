from src.model.Out_Grid_Block import *
from src.model.downsampling import *
from src.model.stem_head import *
from src.stage_config import *




class MaxOutNet(nn.Module):
    """
    Modelo A:
      Stem -> (Stage: OutGridBlock x depth + Downsample) -> Head
    """
    def __init__(
        self,
        num_classes: int,
        stages: List[StageCfg],
        in_ch: int = 3,
        stem_dim: int = 64,
        dpr_max: float = 0.1,
        down_cfg: DownsampleConfig = DownsampleConfig(kind="conv", act="silu", use_bn=True),):

        super().__init__()
        assert len(stages) >= 1
        self.stem = ConvStem(in_ch, stem_dim, act="silu", use_bn=True)

        self.proj_in = nn.Identity()

        if stem_dim != stages[0].dim:
            self.proj_in = nn.Conv2d(stem_dim, stages[0].dim, kernel_size=1, bias=True)

        total_blocks = sum(s.depth for s in stages)
        dprs = make_dpr(total_blocks, dpr_max)
        idx = 0

        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for si, scfg in enumerate(stages):
            
            blocks = nn.ModuleList()
            for _ in range(scfg.depth):
                bcfg = StageCfg(**{**scfg.__dict__, "drop_path": dprs[idx]})
                blocks.append(OutGridBlock(bcfg))

                idx += 1
            self.stages.append(blocks)

            if si < len(stages) - 1:
                self.downs.append(Downsample(scfg.dim, stages[si+1].dim, cfg=down_cfg))

        self.head_norm = nn.BatchNorm2d(stages[-1].dim)
        self.classifier = nn.Linear(stages[-1].dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.proj_in(x)

        for si, blocks in enumerate(self.stages):
            for blk in blocks:
                x = blk(x)
            if si < len(self.downs):
                x = self.downs[si](x)

        x = self.head_norm(x)
        x = x.mean(dim=(2, 3))
        return self.classifier(x)
