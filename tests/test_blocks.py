import torch

from src.model.grid_partition import grid_partition, grid_unpartition
from src.model.outlook_attention import OutlookAttention2d
from src.model.Out_Grid_Block import OutGridBlock
from src.stage_config import StageCfg


def _make_cfg():
    return StageCfg(
        dim=16,
        depth=1,
        num_heads=4,
        grid_size=2,
        window_size=4,
        outlook_heads=4,
        outlook_kernel=3,
        outlook_mlp_ratio=2.0,
        mbconv_expand_ratio=2.0,
        mbconv_se_ratio=0.25,
        mbconv_act="silu",
        use_bn=True,
        attn_drop=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        mlp_ratio=2.0,
        mlp_act="gelu",
    )


def test_grid_partition_roundtrip():
    x = torch.randn(2, 8, 8, 6)
    grids, meta = grid_partition(x, grid_size=2)
    out = grid_unpartition(grids, meta)
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_outlook_attention_shapes():
    attn = OutlookAttention2d(dim=16, num_heads=4, kernel_size=3, stride=1)
    x = torch.randn(2, 16, 8, 8)
    y = attn(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_outgrid_block_forward_matches_manual():
    cfg = _make_cfg()
    block = OutGridBlock(cfg).eval()
    x = torch.randn(2, cfg.dim, 8, 8)

    with torch.no_grad():
        y = block.outlook(x)
        y = block.mbconv(y)
        x_bhwc = y.permute(0, 2, 3, 1).contiguous()

        z = block.norm2(x_bhwc)
        z = block.grid_attn(z)
        x_bhwc = x_bhwc + block.dp2(z)

        z = block.norm3(x_bhwc)
        z = block.mlp(z)
        x_bhwc = x_bhwc + block.dp3(z)

        manual = x_bhwc.permute(0, 3, 1, 2).contiguous()
        direct = block(x)

    assert direct.shape == x.shape
    assert torch.isfinite(direct).all()
    assert torch.allclose(manual, direct, atol=1e-6, rtol=1e-5)
