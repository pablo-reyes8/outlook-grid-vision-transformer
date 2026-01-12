import torch

from src.stage_config import StageCfg
from src.Model_A_OutGridNet import MaxOutNet
from src.Model_B_OutGridNet import OutlookerFrontGridNet


def _make_stages():
    return [
        StageCfg(
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
        ),
        StageCfg(
            dim=32,
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
        ),
    ]


def _assert_logits(logits: torch.Tensor, batch: int, num_classes: int):
    assert logits.shape == (batch, num_classes)
    assert torch.isfinite(logits).all()


def test_model_a_forward():
    stages = _make_stages()
    model = MaxOutNet(
        num_classes=10,
        stages=stages,
        stem_dim=16,
        dpr_max=0.0,
    )

    x = torch.randn(2, 3, 8, 8)
    y = model(x)
    _assert_logits(y, 2, 10)


def test_model_b_forward():
    stages = _make_stages()
    model = OutlookerFrontGridNet(
        num_classes=10,
        stages=stages,
        stem_dim=16,
        outlooker_front_depth=1,
        dpr_max=0.0,
    )

    x = torch.randn(2, 3, 8, 8)
    y = model(x)
    _assert_logits(y, 2, 10)
