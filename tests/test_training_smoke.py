import torch
from torch.utils.data import DataLoader, TensorDataset

from src.Model_A_OutGridNet import MaxOutNet
from src.stage_config import StageCfg
from src.training.train_full_model import train_model


def _make_stage():
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


def test_train_model_smoke(tmp_path):
    torch.manual_seed(0)
    stages = [_make_stage()]
    model = MaxOutNet(num_classes=10, stages=stages, stem_dim=16, dpr_max=0.0)

    images = torch.randn(8, 3, 8, 8)
    labels = torch.randint(0, 10, (8,))
    train_loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    val_loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)

    save_path = tmp_path / "best.pt"
    last_path = tmp_path / "last.pt"

    history, trained = train_model(
        model=model,
        train_loader=train_loader,
        epochs=1,
        val_loader=val_loader,
        device="cpu",
        lr=1e-3,
        weight_decay=0.0,
        autocast_dtype="fp32",
        use_amp=False,
        grad_clip_norm=None,
        warmup_ratio=0.0,
        min_lr=0.0,
        label_smoothing=0.0,
        print_every=0,
        save_path=str(save_path),
        last_path=str(last_path),
        resume_path=None,
        mixup_alpha=0.0,
        cutmix_alpha=0.0,
        mix_prob=0.0,
        num_classes=10,
        channels_last=False,
        early_stop=False,
    )

    assert trained is model
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
    assert save_path.exists()
    assert last_path.exists()
