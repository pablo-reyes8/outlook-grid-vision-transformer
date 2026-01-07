@dataclass
class DummyCfg:
    dim: int = 96

    # Outlooker
    outlook_heads: int = 6
    outlook_kernel: int = 3
    outlook_mlp_ratio: float = 2.0

    # MBConv
    mbconv_expand_ratio: float = 4.0
    mbconv_se_ratio: float = 0.25
    mbconv_act: str = "silu"
    use_bn: bool = True

    # Grid MHSA
    num_heads: int = 6
    grid_size: int = 4
    window_size: int = 8  # no se usa en grid-only, pero tu ctor lo pasa

    # Drops
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    ffn_drop: float = 0.0
    drop_path: float = 0.0

    # MLP (BHWC)
    mlp_ratio: float = 4.0
    mlp_act: str = "gelu"

def _assert_shape(x: torch.Tensor, shape: tuple, name: str = "tensor"):
    assert tuple(x.shape) == tuple(shape), f"{name}: expected shape {shape}, got {tuple(x.shape)}"

def _assert_ndim(x: torch.Tensor, ndim: int, name: str = "tensor"):
    assert x.ndim == ndim, f"{name}: expected ndim={ndim}, got ndim={x.ndim}, shape={tuple(x.shape)}"

def _assert_finite(x: torch.Tensor, name: str = "tensor"):
    assert torch.isfinite(x).all().item(), f"{name}: found non-finite values (nan/inf)"

def _assert_divisible_hw(H: int, W: int, g: int):
    assert (H % g) == 0 and (W % g) == 0, f"H,W must be divisible by grid_size g={g}. Got H={H}, W={W}"


@torch.no_grad()
def test_outlooker_stage(block: OutGridBlock, x: torch.Tensor):
    _assert_ndim(x, 4, "x")
    B, C, H, W = x.shape
    _assert_shape(x, (B, block.outlook.norm1.ln.normalized_shape[0], H, W), "x (pre)")  # C check

    y = block.outlook(x)
    _assert_shape(y, (B, C, H, W), "outlook(x)")
    _assert_finite(y, "outlook(x)")
    return y

@torch.no_grad()
def test_mbconv_stage(block: OutGridBlock, x: torch.Tensor):
    B, C, H, W = x.shape
    y = block.mbconv(x)
    _assert_shape(y, (B, C, H, W), "mbconv(x)")
    _assert_finite(y, "mbconv(x)")
    return y


@torch.no_grad()
def test_grid_stage(block: OutGridBlock, x_nchw: torch.Tensor):
    B, C, H, W = x_nchw.shape
    x_bhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    _assert_shape(x_bhwc, (B, H, W, C), "x_bhwc")

    # divisibilidad
    g = block.grid_attn.cfg.grid_size
    _assert_divisible_hw(H, W, g)

    y = block.norm2(x_bhwc)
    _assert_shape(y, (B, H, W, C), "norm2(x_bhwc)")
    y = block.grid_attn(y)
    _assert_shape(y, (B, H, W, C), "grid_attn(norm2(x_bhwc))")
    _assert_finite(y, "grid_attn output")

    out = x_bhwc + block.dp2(y)
    _assert_shape(out, (B, H, W, C), "residual after grid")
    _assert_finite(out, "after grid residual")

    return out  # BHWC


@torch.no_grad()
def test_mlp_stage(block: OutGridBlock, x_bhwc: torch.Tensor):
    B, H, W, C = x_bhwc.shape

    y = block.norm3(x_bhwc)
    _assert_shape(y, (B, H, W, C), "norm3(x_bhwc)")
    y = block.mlp(y)
    _assert_shape(y, (B, H, W, C), "mlp(norm3(x_bhwc))")
    _assert_finite(y, "mlp output")

    out = x_bhwc + block.dp3(y)
    _assert_shape(out, (B, H, W, C), "residual after mlp")
    _assert_finite(out, "after mlp residual")
    return out

@torch.no_grad()
def test_full_forward_matches_stages(block: OutGridBlock, x: torch.Tensor, atol=1e-6, rtol=1e-5):
    block.eval()

    # manual pipeline
    a = test_outlooker_stage(block, x)
    b = test_mbconv_stage(block, a)
    c = test_grid_stage(block, b)         # BHWC
    d = test_mlp_stage(block, c)          # BHWC
    manual = d.permute(0, 3, 1, 2).contiguous()

    # direct forward
    direct = block(x)

    _assert_shape(direct, x.shape, "block(x)")
    _assert_finite(direct, "block(x)")
    assert torch.allclose(manual, direct, atol=atol, rtol=rtol), \
        "Manual staged pipeline != block.forward output (check wiring/residuals/norms)."

    return direct

def run_all_tests():
    torch.manual_seed(0)

    cfg = DummyCfg(dim=96, grid_size=4)
    blk = MaxOutBlock(cfg).eval()

    x = torch.randn(2, 96, 16, 16)
    assert x.shape[2] % cfg.grid_size == 0 and x.shape[3] % cfg.grid_size == 0

    y = test_full_forward_matches_stages(blk, x)
    print("All tests passed. y:", y.shape)