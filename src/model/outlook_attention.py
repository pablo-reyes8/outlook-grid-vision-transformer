import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(act: str) -> nn.Module:
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{act}'. Use one of: silu|gelu|relu")


class LayerNorm2d(nn.Module):
    """
    LayerNorm sobre el canal C para tensores [B, C, H, W].
    Normaliza por posición (h,w) a través de C (como LN de ViT).
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> [B, H, W, C] -> LN -> [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class MLP2d(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0, act="gelu"):
        super().__init__()
        hidden = max(1, int(dim * mlp_ratio))
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.act = make_activation(act)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class OutlookAttention2d(nn.Module):
    """
    OutlookAttention on [B,C,H,W] (NCHW) with dynamic local aggregation.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,):

        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and >0 (e.g., 3,5,7)")
        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kernel_size = kernel_size
        self.stride = stride

        kk = kernel_size * kernel_size
        bias = bool(qkv_bias)

        # logits per spatial position
        self.attn = nn.Conv2d(dim, num_heads * kk, kernel_size=1, bias=bias)
        # values
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.kernel_size
        s = self.stride
        heads = self.num_heads
        hd = self.head_dim
        kk = k * k

        # attn logits: [B, heads*kk, H, W] -> (optional) pool if stride>1
        a = self.attn(x)
        if s > 1:
            a = F.avg_pool2d(a, kernel_size=s, stride=s)
        _, _, Hs, Ws = a.shape

        # [B, heads, kk, Hs, Ws] -> [B, Hs*Ws, heads, kk]
        a = a.view(B, heads, kk, Hs, Ws).flatten(3).permute(0, 3, 1, 2).contiguous()
        a = F.softmax(a, dim=-1)
        a = self.attn_drop(a)

        # values + unfold neighborhoods
        v = self.v(x)  # [B,C,H,W]
        pad = k // 2
        v_unf = F.unfold(v, kernel_size=k, padding=pad, stride=s)  # [B, C*kk, Hs*Ws]

        # -> [B, Hs*Ws, heads, hd, kk]
        v_unf = v_unf.view(B, heads, hd, kk, Hs * Ws).permute(0, 4, 1, 2, 3).contiguous()

        # weighted sum over kk
        y = (v_unf * a.unsqueeze(3)).sum(dim=-1)  # [B, Hs*Ws, heads, hd]
        y = y.permute(0, 2, 3, 1).contiguous().view(B, C, Hs, Ws)

        y = self.proj(y)
        y = self.proj_drop(y)
        return y