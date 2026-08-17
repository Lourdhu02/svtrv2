"""SVTRv2 recognition network, faithful to arXiv 2411.15858v2.

Structural rules the paper is specific about, and which a looser implementation
tends to get wrong:

- A mixing block is **local or global, never both**.  Local mixing is two
  consecutive grouped convolutions with no normalization or activation between
  them; global mixing is MHSA.  Summing a conv branch onto an attention branch
  inside every block is a different architecture.
- Heads and conv groups are both **D_i / 32**.
- **No positional encoding anywhere**, which is what lets one set of weights
  accept all three MSR canvas sizes.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _svtrv2_heads(dim: int) -> int:
    return max(1, dim // 32)


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """MHSA/MHCA block built on fused scaled-dot-product attention.

    Uses ``F.scaled_dot_product_attention`` rather than an explicit
    ``softmax(q @ k.T) @ v``.  The fused kernel never materializes the
    ``(B, heads, N, N)`` score matrix, which is what dominates runtime on a
    bandwidth-bound part like GB10.  It also exports to ONNX without the
    trace-time shape constants that ``nn.MultiheadAttention`` bakes in, so
    dynamic MSR height/width still work.
    """

    def __init__(self, dim: int, num_heads: int, drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop_p = float(drop)
        self.proj_drop = nn.Dropout(drop)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))
        # Our masks are True == "block this position"; SDPA's bool mask is
        # True == "attend here", so invert.  A 2D (L, L) mask broadcasts over
        # batch and heads; a 4D (B, 1, L, L) mask is already broadcastable.
        mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            mask = ~attn_mask
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.proj_drop(self.out_proj(out))


class GlobalMixing(nn.Module):
    """SVTRv2 global mixing: multi-head self-attention over all tokens."""

    def __init__(self, dim: int, num_heads: Optional[int] = None,
                 drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads or _svtrv2_heads(dim)
        if dim % self.num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={self.num_heads}")
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop_p = float(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """x: (B, N, C) token sequence."""
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Fused SDPA: the (B, heads, N, N) score matrix is never materialized.
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0,
        ).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class LocalMixing(nn.Module):
    """SVTRv2 local mixing: "Conv^2", two consecutive grouped convolutions.

    Paper Sec. 3.2 and Suppl. Sec. 6: SVTR's sliding-window local attention is
    replaced by two consecutive grouped convolutions, with **no normalization
    or activation layer between them**, capturing character-level detail such
    as edges, textures and strokes.  Groups = D_i / 32 = the head count.

    The convolutions are 2D over the (H/8, W/4) feature map.  Convolving the
    flattened sequence instead would only ever reach horizontal neighbours and
    would wrap across row boundaries, which is not local mixing.
    """

    def __init__(self, dim: int, num_heads: Optional[int] = None,
                 local_k: Tuple[int, int] = (5, 5), drop: float = 0.0) -> None:
        super().__init__()
        groups = num_heads or _svtrv2_heads(dim)
        pad = (local_k[0] // 2, local_k[1] // 2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, local_k, 1, pad, groups=groups),
            nn.Conv2d(dim, dim, local_k, 1, pad, groups=groups),
        )
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        """x: (B, N, C) token sequence, reshaped to 2D for the convolutions."""
        b, n, c = x.shape
        h, w = hw
        y = x.transpose(1, 2).reshape(b, c, h, w)
        y = self.conv(y)
        return self.proj_drop(y.flatten(2).transpose(1, 2))


class SVTRBlock(nn.Module):
    """One mixing block: local or global, never both.

    The paper's [L]_m[G]_n permutation gives each block a single mixer -- early
    blocks mix locally, later ones globally.
    """

    def __init__(self, dim: int, num_heads: Optional[int] = None,
                 mixer: str = "Global", mlp_ratio: float = 4.0,
                 drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        if mixer not in ("Local", "Global"):
            raise ValueError(f"mixer must be 'Local' or 'Global', got {mixer!r}")
        self.mixer_type = mixer
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = (
            LocalMixing(dim, num_heads, drop=drop) if mixer == "Local"
            else GlobalMixing(dim, num_heads, drop=drop)
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        x = x + self.drop_path1(self.mixer(self.norm1(x), hw))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class FeatureRearrangementModule(nn.Module):
    """SVTRv2 FRM: rearranges 2D visual features into a reading-order sequence."""

    def __init__(self, dim: int, num_heads: Optional[int] = None, drop: float = 0.0,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        num_heads = num_heads or _svtrv2_heads(dim)
        self.horizontal_attn = MultiHeadAttention(dim, num_heads, drop)
        self.h_norm1 = nn.LayerNorm(dim)
        self.h_norm2 = nn.LayerNorm(dim)
        self.h_mlp = MLP(dim, int(dim * mlp_ratio), drop)

        self.select_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.vertical_attn = MultiHeadAttention(dim, num_heads, drop)
        self.v_norm = nn.LayerNorm(dim)
        self.v_mlp_norm = nn.LayerNorm(dim)
        self.v_mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Feature map of shape (B, D, H/8, W/4).

        Returns:
            Sequence of shape (B, W/4, D).
        """
        b, d, h, w = x.shape
        rows = x.permute(0, 2, 3, 1).reshape(b * h, w, d)
        attn_rows = self.horizontal_attn(rows, rows, rows)
        rows = self.h_norm1(rows + attn_rows)
        rows = self.h_norm2(rows + self.h_mlp(rows))
        fmap = rows.reshape(b, h, w, d)

        cols = fmap.permute(0, 2, 1, 3).reshape(b * w, h, d)
        query = self.select_token.expand(b * w, -1, -1)
        selected = self.vertical_attn(query, cols, cols)
        seq = selected.reshape(b, w, d)
        seq = self.v_norm(seq)
        return self.v_mlp_norm(seq + self.v_mlp(seq))


class SemanticGuidanceBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float = 0.0,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, num_heads, drop)
        self.drop1 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, num_heads, drop)
        self.drop2 = nn.Dropout(drop)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor, visual: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        y = self.norm1(x)
        y = self.self_attn(y, y, y, attn_mask=attn_mask)
        x = x + self.drop1(y)
        y = self.norm2(x)
        y = self.cross_attn(y, visual, visual)
        x = x + self.drop2(y)
        return x + self.mlp(self.norm3(x))


class SemanticGuidanceModule(nn.Module):
    """Training-only SVTRv2 semantic guidance module.

    Left and right context streams exclude the current target character, attend
    to the FRM visual sequence, and predict the target at each label position.
    """

    def __init__(self, dim: int, num_classes: int, num_heads: Optional[int] = None,
                 num_layers: int = 2, max_len: int = 32, drop: float = 0.0,
                 mlp_ratio: float = 4.0) -> None:
        super().__init__()
        num_heads = num_heads or _svtrv2_heads(dim)
        self.num_classes = num_classes
        self.max_len = max_len
        self.char_embed = nn.Embedding(num_classes, dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.side_embed = nn.Parameter(torch.zeros(2, 1, 1, dim))
        self.layers = nn.ModuleList(
            [SemanticGuidanceBlock(dim, num_heads, drop, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, num_classes)

    @staticmethod
    def _stream_mask(ids: torch.Tensor, forward: bool) -> torch.Tensor:
        """Attention mask for one context stream. True == blocked.

        Returns (B, 1, L, L), combining:
          - direction: the left stream may only look backwards, the right
            stream only forwards;
          - key padding: positions holding pad id 0 carry no context and must
            not be attended to.  Labels are padded to a fixed length (32) but
            are only ~6 characters, so without this the streams spend most of
            their attention on padding.

        The diagonal is always left open, otherwise a query whose entire
        context is padding would have every key masked and softmax would
        produce NaN.
        """
        b, length = ids.shape
        device = ids.device
        ones = torch.ones(length, length, dtype=torch.bool, device=device)
        direction = ones.triu(1) if forward else ones.tril(-1)
        blocked = direction.unsqueeze(0) | ids.eq(0).view(b, 1, length)
        blocked = blocked & ~torch.eye(length, dtype=torch.bool, device=device)
        return blocked.unsqueeze(1)

    def _position_slice(self, length: int, device: torch.device) -> torch.Tensor:
        if length <= self.max_len:
            return self.pos_embed[:, :length]
        extra = torch.zeros(1, length - self.max_len, self.pos_embed.size(-1), device=device)
        return torch.cat([self.pos_embed, extra], dim=1)

    def forward(self, visual: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            visual: FRM features, (B, T, D).
            targets: Padded CTC targets, (B, L), with 0 as padding.

        Returns:
            Logits shaped (B, 2, L, C), for left and right context streams.
        """
        b, length = targets.shape
        if length == 0:
            return visual.new_zeros(b, 2, 0, self.num_classes)

        # Position i predicts character i from its surrounding context only:
        # the left stream sees characters < i, the right stream sees > i.
        left_ids = torch.zeros_like(targets)
        left_ids[:, 1:] = targets[:, :-1]
        right_ids = torch.zeros_like(targets)
        right_ids[:, :-1] = targets[:, 1:]

        pos = self._position_slice(length, targets.device)
        left = self.char_embed(left_ids) + pos + self.side_embed[0]
        right = self.char_embed(right_ids) + pos + self.side_embed[1]

        # An anti-causal mask gives the right stream its forward context
        # in place, so no flip is needed and padding stays at the tail where
        # the key-padding mask handles it.
        left_mask = self._stream_mask(left_ids, forward=True)
        right_mask = self._stream_mask(right_ids, forward=False)
        for layer in self.layers:
            left = layer(left, visual, left_mask)
            right = layer(right, visual, right_mask)

        left_logits = self.proj(self.norm(left))
        right_logits = self.proj(self.norm(right))
        return torch.stack([left_logits, right_logits], dim=1)


class ContextualHead(SemanticGuidanceModule):
    """Backward-compatible name for the SVTRv2 semantic guidance module."""


class SVTRNet(nn.Module):
    """SVTRv2 network for cropped mechanical kWh meter OCR."""

    def __init__(self, num_classes: int = 12,
                 dims: Tuple[int, int, int] = (64, 128, 256),
                 depths: Tuple[int, int, int] = (3, 6, 3),
                 mixers: Optional[Tuple[Tuple[str, ...], ...]] = None,
                 heads: Optional[Tuple[int, int, int]] = None,
                 drop: float = 0.08, drop_path: float = 0.1,
                 sgm_layers: int = 2, sgm_max_len: int = 32,
                 mlp_ratio: float = 4.0, blank_bias: float = -2.0, **_) -> None:
        super().__init__()
        # Paper rule: heads = groups = D_i / 32.
        heads = tuple(_svtrv2_heads(d) for d in dims)

        # Default permutation follows the paper's [L]_m[G]_n for T/S: stage 1
        # all local, stage 2 half local then half global, stage 3 all global.
        if mixers is None:
            half = depths[1] // 2
            mixers = (
                ("Local",) * depths[0],
                ("Local",) * half + ("Global",) * (depths[1] - half),
                ("Global",) * depths[2],
            )
        if len(mixers) != 3 or any(len(m) != d for m, d in zip(mixers, depths)):
            raise ValueError(
                f"mixers {[len(m) for m in mixers]} must match depths {list(depths)}"
            )

        total_blocks = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, total_blocks)]

        # Stem: two strided 3x3 convs, each Conv-BN-GELU (paper Fig. 2).
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, dims[0] // 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        idx = 0
        self.stage1 = nn.ModuleList([
            SVTRBlock(dims[0], heads[0], mixer=mixers[0][i], mlp_ratio=mlp_ratio,
                      drop=drop, drop_path=dpr[idx + i])
            for i in range(depths[0])
        ])
        idx += depths[0]
        # Merges: 3x3 conv then LayerNorm (paper Fig. 2, official `sub_norm`).
        # sub_k = (2, 1) halves height only; (1, 1) keeps the map size.
        self.merge1 = nn.Conv2d(dims[0], dims[1], 3, (2, 1), 1)
        self.merge1_norm = nn.LayerNorm(dims[1])
        self.stage2 = nn.ModuleList([
            SVTRBlock(dims[1], heads[1], mixer=mixers[1][i], mlp_ratio=mlp_ratio,
                      drop=drop, drop_path=dpr[idx + i])
            for i in range(depths[1])
        ])
        idx += depths[1]
        # Stride 2 in width as well as the usual height reduction in merge1.
        # The paper emits W/4 timesteps because it recognises up to 25
        # characters; meter labels are 5-7, so W/4 gave T/L ~= 10.7 and CTC
        # settled into the all-blank minimum (at the true optimum ~90% of
        # frames are blank).  W/8 brings T/L to ~5 while leaving well above the
        # 2L-1 frames CTC needs to emit repeated digits.
        self.merge2 = nn.Conv2d(dims[1], dims[2], 3, (1, 2), 1)
        self.merge2_norm = nn.LayerNorm(dims[2])
        self.stage3 = nn.ModuleList([
            SVTRBlock(dims[2], heads[2], mixer=mixers[2][i], mlp_ratio=mlp_ratio,
                      drop=drop, drop_path=dpr[idx + i])
            for i in range(depths[2])
        ])
        self.mixers = mixers
        self.norm = nn.LayerNorm(dims[2])
        self.frm = FeatureRearrangementModule(dims[2], heads[2], drop, mlp_ratio=mlp_ratio)
        self.head = nn.Linear(dims[2], num_classes)
        self.sgm = SemanticGuidanceModule(
            dims[2], num_classes, heads[2], num_layers=sgm_layers,
            max_len=sgm_max_len, drop=drop, mlp_ratio=mlp_ratio,
        )
        self.feature_dim = dims[2]
        self.num_classes = num_classes
        self.blank_bias = blank_bias
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

        nn.init.trunc_normal_(self.frm.select_token, std=0.02)
        nn.init.trunc_normal_(self.sgm.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.sgm.side_embed, std=0.02)
        if self.head.bias is not None:
            with torch.no_grad():
                self.head.bias.zero_()
                self.head.bias[0] = self.blank_bias

    @staticmethod
    def _to_2d(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        return x.transpose(1, 2).contiguous().view(b, c, h, w)

    @staticmethod
    def _run_stage(blocks: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        """Run a stage over a (B, C, H, W) map.

        Blocks operate on the token sequence but local mixing needs the 2D
        shape, so (H, W) is threaded through explicitly.
        """
        h, w = x.shape[2], x.shape[3]
        seq = x.flatten(2).transpose(1, 2)
        for block in blocks:
            seq = block(seq, (h, w))
        return SVTRNet._to_2d(seq, h, w)

    @staticmethod
    def _apply_norm_2d(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the channel dim of a (B, C, H, W) map."""
        h, w = x.shape[2], x.shape[3]
        seq = norm(x.flatten(2).transpose(1, 2))
        return SVTRNet._to_2d(seq, h, w)

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # No absolute positional encoding anywhere: paper Sec. 3.2 drops it so
        # the model can handle multiple input sizes (MSR).  Positional
        # information comes from the grouped convolutions in local mixing.
        x = self.patch_embed(x)
        x = self._run_stage(self.stage1, x)
        x = self._apply_norm_2d(self.merge1_norm, self.merge1(x))
        x = self._run_stage(self.stage2, x)
        x = self._apply_norm_2d(self.merge2_norm, self.merge2(x))
        x = self._run_stage(self.stage3, x)
        h, w = x.shape[2], x.shape[3]
        seq = self.norm(x.flatten(2).transpose(1, 2))
        return self._to_2d(seq, h, w)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.frm(self.forward_feature_map(x))

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward_sgm(self, features: torch.Tensor, padded_targets: torch.Tensor) -> torch.Tensor:
        return self.sgm(features, padded_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        # fp32 head: keeps CTC log-probs numerically stable under AMP.
        return self.head(features.float()).float().log_softmax(dim=2)

    def load_pretrained(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        current = self.state_dict()
        keep = {
            k: v for k, v in state.items()
            if k in current and tuple(v.shape) == tuple(current[k].shape)
            and not k.startswith(("head.", "sgm."))
        }
        self.load_state_dict(keep, strict=False)
