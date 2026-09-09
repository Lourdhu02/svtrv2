"""ARD part 1 — adaptive MSR routing (novel).

Static MSR assigns each crop a canvas from hand-set aspect-ratio edges.  That
is blind to what actually matters: which canvas lets the *recognizer* read the
sample best.  Curved or sparse text with a moderate aspect ratio often reads
better on a wider canvas than the geometric rule suggests.

This module learns that choice.  A tiny router network sees a cheap,
fixed-size probe of the crop (32x128 fit-pad, a tiny fraction of a backbone
forward) plus the aspect ratio and predicts one MSR bin per sample.  It is
trained with a **loss-based preference objective**: every ``explore_every``
steps a small subset of the batch is materialized on *two* candidate canvases
(exact fit-pad from the original image, no double-resize approximation),
forwarded through the current model, and the router is pushed to prefer the
canvas with the lower CTC loss.  Decisiveness is kept by an entropy
regularizer; an optional geometric prior keeps the router near the static rule
early in training.

At inference the router replaces the static rule at negligible cost and the
recognition graph stays CTC-only — routing adds no linguistic context and no
extra decode machinery.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MSR_BINS

PROBE_H, PROBE_W = 32, 128
NUM_BINS = len(MSR_BINS)


# ------------------------------------------------------------------ probe feed


def probe_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Fit-pad an RGB crop to the fixed probe canvas and normalize.

    Same normalization contract as the training transform (mean/std 0.5), no
    augmentation: the router must see the honest geometry of the crop.
    """
    from .transforms import fit_pad

    padded = fit_pad(image_rgb, PROBE_H, PROBE_W, pad_mode="edge")
    x = padded.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def canvas_from_image(image_rgb: np.ndarray, bin_index: int,
                      pad_mode: str = "edge") -> np.ndarray:
    """Exact fit-pad of an RGB crop onto MSR bin ``bin_index``'s canvas."""
    from .transforms import fit_pad

    b = MSR_BINS[bin_index]
    return fit_pad(image_rgb, int(b["height"]), int(b["width"]), pad_mode=pad_mode)


def canvas_to_tensor(canvas: np.ndarray) -> torch.Tensor:
    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def geometric_prior(aspect_ratios: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Static-rule bin distribution for a batch of aspect ratios.

    One-hot of the hand-set edges (first bin whose ``max_ar`` exceeds the crop
    ratio, matching ``select_msr_bin``), label-smoothed by ``eps`` so the prior
    never pins the router to the geometric rule.
    """
    edges = torch.tensor([b["max_ar"] for b in MSR_BINS[:-1]],
                         device=aspect_ratios.device)
    idx = torch.bucketize(aspect_ratios, edges, right=True)
    b = aspect_ratios.shape[0]
    prior = torch.full((b, NUM_BINS), eps / NUM_BINS, device=aspect_ratios.device)
    prior[torch.arange(b, device=idx.device), idx] += 1.0 - eps
    return prior


# ---------------------------------------------------------------------- router


class MSRRouter(nn.Module):
    """Tiny per-sample canvas router.

    Input: the (B, 3, 32, 128) probe canvas plus the raw aspect ratio.  Three
    strided depthwise-separable conv stages (each 2x downsample) end in a
    global pool; the pooled appearance feature is fused with a small MLP
    embedding of the aspect ratio and mapped to one logit per MSR bin.  The
    whole router is well under 40k parameters -- noise next to the backbone --
    and its probe forward is a tiny fraction of one recognition pass.
    """

    def __init__(self, width: int = PROBE_W, channels: int = 32,
                 ar_dim: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            self._dw_pw(16, 32, stride=2),
            self._dw_pw(32, channels, stride=2),
        ])
        self.ar_mlp = nn.Sequential(nn.Linear(1, ar_dim), nn.GELU(),
                                    nn.Linear(ar_dim, ar_dim))
        self.head = nn.Sequential(
            nn.Linear(channels + ar_dim, 64), nn.GELU(), nn.Linear(64, NUM_BINS),
        )

    @staticmethod
    def _dw_pw(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            # depthwise
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.GELU(),
            # pointwise
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU(),
        )

    def forward(self, probe: torch.Tensor, aspect_ratios: torch.Tensor) -> torch.Tensor:
        """Returns router logits (B, NUM_BINS)."""
        x = self.stem(probe)
        for blk in self.blocks:
            x = blk(x)
        pooled = x.mean(dim=(2, 3))
        ar = self.ar_mlp(aspect_ratios.view(-1, 1).float())
        return self.head(torch.cat([pooled, ar], dim=1))


# ------------------------------------------------------- assignment utilities


def gumbel_noise(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel perturbation for exploration-style categorical sampling."""
    return logits + torch.empty_like(logits).exponential_().log().neg()


def route_bins(logits: torch.Tensor, hard: bool = True,
               tau: float = 1.0) -> torch.Tensor:
    """Turn router logits into a (B, NUM_BINS) assignment matrix.

    Hard mode is straight-through: the forward pass is a one-hot argmax (each
    sample really lands on one canvas) while gradients flow through the soft
    distribution.  ``hard=False`` returns the plain softmax, for regularizers
    and diagnostics.
    """
    soft = torch.softmax(logits / tau, dim=-1)
    if not hard:
        return soft
    hard_idx = soft.argmax(dim=-1)
    one_hot = F.one_hot(hard_idx, logits.shape[-1]).to(logits.dtype)
    return one_hot + (soft - soft.detach())


def soft_route_sample(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Differentiable Gumbel-softmax sample, straight-through."""
    soft = torch.softmax(gumbel_noise(logits) / tau, dim=-1)
    hard_idx = soft.argmax(dim=-1)
    one_hot = F.one_hot(hard_idx, logits.shape[-1]).to(logits.dtype)
    return one_hot + (soft - soft.detach())


# ------------------------------------------------------------------ objectives


class RouterLoss(nn.Module):
    """Loss-based preference objective for the router.

    Preference term: given two candidate canvases per sample with measured CTC
    losses, push the router logit of the better canvas above the worse one
    (pairwise logistic loss -- a Bradley-Terry view of "which canvas reads
    this sample best").  Entropy term: keep assignments decisive.  Prior term
    (optional): keep the router distribution close to the smoothed static
    geometric rule early in training.
    """

    def __init__(self, weight: float = 0.5, entropy: float = 0.01,
                 prior: float = 0.0) -> None:
        super().__init__()
        self.weight = float(weight)
        self.entropy = float(entropy)
        self.prior = float(prior)

    def preference(self, logits: torch.Tensor, bin_a: torch.Tensor,
                   bin_b: torch.Tensor, loss_a: torch.Tensor,
                   loss_b: torch.Tensor) -> torch.Tensor:
        """Push the router toward the canvas with the lower measured CTC loss."""
        better_is_a = (loss_a <= loss_b).float()
        z_a = logits.gather(1, bin_a.view(-1, 1)).squeeze(1)
        z_b = logits.gather(1, bin_b.view(-1, 1)).squeeze(1)
        # margin positive when the better canvas already wins
        margin = better_is_a * (z_a - z_b) + (1.0 - better_is_a) * (z_b - z_a)
        return F.softplus(-margin).mean()

    def entropy_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Negative entropy: minimizing it makes the router decisive."""
        p = torch.softmax(logits, dim=-1)
        ent = -(p.clamp_min(1e-8).log() * p).sum(-1)
        return ent.mean()

    def prior_penalty(self, logits: torch.Tensor,
                      aspect_ratios: torch.Tensor) -> torch.Tensor:
        prior = geometric_prior(aspect_ratios)
        log_p = torch.log_softmax(logits, dim=-1)
        return -(prior * log_p).sum(-1).mean()

    def forward(self, logits: torch.Tensor, aspect_ratios: torch.Tensor,
                preference: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """Total router loss; ``preference`` may be None outside explore steps."""
        terms = dict(pref=logits.new_zeros(()),
                     ent=self.entropy_penalty(logits))
        if preference is not None:
            terms["pref"] = preference
        loss = self.weight * terms["pref"] + self.entropy * terms["ent"]
        if self.prior > 0.0:
            terms["prior"] = self.prior_penalty(logits, aspect_ratios)
            loss = loss + self.prior * terms["prior"]
        terms["total"] = loss
        return loss, terms


# -------------------------------------------------------------------- serving


def load_router(ckpt: dict, device: Optional[torch.device] = None) -> Optional[MSRRouter]:
    """Rebuild the router from a training checkpoint, or None if it has none."""
    state = ckpt.get("router_state")
    if not state:
        return None
    router = MSRRouter()
    router.load_state_dict(state)
    if device is not None:
        router = router.to(device)
    router.eval()
    return router


def routed_bin_name(probe: torch.Tensor, aspect_ratio: float,
                    router: MSRRouter) -> str:
    """Single-sample routing for serving code paths."""
    with torch.no_grad():
        logits = router(probe.unsqueeze(0),
                        torch.tensor([aspect_ratio], device=probe.device,
                                     dtype=probe.dtype))
        return str(MSR_BINS[int(logits.argmax(1).item())]["name"])
