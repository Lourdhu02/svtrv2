"""CTC and rolling-counter-aware loss variants."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _roll_up(class_idx: int) -> int:
    """Next digit on the wheel: class for digit d -> class for (d+1) mod 10.

    Class indices: blank=0, digits '0'..'9' -> 1..10, '.' -> 11. Only 1..10 roll.
    """
    return (class_idx % 10) + 1 if 1 <= class_idx <= 10 else class_idx


class AugLoss(nn.Module):
    """Rolling-tolerant CTC (after Yang et al.).

    A mid-rotation wheel is labeled as the lower digit, but the model may legitimately
    see the upper neighbor. For each sample we also score the target with one of its
    last ``positions`` digits rolled up, and take the per-sample minimum loss — so the
    model is not punished for reading the in-between state either way.
    """

    def __init__(self, blank: int = 0, positions: int = 1, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction="none")
        self.positions = max(1, positions)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        best = self.ctc(log_probs, targets, input_lengths, target_lengths)  # (B,)
        seqs, off = [], 0
        for L in target_lengths.tolist():
            seqs.append(targets[off:off + L])
            off += L
        for k in range(1, self.positions + 1):
            alt, changed = [], False
            for s in seqs:
                s2 = s.clone()
                if len(s2) >= k:
                    j = len(s2) - k
                    rolled = _roll_up(int(s2[j]))
                    if rolled != int(s2[j]):
                        s2[j] = rolled
                        changed = True
                alt.append(s2)
            if not changed:
                continue
            loss_k = self.ctc(log_probs, torch.cat(alt), input_lengths, target_lengths)
            best = torch.minimum(best, loss_k)
        return best.mean()


class FocalCTCLoss(nn.Module):
    """CTC with focal weighting to concentrate on hard (confusable) examples."""

    def __init__(self, blank: int = 0, gamma: float = 2.0, zero_infinity: bool = True):
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction="none")
        self.gamma = gamma

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        per = self.ctc(log_probs, targets, input_lengths, target_lengths)
        p = torch.exp(-per)
        return (((1.0 - p) ** self.gamma) * per).mean()


class CenterLoss(nn.Module):
    """Pull per-timestep features toward learned class centers (argmax pseudo-labels)."""

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.centers)

    def forward(self, features, logits):
        labels = logits.argmax(2).reshape(-1)
        flat = features.reshape(-1, features.shape[-1])
        return ((flat - self.centers[labels]) ** 2).mean()


def build_criterion(cfg: Dict) -> nn.Module:
    """Main CTC criterion selected by config (standard / focal / AugLoss)."""
    if cfg.get("aug_loss"):
        return AugLoss(blank=0, positions=cfg.get("aug_loss_positions", 1))
    if cfg.get("focal_gamma", 0.0) > 0:
        return FocalCTCLoss(blank=0, gamma=cfg["focal_gamma"])
    return nn.CTCLoss(blank=0, zero_infinity=True)


def get_criterion_name(cfg: Dict) -> str:
    """Return a human-readable name for the criterion based on config."""
    if cfg.get("aug_loss"):
        return f"AugLoss(positions={cfg.get('aug_loss_positions', 1)})"
    if cfg.get("focal_gamma", 0.0) > 0:
        return f"FocalCTCLoss(gamma={cfg['focal_gamma']})"
    return "CTCLoss"
