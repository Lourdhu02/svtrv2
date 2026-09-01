"""Confidence calibration and abstention.

The weakest emitted digit's probability is the sequence confidence (one shaky digit
makes an exact-match read uncertain). Temperature scaling calibrates it; a threshold
chosen for a target precision routes low-confidence reads to humans.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch

from .text import CTCCodec


def _norm_logprobs(log_probs: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = log_probs / T
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return np.log(np.clip(p, 1e-12, 1.0))


def decode_conf(log_probs, codec: CTCCodec, T: float = 1.0) -> Tuple[str, float, List[float]]:
    """Greedy-decode (T, C) log-probs with calibrated per-digit confidence."""
    lp = log_probs.detach().cpu().numpy() if isinstance(log_probs, torch.Tensor) else np.asarray(log_probs)
    return codec.decode_with_conf(_norm_logprobs(lp, T))


def expected_calibration_error(confs: Sequence[float], correct: Sequence[int], bins: int = 10) -> float:
    confs, correct = np.asarray(confs), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confs[m].mean())
    return float(ece)


class TemperatureScaler:
    """Fit a single temperature T to minimize ECE of the min-digit confidence."""

    def __init__(self, T: float = 1.0):
        self.T = float(T)

    def fit(self, logprob_list: Sequence[np.ndarray], labels: Sequence[str],
            codec: CTCCodec, grid: Sequence[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)):
        best_T, best_ece = 1.0, float("inf")
        for T in grid:
            confs, correct = [], []
            for lp, gt in zip(logprob_list, labels):
                text, mn, _ = codec.decode_with_conf(_norm_logprobs(np.asarray(lp), T))
                confs.append(mn)
                correct.append(int(text == gt))
            ece = expected_calibration_error(confs, correct)
            if ece < best_ece:
                best_ece, best_T = ece, T
        self.T = best_T
        return self


def choose_threshold(confs: Sequence[float], correct: Sequence[int],
                     target_precision: float = 0.995) -> float:
    """Lowest confidence threshold whose accepted reads hit the target precision.

    Returns tau; reads with confidence < tau should be abstained (sent to review).
    """
    order = np.argsort(confs)[::-1]
    confs, correct = np.asarray(confs)[order], np.asarray(correct)[order]
    acc_correct = np.cumsum(correct)
    precision = acc_correct / np.arange(1, len(confs) + 1)
    ok = np.where(precision >= target_precision)[0]
    if len(ok) == 0:
        return 1.0  # cannot meet target -> abstain on everything
    return float(confs[ok[-1]])


@torch.no_grad()
def tta_logits(net, tensors: torch.Tensor, domain_ids=None) -> torch.Tensor:
    """Average probabilities across TTA views, return as log-probs for decoding."""
    log_probs = net(tensors, domain_ids).float()
    return log_probs.exp().mean(0, keepdim=True).clamp_min(1e-12).log()


def compute_precision_recall_at_thresholds(confs: Sequence[float], correct: Sequence[int],
                                            thresholds: Sequence[float] = None) -> List[Dict]:
    """Compute precision, recall, and coverage at multiple confidence thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 21)
    confs, correct = np.asarray(confs), np.asarray(correct)
    results = []
    for tau in thresholds:
        mask = confs >= tau
        if mask.any():
            precision = correct[mask].mean()
            recall = correct[mask].sum() / max(correct.sum(), 1)
            coverage = mask.mean()
        else:
            precision, recall, coverage = 0.0, 0.0, 0.0
        results.append({
            "threshold": float(tau),
            "precision": float(precision),
            "recall": float(recall),
            "coverage": float(coverage),
        })
    return results
