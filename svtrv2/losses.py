import torch
import torch.nn as nn
import torch.nn.functional as F


class SGMLoss(nn.Module):
    """Cross-entropy for the two SVTRv2 semantic guidance streams."""

    def __init__(self, ignore_index: int = 0) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:
            logits: (B, 2, L, C) from SemanticGuidanceModule.
            targets: (B, L), padded with ignore_index.
        """
        if targets.numel() == 0:
            return logits.sum() * 0.0
        expanded = targets.unsqueeze(1).expand(-1, logits.size(1), -1)
        mask = expanded.ne(self.ignore_index)
        if not bool(mask.any()):
            return logits.sum() * 0.0
        num_classes = logits.size(-1)
        return F.cross_entropy(
            logits.reshape(-1, num_classes)[mask.reshape(-1)],
            expanded.reshape(-1)[mask.reshape(-1)],
        )


class UniformAlignmentLoss(nn.Module):
    """Early CTC anti-collapse loss with simple left-to-right target expansion.

    During warmup, this gives the visual classifier a direct non-blank signal at
    every valid timestep. CTC remains the main objective and handles final
    alignment.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        log_probs: torch.Tensor,
        padded_targets: torch.Tensor,
        target_lengths: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        b, t, c = log_probs.shape
        device = log_probs.device
        # Fully vectorized: the previous per-sample Python loop called .item()
        # twice per sample, forcing 2*batch GPU->CPU syncs every single step.
        lengths = target_lengths.to(device).view(b, 1)
        steps = input_lengths.to(device).view(b, 1).clamp(min=0, max=t)
        ar = torch.arange(t, device=device).view(1, t)

        # positions[i, j] = j * length_i // steps_i, clamped to the last target
        pos = (ar * lengths) // steps.clamp(min=1)
        pos = torch.minimum(pos, (lengths - 1).clamp(min=0))
        pos = pos.clamp(0, padded_targets.size(1) - 1)

        valid = (ar < steps) & (lengths > 0)
        aligned = torch.where(
            valid,
            padded_targets.to(device).gather(1, pos),
            torch.full_like(pos, self.ignore_index),
        )
        return F.nll_loss(
            log_probs.reshape(-1, c),
            aligned.reshape(-1),
            ignore_index=self.ignore_index,
        )
