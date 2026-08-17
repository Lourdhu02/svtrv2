import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCTCLoss(nn.Module):
    """CTC loss with focal weighting to focus on hard examples.

    Down-weights easy (low-loss) samples so the model concentrates on
    confusable digits like 6/8 and 1/7.  gamma=0 recovers standard CTC.
    """

    def __init__(self, blank: int = 0, gamma: float = 2.0, zero_infinity: bool = True) -> None:
        super().__init__()
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')
        self.gamma = gamma

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # reduction='none' returns the loss SUMMED over each target sequence.
        # Exponentiating that directly gives p ~ 1e-5 for any real label, so
        # (1 - p)^gamma collapses to 1 and the focal term silently does
        # nothing.  Normalising per character first makes p a genuine mean
        # per-character likelihood, which is what focal weighting expects.
        per_sample = self.ctc(log_probs, targets, input_lengths, target_lengths)
        per_char = per_sample / target_lengths.clamp(min=1).to(per_sample.dtype)
        p = torch.exp(-per_char)
        focal = ((1.0 - p) ** self.gamma) * per_char
        return focal.mean()


class CenterLoss(nn.Module):
    """Center loss that pulls per-timestep features toward learned class centers.

    Uses CTC argmax predictions as pseudo-labels.  Encourages the backbone to
    produce tightly clustered embeddings for each digit class.
    """

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.kaiming_normal_(self.centers)

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Compute center loss using argmax pseudo-labels.

        Args:
            features: backbone features (B, T, D)
            logits: model output log-probs (B, T, C) — used for argmax labels
        """
        labels = logits.argmax(2)
        B, T, D = features.shape
        flat_feat = features.reshape(-1, D)
        flat_labels = labels.reshape(-1)
        centers_batch = self.centers[flat_labels]
        # Mean over feature dim keeps the loss O(1) regardless of D.
        return ((flat_feat - centers_batch) ** 2).mean()


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

    Meter labels are read left-to-right from tightly cropped windows. During
    warmup, this gives the visual classifier a direct non-blank signal at every
    valid timestep. CTC remains the main objective and handles final alignment.
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
