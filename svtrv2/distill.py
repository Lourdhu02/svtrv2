"""ARD part 2 — inference-preserving SGM distillation (novel).

SVTRv2's Semantic Guidance Module gives the model linguistic context while
training, then is thrown away at test time -- the trained CTC head never
benefits from it.  This module closes that gap with **alignment-aware
distillation**: the SGM's two context streams become a soft teacher over the
charset, and the CTC head is trained to match that teacher *at the timesteps
each character actually occupies*.

The mapping from label positions to CTC timesteps comes in two flavours:

- ``uniform``: the same monotone spread the alignment warmup uses -- label i
  occupies timestep ``i * T / L`` (midpoint of its occupied range).  Cheap and
  stable, exact enough early in training.
- ``viterbi``: the single highest-probability CTC alignment of the *current*
  model (dynamic programming over the extended blank/label alphabet).  Each
  character is distilled at the timestep the model itself believes that
  character lives on, which sharpens as training progresses.

Both keep the student graph exactly the CTC-only inference graph: distillation
only changes the loss, never the architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


def uniform_positions(pos: torch.Tensor, target_lengths: torch.Tensor,
                      ) -> torch.Tensor:
    """Midpoint timestep of each label position from a monotone (B, T) map.

    ``pos`` assigns each timestep a label position (the uniform spread used by
    the alignment warmup).  For each label i we distill at the midpoint of the
    timestep run assigned to it; labels with no timestep fall back to the
    nearest proportional position, matching the warmup's fallback rule.

    Fully vectorized: run lengths come from a scatter-add bincount, run starts
    from a cumulative sum.
    """
    b, t = pos.shape
    device = pos.device
    l_max = int(target_lengths.max().item()) if b else 0
    if l_max == 0:
        return torch.zeros(b, 0, dtype=torch.long, device=device)

    counts = torch.zeros(b, l_max, dtype=torch.long, device=device)
    counts.scatter_add_(1, pos, torch.ones_like(pos))
    # start of label l's run = number of timesteps mapped to labels < l
    start = counts.cumsum(dim=1) - counts
    valid = counts > 0
    mid = start + (counts - 1) // 2

    lengths = target_lengths.clamp(min=1).view(b, 1)
    labels = torch.arange(l_max, device=device).view(1, l_max)
    fallback = (labels * max(t - 1, 0)) // (lengths - 1).clamp(min=1)
    out = torch.where(valid, mid, fallback)
    return out.clamp(0, t - 1).long()


def viterbi_positions(log_probs: torch.Tensor, padded_targets: torch.Tensor,
                      target_lengths: torch.Tensor,
                      input_lengths: torch.Tensor) -> torch.Tensor:
    """(B, L) timestep of each label position on the best CTC path.

    Single-best-path CTC dynamic programming over the extended alphabet
    ``[b, y1, b, y2, ..., b, yL, b]`` with the standard transition rule
    (stay / advance one / advance two when ``z_s != z_{s-2}``).  Each label
    position is mapped to the *first* timestep its label is emitted on the
    highest-probability alignment of the current model, so distillation lands
    exactly where the recognizer believes the character lives.

    Memory: (B, T, 2L+1) score and choice tensors -- for the paper-scale
    shapes here (B=256, T<=96, L<=32) that is a few tens of MB, acceptable
    for a train-only objective.
    """
    b, t, _ = log_probs.shape
    device = log_probs.device
    l_max = int(target_lengths.max().item()) if b else 0
    if l_max == 0:
        return torch.zeros(b, 0, dtype=torch.long, device=device)

    # Extended alphabet: blank at even states, y_{(s+1)/2} at odd states.
    s_max = 2 * l_max + 1
    labels = padded_targets.clamp(min=0)                        # (B, L), 0=pad
    z = torch.zeros(b, s_max, dtype=torch.long, device=device)
    z[:, 1::2] = labels
    emit = log_probs.gather(2, z.unsqueeze(1).expand(b, t, s_max))  # (B, T, S)

    # Advancing two states is illegal when z_s == z_{s-2} (same label twice
    # with no blank between).  Additive 0 / NEG_INF guard per state.
    same = z[:, 2:] == z[:, :-2]                                # (B, S-2)
    skip2 = torch.where(same, NEG_INF, 0.0)
    skip2 = F.pad(skip2, (2, 0), value=NEG_INF)                 # (B, S)

    # states reachable by a sample of label length l: s <= 2l
    reach = torch.arange(s_max, device=device).view(1, -1) \
        <= (2 * target_lengths.clamp(min=0)).view(b, 1)

    alpha = torch.full((b, s_max), NEG_INF, device=device, dtype=log_probs.dtype)
    alpha[:, 0] = log_probs[:, 0, 0]                            # blank at t=0
    alpha[:, 1] = log_probs[:, 0].gather(1, labels[:, :1]).squeeze(1)
    alpha = torch.where(reach, alpha, torch.full_like(alpha, NEG_INF))

    choices = torch.zeros(b, t, s_max, dtype=torch.int8, device=device)
    for tt in range(1, t):
        stay = alpha
        adv1 = torch.cat([torch.full((b, 1), NEG_INF, device=device,
                                     dtype=alpha.dtype), alpha[:, :-1]], dim=1)
        adv2 = torch.cat([torch.full((b, 2), NEG_INF, device=device,
                                     dtype=alpha.dtype),
                          alpha[:, :-2] + skip2[:, 2:]], dim=1)
        best, arg = torch.stack([stay, adv1, adv2], dim=0).max(dim=0)
        alpha = torch.where(reach, best, torch.full_like(best, NEG_INF)) + emit[:, tt]
        choices[:, tt] = arg.to(torch.int8)

    s = alpha.argmax(dim=1)                                     # (B,) best end state

    # Backtrace.  Occurrences are overwritten from the last timestep toward
    # the first, so each label ends up holding its FIRST emission timestep.
    out = torch.zeros(b, l_max, dtype=torch.long, device=device)
    for tt in range(t - 1, -1, -1):
        odd = s % 2 == 1
        rows = torch.nonzero(odd, as_tuple=False).squeeze(1)
        if rows.numel():
            out[rows, (s[rows] - 1) // 2] = tt
        step = choices[:, tt].gather(1, s.view(-1, 1)).squeeze(1).long()
        s = (s - step).clamp(min=0)
    return out


def align_positions(log_probs: torch.Tensor, padded_targets: torch.Tensor,
                    target_lengths: torch.Tensor, input_lengths: torch.Tensor,
                    mode: str = "uniform") -> torch.Tensor:
    """Dispatch label-position -> timestep mapping for distillation."""
    if mode == "viterbi":
        return viterbi_positions(log_probs, padded_targets, target_lengths,
                                 input_lengths)
    if mode != "uniform":
        raise ValueError(f"distill_align must be 'uniform' or 'viterbi', got {mode!r}")
    b, t, _ = log_probs.shape
    lengths = target_lengths.clamp(min=0).view(b, 1)
    steps = input_lengths.clamp(min=1, max=t).view(b, 1)
    ar = torch.arange(t, device=log_probs.device).view(1, t)
    # The warmup's spread: timestep t is responsible for label pos[t].
    pos = torch.minimum((ar * lengths) // steps, (lengths - 1).clamp(min=0))
    return uniform_positions(pos, target_lengths)


class AlignmentDistillLoss(nn.Module):
    """Distill the train-only SGM teacher into the CTC head (ARD).

    Teacher: the mean of the SGM's left/right stream distributions at each
    label position, softened by ``temperature``.  Student: the CTC head's
    distribution at the timestep that label position is aligned to (uniform
    spread or Viterbi best path).  Objective:

        L = (1 - ce_mix) * KL(teacher || student)  +  ce_mix * CE(GT || student)

    computed over valid (non-pad) label positions.  The KL term transfers the
    linguistic context the SGM sees; the small CE mix keeps the student
    anchored to the ground truth so the teacher's mistakes cannot dominate.
    Inference stays exactly CTC: this only reshapes the head's training signal.
    """

    def __init__(self, align: str = "uniform", temperature: float = 2.0,
                 ce_mix: float = 0.2, ignore_index: int = 0) -> None:
        super().__init__()
        if align not in ("uniform", "viterbi"):
            raise ValueError(f"align must be 'uniform' or 'viterbi', got {align!r}")
        self.align = align
        self.temperature = float(temperature)
        self.ce_mix = float(ce_mix)
        self.ignore_index = ignore_index

    def teacher_distribution(self, sgm_logits: torch.Tensor) -> torch.Tensor:
        """(B, 2, L, C) SGM logits -> (B, L, C) log teacher distribution."""
        tau = max(self.temperature, 1e-3)
        probs = 0.5 * (torch.softmax(sgm_logits[:, 0] / tau, dim=-1)
                       + torch.softmax(sgm_logits[:, 1] / tau, dim=-1))
        return probs.clamp_min(1e-8).log()

    def forward(self, student_log_probs: torch.Tensor, sgm_logits: torch.Tensor,
                padded_targets: torch.Tensor, target_lengths: torch.Tensor,
                input_lengths: torch.Tensor) -> torch.Tensor:
        b, t, c = student_log_probs.shape
        if padded_targets.numel() == 0 or b == 0:
            return student_log_probs.sum() * 0.0

        pos = align_positions(student_log_probs.detach(), padded_targets,
                              target_lengths, input_lengths, self.align)
        pos = pos.clamp(0, t - 1)                                   # (B, L)
        if pos.shape[1] == 0:
            return student_log_probs.sum() * 0.0

        teacher = self.teacher_distribution(sgm_logits.detach())    # (B, L, C)
        student = student_log_probs.gather(
            1, pos.unsqueeze(-1).expand(b, pos.shape[1], c))        # (B, L, C)

        # pos' width is max(target_lengths); a wider padded_targets row is
        # pure padding and is truncated so the mask lines up with pos.
        targets = padded_targets[:, :pos.shape[1]]
        valid = targets.ne(self.ignore_index)
        valid &= torch.arange(pos.shape[1], device=student.device).view(1, -1) \
            < target_lengths.view(b, 1)
        if not bool(valid.any()):
            return student_log_probs.sum() * 0.0

        kl = (teacher.exp() * (teacher - student)).sum(-1)          # (B, L)
        loss_kl = (kl * valid.to(kl.dtype)).sum() / valid.sum().clamp(min=1)

        if self.ce_mix > 0.0:
            one_hot = F.one_hot(targets.clamp(min=0), c).to(student.dtype)
            ce = -(one_hot * student).sum(-1)
            loss_ce = (ce * valid.to(ce.dtype)).sum() / valid.sum().clamp(min=1)
            return (1.0 - self.ce_mix) * loss_kl + self.ce_mix * loss_ce
        return loss_kl




