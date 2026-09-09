"""ARD novelty tests — adaptive routing, SGM distillation, engine integration.

CPU-only, synthetic, dataset-free where possible:

    python tests/test_novelty.py     # standalone runner
    python -m pytest tests/test_novelty.py
"""
from __future__ import annotations

import atexit
import itertools
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svtrv2.config import DEFAULTS, MODELS, MSR_BINS, NUM_CLASSES  # noqa: E402
from svtrv2.distill import (AlignmentDistillLoss, align_positions,  # noqa: E402
                            uniform_positions, viterbi_positions)
from svtrv2.dataset import TextDataset, collate_fn  # noqa: E402
from svtrv2.engine import fit, load_checkpoint, predict_dir  # noqa: E402
from svtrv2.model import LocalMixing, SVTRNet  # noqa: E402
from svtrv2.routing import (MSRRouter, RouterLoss, geometric_prior,  # noqa: E402
                            route_bins)
from svtrv2.text import CTCCodec  # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="svtrv2-novelty-"))
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))


def _render(text: str, dw: int = 40, dh: int = 72) -> np.ndarray:
    img = np.full((dh + 20, len(text) * dw + 20, 3), 200, np.uint8)
    for i in range(len(text)):
        x = 10 + i * dw
        cv2.rectangle(img, (x, 10), (x + dw - 10, dh + 10 - 10), (40, 40, 40), -1)
    return img


def _make_dataset(root: Path, n: int = 16) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        label = f"{i % 10}{(i * 7) % 10}{(i * 3) % 10}{i % 5}"
        img = _render(label, dw=30 + (i % 4) * 8, dh=60 + (i % 3) * 12)
        fname = f"G{i % 4:03d}-{i:04d}.png"
        cv2.imwrite(str(root / "images" / fname), img)
        lines.append(f"{fname}\t{label}")
    (root / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------- geometry


def test_local_mixing_is_two_3x3_grouped_convs():
    """Official LNConvTwo33: two grouped 3x3 convs, D/32 groups, no norm/act between."""
    import torch.nn as nn

    m = LocalMixing(256)
    assert len(m.conv) == 2
    for conv in m.conv:
        assert conv.kernel_size == (3, 3)
        assert conv.groups == 256 // 32
        assert not any(isinstance(x, (nn.BatchNorm2d, nn.GELU, nn.LayerNorm,
                                      nn.ReLU)) for x in m.conv), \
            "no normalization or activation may sit between the two convs"


def test_backbone_timesteps_are_width_over_four():
    """Official sub_k: no stage-2 downsample -> final map H/8 x W/4."""
    net = SVTRNet(num_classes=NUM_CLASSES, **MODELS["svtrv2-t"]).eval()
    for b in MSR_BINS:
        x = torch.randn(2, 3, b["height"], b["width"])
        with torch.no_grad():
            feats = net.forward_feature_map(x)
        assert feats.shape[2] == b["feature_h"], b["name"]
        assert feats.shape[3] * 4 == b["width"], b["name"]
        seq = net.forward_features(x)
        assert seq.shape[1] == b["timesteps"], b["name"]


# --------------------------------------------------------------------- router


def test_router_shapes_and_hard_route():
    torch.manual_seed(0)
    router = MSRRouter().eval()
    probe = torch.randn(4, 3, 32, 128)
    ar = torch.tensor([0.8, 1.2, 2.0, 4.0])
    with torch.no_grad():
        logits = router(probe, ar)
    assert logits.shape == (4, len(MSR_BINS))
    hard = route_bins(logits, hard=True)
    assert torch.allclose(hard.sum(-1), torch.ones(4))
    assert ((hard == 0) | (hard == 1)).all()
    assert hard.argmax(-1).tolist() == logits.argmax(-1).tolist()


def test_hard_route_is_straight_through():
    logits = torch.randn(3, len(MSR_BINS), requires_grad=True)
    weights = torch.arange(len(MSR_BINS), dtype=torch.float32)
    assignment = route_bins(logits, hard=True)
    (assignment @ weights).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_geometric_prior_matches_the_static_rule():
    ar = torch.tensor([0.5, 1.6, 2.6, 5.0])
    prior = geometric_prior(ar)
    expected = [0, 1, 2, 3]
    assert prior.argmax(-1).tolist() == expected
    assert torch.allclose(prior.sum(-1), torch.ones(4))


def test_router_preference_learns_the_better_bin():
    """The pairwise preference loss moves probability toward the lower-CTC bin."""
    torch.manual_seed(0)
    router = MSRRouter()
    opt = torch.optim.Adam(router.parameters(), lr=3e-2)
    loss_fn = RouterLoss(weight=1.0, entropy=0.0)
    probe = torch.randn(8, 3, 32, 128)
    ar = torch.full((8,), 2.0)
    bin_a = torch.zeros(8, dtype=torch.long)      # candidate: bin 0
    bin_b = torch.ones(8, dtype=torch.long)       # candidate: bin 1
    loss_a = torch.full((8,), 2.0)                # bin 0 reads worse
    loss_b = torch.full((8,), 0.5)                # bin 1 reads better
    logits0 = router(probe, ar).detach()
    before = torch.softmax(logits0, -1)[:, 1].mean()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        logits = router(probe, ar)
        loss, _ = loss_fn(logits, ar, loss_fn.preference(logits, bin_a, bin_b,
                                                         loss_a, loss_b))
        loss.backward()
        opt.step()
    with torch.no_grad():
        after = torch.softmax(router(probe, ar), -1)[:, 1].mean()
    assert after > before + 0.1, (before, after)


# ------------------------------------------------------- uniform + viterbi align


def test_uniform_positions_picks_run_midpoints():
    """T=8, L=4 -> pos [0,0,1,1,2,2,3,3]; midpoints 0,2,4,6."""
    pos = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3]], dtype=torch.long)
    lengths = torch.tensor([[4]])
    out = uniform_positions(pos, lengths)
    assert out.tolist() == [[0, 2, 4, 6]]


def test_uniform_positions_falls_back_when_a_label_has_no_run():
    pos = torch.tensor([[0, 0, 0, 3, 3, 3, 3, 3]], dtype=torch.long)  # labels 1,2 skipped
    lengths = torch.tensor([[4]])
    out = uniform_positions(pos, lengths)
    assert out[0, 0] == 1  # midpoint of the run for label 0
    assert out[0, 1] >= 0 and out[0, 2] >= 0  # proportional fallback, in range
    assert out[0, 3] == 5  # midpoint of label 3's run


def test_viterbi_matches_an_obvious_alignment():
    """One-hot-ish log probs: b A b B b -> A at t=1, B at t=3."""
    t, c = 5, 4
    log_probs = torch.full((1, t, c), -8.0).log_softmax(-1)
    for tt, sym in [(0, 0), (1, 1), (2, 0), (3, 2), (4, 0)]:
        log_probs[0, tt, sym] = -0.01
    padded = torch.tensor([[1, 2]])
    lengths = torch.tensor([[2]])
    in_len = torch.tensor([[t]])
    out = viterbi_positions(log_probs, padded, lengths, in_len)
    assert out.tolist() == [[1, 3]]


def test_viterbi_repeated_labels_need_blank_separated_emissions():
    """Labels [A, A]: the best path must be b A b A b, not A A ... ."""
    t, c = 5, 3
    log_probs = torch.full((1, t, c), -8.0).log_softmax(-1)
    for tt, sym in [(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)]:
        log_probs[0, tt, sym] = -0.01
    padded = torch.tensor([[1, 1]])
    out = viterbi_positions(log_probs, padded, torch.tensor([[2]]),
                            torch.tensor([[t]]))
    assert out.tolist() == [[1, 3]]


def test_viterbi_equals_brute_force_on_random_inputs():
    """Exhaustive enumeration of legal CTC state paths, tiny shapes."""
    t, c, l_len = 5, 4, 2
    s_max = 2 * l_len + 1
    for trial in range(20):
        log_probs = torch.log_softmax(torch.randn(1, t, c), -1)
        y = torch.tensor([[1, 3]])
        z = torch.zeros(s_max, dtype=torch.long)
        z[1::2] = y[0]
        best_key = None
        best_first = None
        # Enumerate every legal state path: start at state 0 or 1, moves are
        # stay / +1 / +2, and +2 is illegal when z_s == z_{s-2}.
        for moves in itertools.product([0, 1, 2], repeat=t - 1):
            for start in (0, 1):
                path = [start]
                legal = True
                for tt, mv in enumerate(moves, 1):
                    s = path[-1] + mv
                    if s >= s_max or (mv == 2 and z[s] == z[s - 2]):
                        legal = False
                        break
                    path.append(s)
                if not legal:
                    continue
                # Tie-break deterministically on the path tuple so the brute
                # force picks exactly one alignment when scores tie.
                key = (sum(log_probs[0, tt, z[path[tt]]].item() for tt in range(t)),
                       tuple(-p for p in path))
                if best_key is None or key > best_key:
                    best_key = key
                    first = {}
                    for tt, s in enumerate(path):
                        if s % 2 == 1 and (s - 1) // 2 not in first:
                            first[(s - 1) // 2] = tt
                    best_first = [first.get(i, 0) for i in range(l_len)]
        out = viterbi_positions(log_probs, y, torch.tensor([[l_len]]),
                                torch.tensor([[t]]))
        assert out[0].tolist() == best_first, (trial, out[0].tolist(), best_first)


def test_distill_loss_is_finite_and_masked():
    torch.manual_seed(0)
    b, t, c, l = 3, 12, NUM_CLASSES, 6
    student = torch.log_softmax(torch.randn(b, t, c), -1)
    sgm_logits = torch.randn(b, 2, l, c)
    padded = torch.zeros(b, l, dtype=torch.long)
    padded[0, :4] = torch.tensor([1, 2, 3, 4])
    padded[1, :6] = torch.tensor([5, 6, 7, 8, 9, 10])
    lengths = torch.tensor([[4], [6], [0]])
    in_len = torch.full((b, 1), t)

    for align in ("uniform", "viterbi"):
        loss_fn = AlignmentDistillLoss(align=align, ce_mix=0.2)
        val = loss_fn(student, sgm_logits, padded, lengths, in_len)
        assert torch.isfinite(val), align
        assert val.item() >= 0.0, align

    # An all-pad batch has nothing to distill -> exact zero.
    empty = torch.zeros(b, 1, dtype=torch.long)
    zero_lengths = torch.zeros(b, 1, dtype=torch.long)
    val = AlignmentDistillLoss()(student, sgm_logits, empty, zero_lengths, in_len)
    assert val.item() == 0.0


def test_align_positions_dispatch_validates_mode():
    log_probs = torch.randn(1, 8, NUM_CLASSES)
    try:
        align_positions(log_probs, torch.tensor([[1, 2]]), torch.tensor([[2]]),
                        torch.tensor([[8]]), mode="bogus")
    except ValueError:
        return
    raise AssertionError("bogus align mode must raise")


# --------------------------------------------------------- dataset + engine


def test_router_fields_flow_through_dataset_and_collate():
    from svtrv2.transforms import build_msr_transforms

    root = _make_dataset(_TMP / "ds_router", n=8)
    tf = build_msr_transforms(True, "none", "edge")
    ds = TextDataset(root / "images", _samples_of(root), tf, CTCCodec(),
                     emit_router_fields=True)
    item = ds[0]
    assert len(item) == 8
    tensor, target, length, label, bin_name, probe, ar, orig = item
    assert probe.shape == (3, 32, 128)
    assert isinstance(ar, float) and ar > 0.0
    assert isinstance(orig, np.ndarray) and orig.ndim == 3

    batch = collate_fn([ds[0], ds[1]])
    assert len(batch) == 9
    images, flat, lengths, labels, padded, bin_name, probes, ars, origs = batch
    assert probes.shape == (2, 3, 32, 128)
    assert ars.shape == (2,) and ars.dtype == torch.float32
    assert len(origs) == 2


def _samples_of(root: Path):
    from svtrv2.dataset import parse_manifest

    return parse_manifest(str(root), verbose=False)


def _fit_cfg(tmp_name: str) -> dict:
    return dict(epochs=1, batch=4, lr=3e-4, workers=0, warmup_epochs=1, patience=5,
                amp=False, device="cpu", compile=False,
                split=(0.75, 0.125, 0.125), seed=0, group_split=True,
                aug_level="none", pad_mode="edge", img_h=32, img_w=128,
                resize_mode="pad", blank_bias=-2.0, ema_decay=0.9, eval_ema=True,
                ctc_weight=1.0, align_weight=0.5, align_warmup_epochs=2,
                sgm_weight=1.0, sgm_start_epoch=1, sgm_warmup_epochs=1,
                grad_clip=5.0, min_lr=1e-6, weight_decay=1e-4,
                # ARD on: routing + viterbi distillation, explore every step
                route=True, router_explore_every=1, router_eval_bs=4,
                router_weight=0.5, router_entropy=0.01, router_lr=1e-3,
                distill=True, distill_weight=0.5, distill_align="viterbi",
                distill_temperature=2.0, distill_ce_mix=0.2, distill_start_epoch=1)


def test_fit_trains_with_routing_and_distillation_end_to_end():
    """One CPU epoch with both ARD features on: losses finite, ckpt carries router."""
    root = _make_dataset(_TMP / "ds_ard", n=12)
    cfg = _fit_cfg("ard")
    run = _TMP / "runs_ard" / "ard"
    best = fit("svtrv2-t", str(root), cfg, project=str(_TMP / "runs_ard"), name="ard")
    assert Path(best).exists()

    _net, ckpt = load_checkpoint(best, torch.device("cpu"))
    assert ckpt.get("router_state"), "checkpoint must carry the trained router"
    # last.pth carries the epoch history (best.pth stores EMA weights only).
    _, last_ckpt = load_checkpoint(str(run / "last.pth"), torch.device("cpu"))
    hist = last_ckpt.get("history") or []
    assert hist and all(np.isfinite(hist[-1][f"train_{k}"])
                        for k in ("total", "ctc", "align", "sgm", "distill", "router"))


def test_predict_dir_uses_the_checkpoint_router():
    from svtrv2.routing import load_router

    ckpt_path = _TMP / "runs_ard" / "ard" / "best.pth"
    net, ckpt = load_checkpoint(str(ckpt_path), torch.device("cpu"))
    router = load_router(ckpt, torch.device("cpu"))
    assert router is not None
    root = _TMP / "ds_ard"
    results = predict_dir(net, root / "images", CTCCodec(), torch.device("cpu"),
                          batch=4, router=router)
    assert len(results) == 12
    assert all(isinstance(t, str) and 0.0 <= c <= 1.0 for _, t, c in results)


def test_defaults_keep_ard_off():
    """The paper baseline must be untouched when flags are absent."""
    assert DEFAULTS["route"] is False
    assert DEFAULTS["distill"] is False
    assert DEFAULTS["scheduler"] == "cosine"
    assert DEFAULTS["filter_wd"] is False
