"""Portable CPU test suite — synthetic data, no dataset dependency.

    python tests/test_engine.py     # standalone runner
    python -m pytest tests          # or pytest
"""
from __future__ import annotations

import atexit
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from svtrv2.config import (MODELS, MSR_BINS, NUM_CLASSES, VARIANTS,  # noqa: E402
                           resolve_model_name)
from svtrv2.dataset import (MSRBatchSampler, TextDataset, build_loaders,  # noqa: E402
                            collate_fn, group_split, parse_manifest, save_splits)
from svtrv2.engine import (compute_metrics, fit, load_checkpoint,  # noqa: E402
                           loss_weights, predict_dir, predict_image)
from svtrv2.export import export_onnx  # noqa: E402
from svtrv2.losses import SGMLoss, UniformAlignmentLoss  # noqa: E402
from svtrv2.model import SVTRNet  # noqa: E402
from svtrv2.text import CTCCodec, is_valid_label  # noqa: E402
from svtrv2.transforms import (backlight_bloom, build_msr_transforms,  # noqa: E402
                               fit_pad, glare, moire, segment_fade,
                               select_msr_bin, viewing_angle_washout)

_TMP = Path(tempfile.mkdtemp(prefix="svtrv2-test-"))
atexit.register(lambda: shutil.rmtree(_TMP, ignore_errors=True))

_SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
             "5": "afgcd", "6": "afgcde", "7": "abc", "8": "abcdefg", "9": "abcfgd"}


def _render(text: str, dw: int = 40, dh: int = 72, margin: int = 10,
            emissive: bool = False) -> np.ndarray:
    """Draw a seven-segment display sample, dark-on-light or light-on-dark."""
    w, h = len(text) * dw + 2 * margin, dh + 2 * margin
    bg, on, off = (((28, 30, 34), (235, 240, 245), (52, 55, 60)) if emissive
                   else ((196, 196, 192), (35, 38, 42), (176, 178, 180)))
    img = np.full((h, w, 3), bg, np.uint8)
    for i, ch in enumerate(text):
        x0, y0 = margin + i * dw, margin
        t, ww = max(3, dw // 7), dw - dw // 4
        segs = {"a": (x0 + t, y0, ww - 2 * t, t), "b": (x0 + ww - t, y0 + t, t, dh // 2 - t),
                "c": (x0 + ww - t, y0 + dh // 2, t, dh // 2 - t),
                "d": (x0 + t, y0 + dh - t, ww - 2 * t, t),
                "e": (x0, y0 + dh // 2, t, dh // 2 - t), "f": (x0, y0 + t, t, dh // 2 - t),
                "g": (x0 + t, y0 + dh // 2 - t // 2, ww - 2 * t, t)}
        lit = _SEGMENTS.get(ch, "")
        for k, (x, y, bw, bh) in segs.items():
            cv2.rectangle(img, (x, y), (x + bw, y + bh), on if k in lit else off, -1)
    return img


def _make_dataset(root: Path, n: int = 48) -> Path:
    """A tiny dataset with several samples per group_id, so group-split has work."""
    rng = random.Random(0)
    (root / "images").mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.choice([5, 6])))
        label = digits[:-1] + "." + digits[-1] if rng.random() < 0.4 else digits
        img = _render(digits, dw=rng.randint(30, 48), dh=rng.randint(60, 84),
                      emissive=rng.random() < 0.4)
        fname = f"WB{i % 8:03d}-{i:04d}.png"
        cv2.imwrite(str(root / "images" / fname), img)
        lines.append(f"{fname}\t{label}")
    (root / "labels.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# -------------------------------------------------------------------- config


def test_variants_follow_the_paper():
    assert VARIANTS == ("svtrv2-t", "svtrv2-s", "svtrv2-b", "svtrv2-xl")
    for name, spec in MODELS.items():
        dims, depths, mixers = spec["dims"], spec["depths"], spec["mixers"]
        assert len(dims) == len(depths) == len(mixers) == 3, name
        # One mixer entry per block, and every block is local or global -- never both.
        for stage_mixers, depth in zip(mixers, depths):
            assert len(stage_mixers) == depth, name
            assert set(stage_mixers) <= {"Local", "Global"}, name
        # heads = groups = D/32 requires every dim to be a multiple of 32.
        assert all(d % 32 == 0 for d in dims), name
        # Local blocks come first, global last: the [L]_m[G]_n permutation.
        flat = [m for stage in mixers for m in stage]
        assert flat == sorted(flat, key=lambda m: 0 if m == "Local" else 1), name


def test_aliases_resolve():
    assert resolve_model_name("t") == "svtrv2-t"
    assert resolve_model_name("s") == "svtrv2-s"
    assert resolve_model_name("b") == "svtrv2-b"
    assert resolve_model_name("XL") == "svtrv2-xl"
    assert resolve_model_name("svtrv2-s") == "svtrv2-s"


def test_msr_bins_are_contiguous_and_consistent():
    assert MSR_BINS[0]["min_ar"] == 0.0
    assert MSR_BINS[-1]["max_ar"] == float("inf")
    for a, b in zip(MSR_BINS, MSR_BINS[1:]):
        assert a["max_ar"] == b["min_ar"], "bin edges must not leave a gap"
    for b in MSR_BINS:
        # The backbone strides width by 8; timesteps must agree or the CTC input
        # lengths silently disagree with the real sequence length.
        assert b["timesteps"] == b["width"] // 8, b["name"]


# ---------------------------------------------------------------------- text


def test_ctc_decode_keeps_blank_separated_repeats():
    codec = CTCCodec()
    one = codec.c2i["1"]
    assert codec.decode([one, 0, one]) == "11"
    assert codec.decode([one, one]) == "1"
    assert codec.decode([0, 0]) == ""


def test_codec_roundtrip_and_validation():
    codec = CTCCodec()
    assert codec.encode("01.5") == [codec.c2i[c] for c in "01.5"]
    assert is_valid_label("01234.5")
    assert not is_valid_label("12a4")
    assert not is_valid_label("")


def test_decode_with_conf_reports_the_weakest_digit():
    codec = CTCCodec()
    t, c = 4, NUM_CLASSES
    logits = np.full((t, c), -20.0, dtype=np.float32)
    logits[0, codec.c2i["7"]] = np.log(0.99)
    logits[1, 0] = np.log(0.99)
    logits[2, codec.c2i["3"]] = np.log(0.40)      # the weak one
    logits[3, 0] = np.log(0.99)
    text, conf, per = codec.decode_with_conf(logits)
    assert text == "73"
    assert conf == min(per)
    assert abs(conf - 0.40) < 1e-4


# ---------------------------------------------------------------- transforms


def test_fit_pad_shapes_and_pad_modes():
    img = _render("012345")
    for mode in ("edge", "reflect", "black"):
        assert fit_pad(img, 32, 128, pad_mode=mode).shape == (32, 128, 3), mode
    # Black padding really is black in the padded band.
    tall = _render("01", dw=40, dh=200)
    assert fit_pad(tall, 32, 128, pad_mode="black")[:, 0].max() == 0


def test_select_msr_bin_covers_every_aspect_ratio():
    assert select_msr_bin(100, 100)["name"] == "short"     # AR 1.0
    assert select_msr_bin(64, 159)["name"] == "medium"     # AR 2.48
    assert select_msr_bin(64, 220)["name"] == "long"       # AR 3.4
    assert select_msr_bin(50, 500)["name"] == "xlong"      # AR 10.0


def test_display_augmentations_are_shape_and_dtype_preserving():
    random.seed(1)
    img = _render("048726")
    for fn in (glare, segment_fade, backlight_bloom, viewing_angle_washout, moire):
        out = fn(img.copy())
        assert out.shape == img.shape, fn.__name__
        assert out.dtype == np.uint8, fn.__name__
        assert not np.array_equal(out, img), f"{fn.__name__} was a no-op"


def test_bloom_respects_display_polarity():
    """Dark-on-light strokes must spread darker, emissive strokes brighter.

    A polarity-blind implementation is a near no-op on one of the two, which is
    exactly the bug this guards against.
    """
    random.seed(3)
    dark_on_light = _render("048726", emissive=False)
    emissive = _render("048726", emissive=True)
    assert backlight_bloom(dark_on_light.copy()).mean() < dark_on_light.mean()
    assert backlight_bloom(emissive.copy()).mean() > emissive.mean()


def test_segment_fade_never_erases_a_segment():
    """Fading must attenuate, not delete: deleting a segment turns an 8 into a 9
    while the label still says 8, which trains the model to hallucinate."""
    random.seed(0)
    img = _render("888888")
    for _ in range(20):
        assert segment_fade(img.copy()).std() > 0.35 * img.std()


def test_eval_transforms_do_not_augment():
    img = _render("012345")
    tfs = build_msr_transforms(training=False, aug_level="none")
    a = tfs["medium"](image=img)["image"]
    b = tfs["medium"](image=img)["image"]
    assert torch.equal(a, b), "evaluation preprocessing must be deterministic"
    assert a.shape == (3, 32, 192)


# --------------------------------------------------------------------- model


def test_every_variant_builds_and_accepts_all_msr_canvases():
    for name, spec in MODELS.items():
        net = SVTRNet(num_classes=NUM_CLASSES, **spec).eval()
        for b in MSR_BINS:
            with torch.no_grad():
                out = net(torch.randn(1, 3, int(b["height"]), int(b["width"])))
            assert out.shape == (1, b["timesteps"], NUM_CLASSES), (name, b["name"])


def test_forward_is_log_softmax():
    net = SVTRNet(num_classes=NUM_CLASSES, **MODELS["svtrv2-s"]).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 3, 64, 192))
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 24), atol=1e-4)


def test_blank_bias_biases_the_blank_class():
    net = SVTRNet(num_classes=NUM_CLASSES, **MODELS["svtrv2-s"], blank_bias=-2.0)
    assert abs(float(net.head.bias[0]) + 2.0) < 1e-6


def test_sgm_is_off_the_inference_path():
    """The SGM is train-only: zeroing every one of its parameters must not move
    the recognition head's output, which is what lets a deployed model drop it."""
    torch.manual_seed(0)
    net = SVTRNet(num_classes=NUM_CLASSES, **MODELS["svtrv2-s"]).eval()
    x = torch.randn(1, 3, 64, 192)
    with torch.no_grad():
        before = net(x).clone()
        for n, p in net.named_parameters():
            if n.startswith("sgm."):
                p.zero_()
        after = net(x)
    assert torch.equal(before, after)


# -------------------------------------------------------------------- losses


def test_loss_schedule_matches_the_two_phase_recipe():
    cfg = dict(ctc_weight=1.0, align_weight=0.5, align_warmup_epochs=40,
               sgm_weight=1.0, sgm_start_epoch=5, sgm_warmup_epochs=10)
    assert loss_weights(1, cfg) == (1.0, 0.5, 0.0), "SGM must be off in phase 1"
    assert loss_weights(4, cfg)[2] == 0.0
    assert 0 < loss_weights(5, cfg)[2] < 1.0, "SGM ramps, it does not switch on"
    assert loss_weights(14, cfg)[2] == 1.0
    aligns = [loss_weights(e, cfg)[1] for e in range(1, 45)]
    assert all(a >= b for a, b in zip(aligns, aligns[1:])), "align must decay monotonically"
    assert aligns[-1] == 0.0


def test_alignment_loss_is_finite_and_gradable():
    torch.manual_seed(0)
    b, t, c, ell = 3, 24, NUM_CLASSES, 6
    raw = torch.randn(b, t, c, requires_grad=True)
    loss = UniformAlignmentLoss()(raw.log_softmax(-1), torch.randint(1, c, (b, ell)),
                                  torch.full((b,), ell), torch.full((b,), t))
    assert torch.isfinite(loss)
    loss.backward()
    assert raw.grad is not None


def test_sgm_loss_ignores_padding_and_survives_empty_targets():
    torch.manual_seed(0)
    b, ell, c = 2, 5, NUM_CLASSES
    logits = torch.randn(b, 2, ell, c)
    targets = torch.zeros(b, ell, dtype=torch.long)
    # All padding -> no supervised position -> zero, not NaN.
    assert float(SGMLoss()(logits, targets)) == 0.0
    targets[:, :3] = torch.randint(1, c, (b, 3))
    assert torch.isfinite(SGMLoss()(logits, targets))


# ------------------------------------------------------------------- dataset


def test_manifest_and_group_split_prevent_group_leakage():
    root = _make_dataset(_TMP / "ds_split")
    samples = parse_manifest(str(root))
    assert len(samples) == 48
    splits = group_split(samples, (0.7, 0.15, 0.15), seed=0, by_group=True)
    seen: dict = {}
    for name, items in splits.items():
        for s in items:
            assert seen.setdefault(s.group_id, name) == name, \
                f"group {s.group_id} spans two splits"
    assert sum(len(v) for v in splits.values()) == len(samples)


def test_msr_batch_sampler_never_mixes_bins():
    root = _make_dataset(_TMP / "ds_sampler")
    ds = TextDataset(root / "images", parse_manifest(str(root)),
                      build_msr_transforms(training=False, aug_level="none"))
    for batch in MSRBatchSampler(ds, batch_size=4, shuffle=True, drop_last=False):
        bins = {ds.sample_bins[i]["name"] for i in batch}
        assert len(bins) == 1, f"batch spans bins {bins}"


def test_collate_pads_targets_for_sgm():
    root = _make_dataset(_TMP / "ds_collate")
    ds = TextDataset(root / "images", parse_manifest(str(root)),
                      build_msr_transforms(training=False, aug_level="none"))
    idx = next(iter(MSRBatchSampler(ds, 4, shuffle=False, drop_last=False)))
    images, flat, lengths, labels, padded, bin_name = collate_fn([ds[i] for i in idx])
    assert images.shape[0] == len(idx)
    assert int(lengths.sum()) == flat.numel(), "flat targets must match the lengths"
    assert padded.shape == (len(idx), int(lengths.max()))
    for row, n in zip(padded, lengths):
        assert (row[int(n):] == 0).all(), "padding must be the pad id 0"
    assert bin_name in {b["name"] for b in MSR_BINS}
    assert len(labels) == len(idx)


def test_build_loaders_reuses_persisted_splits():
    root = _make_dataset(_TMP / "ds_loaders")
    cfg = dict(batch=4, workers=0, split=(0.7, 0.15, 0.15), seed=1,
               group_split=True, aug_level="none", pad_mode="edge")
    _t, _v, _te, meta = build_loaders(str(root), cfg)
    split_dir = _TMP / "ds_loaders_splits"
    save_splits(meta["splits"], split_dir)
    _t2, _v2, _te2, meta2 = build_loaders(str(root), cfg, split_dir=str(split_dir))
    assert [s.fname for s in meta["splits"]["val"]] == [s.fname for s in meta2["splits"]["val"]]


# ------------------------------------------------------------------- metrics


def test_metrics_use_edit_distance():
    m = compute_metrics(["12345", "0000"], ["12345", "0000"])
    assert m["exact"] == 1.0 and m["cer"] == 0.0
    # A dropped leading digit is one edit, not five positional mismatches.
    m = compute_metrics(["2345"], ["12345"])
    assert m["exact"] == 0.0
    assert abs(m["cer"] - 1 / 5) < 1e-9


# ---------------------------------------------------------------- end to end


def test_train_predict_export_roundtrip():
    root = _make_dataset(_TMP / "ds_e2e", n=32)
    cfg = dict(epochs=2, batch=4, lr=3e-4, weight_decay=1e-4, workers=0,
               warmup_epochs=1, patience=10, amp=False, device="cpu",
               split=(0.75, 0.125, 0.125), seed=0, group_split=True,
               aug_level="digital", pad_mode="edge", img_h=32, img_w=128,
               resize_mode="pad", blank_bias=-2.0, ema_decay=0.9, eval_ema=True,
               ctc_weight=1.0, align_weight=0.5, align_warmup_epochs=5,
               sgm_weight=1.0, sgm_start_epoch=1, sgm_warmup_epochs=1,
               grad_clip=5.0, min_lr=1e-6)
    best = fit("svtrv2-s", str(root), cfg, project=str(_TMP / "runs"), name="e2e")
    assert Path(best).exists()

    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert ckpt["model_name"] == "svtrv2-s"
    assert ckpt["charset"] == CTCCodec.charset
    assert (Path(best).parent / "splits" / "val.txt").exists()

    net, _ = load_checkpoint(best, torch.device("cpu"))
    text, conf = predict_image(net, root / "images" / "WB000-0000.png",
                               CTCCodec(), torch.device("cpu"))
    assert isinstance(text, str) and 0.0 <= conf <= 1.0
    assert len(predict_dir(net, root / "images", CTCCodec(), torch.device("cpu"))) == 32

    out = export_onnx(net, 32, 128, str(_TMP / "e2e.onnx"), check=True)
    assert Path(out).exists()


def test_resume_restores_optimizer_epoch_and_history():
    """A 250-epoch run that dies must continue, not restart.

    Guards the checkpoint key names too: extras were once all suffixed
    "_state", so resume looked for `history` and found `history_state`.
    """
    root = _make_dataset(_TMP / "ds_resume", n=16)
    base = dict(batch=4, lr=3e-4, weight_decay=1e-4, workers=0, warmup_epochs=1,
                patience=50, amp=False, device="cpu", compile=False,
                split=(0.75, 0.125, 0.125), seed=0, group_split=True,
                aug_level="none", pad_mode="edge", img_h=32, img_w=128,
                resize_mode="pad", blank_bias=-2.0, ema_decay=0.9, eval_ema=True,
                ctc_weight=1.0, align_weight=0.5, align_warmup_epochs=5,
                sgm_weight=1.0, sgm_start_epoch=1, sgm_warmup_epochs=1,
                grad_clip=5.0, min_lr=1e-6)
    project = str(_TMP / "runs_resume")
    fit("svtrv2-s", str(root), dict(base, epochs=2), project=project, name="r")

    last = Path(project) / "r" / "last.pth"
    ck = torch.load(last, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2
    for key in ("optimizer_state", "scheduler_state", "history", "best", "wait"):
        assert key in ck, f"{key} missing from last.pth"
    assert len(ck["history"]) == 2

    fit("svtrv2-s", str(root), dict(base, epochs=4, resume=True),
        project=project, name="r")
    ck2 = torch.load(last, map_location="cpu", weights_only=False)
    assert ck2["epoch"] == 4
    assert len(ck2["history"]) == 4, "resume must extend history, not restart it"
    assert [h["epoch"] for h in ck2["history"]] == [1, 2, 3, 4]


def test_resume_without_a_checkpoint_fails_loudly():
    """Silently starting from scratch would waste hours before anyone noticed."""
    root = _make_dataset(_TMP / "ds_resume_missing", n=8)
    cfg = dict(epochs=1, batch=4, lr=3e-4, workers=0, warmup_epochs=1, patience=5,
               amp=False, device="cpu", compile=False, resume=True,
               split=(0.75, 0.125, 0.125), seed=0, group_split=True,
               aug_level="none", pad_mode="edge", img_h=32, img_w=128,
               resize_mode="pad", blank_bias=-2.0, ema_decay=0.9, eval_ema=True,
               ctc_weight=1.0, align_weight=0.5, align_warmup_epochs=5,
               sgm_weight=1.0, sgm_start_epoch=1, sgm_warmup_epochs=1,
               grad_clip=5.0, min_lr=1e-6, weight_decay=1e-4)
    try:
        fit("svtrv2-s", str(root), cfg, project=str(_TMP / "runs_missing"), name="nope")
    except FileNotFoundError:
        return
    raise AssertionError("resume with no last.pth must raise FileNotFoundError")


def test_inference_transform_follows_the_checkpoints_pad_mode():
    """Serving with a different pad_mode than training shows the model a border
    it never saw."""
    from svtrv2.engine import transform_for

    assert transform_for({"model_config": {"pad_mode": "black"}}).transforms
    img = _render("012345", dw=40, dh=200)   # tall, so padding is wide
    black = transform_for({"model_config": {"pad_mode": "black"}})(image=img)["image"]
    edge = transform_for({"model_config": {"pad_mode": "edge"}})(image=img)["image"]
    assert not torch.equal(black, edge), "pad_mode must actually change preprocessing"
    # Missing model_config falls back to the default rather than crashing.
    assert transform_for({})(image=img)["image"].shape == edge.shape


def test_predict_dir_batches_bins_and_survives_unreadable_files():
    root = _make_dataset(_TMP / "ds_predict", n=12)
    # A file with an image extension that cv2 cannot decode.
    (root / "images" / "broken.png").write_text("not an image", encoding="utf-8")

    net = SVTRNet(num_classes=NUM_CLASSES, **MODELS["svtrv2-s"]).eval()
    results = predict_dir(net, root / "images", CTCCodec(), torch.device("cpu"), batch=4)
    assert len(results) == 13
    by_name = {n: (t, c) for n, t, c in results}
    assert by_name["broken.png"] == ("", 0.0), "unreadable file must not kill the run"
    # Filename order is preserved despite bin grouping.
    assert [n for n, _, _ in results] == sorted(by_name, key=str.lower)


def _run_standalone() -> int:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n  {len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
