"""svtrv2 command-line interface."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .config import MODELS, VARIANTS, load_config, resolve_model_name


def _add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="svtrv2-m", help=f"one of {', '.join(VARIANTS)} (or s/m/l/xl)")
    p.add_argument("--data", required=True, help="dataset dir (images/ + labels.txt)")
    p.add_argument("--config", default=None, help="optional YAML overriding the defaults")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda, cpu, auto")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="exp")
    p.add_argument("--aug-level", default=None, choices=["digital", "light", "none"])
    p.add_argument("--pad-mode", default=None, choices=["edge", "reflect", "black"])
    p.add_argument("--amp-dtype", default=None, choices=["bf16", "fp16"])
    p.add_argument("--resume", action="store_true",
                   help="continue from <project>/<name>/last.pth with optimizer, "
                        "scheduler, EMA, epoch and history intact")
    p.add_argument("--no-compile", action="store_true",
                   help="disable torch.compile (default: on for CUDA)")
    p.add_argument("--no-ema-eval", action="store_true",
                   help="validate and select on live weights instead of EMA")
    p.add_argument("--ctc-weight", type=float, default=None)
    p.add_argument("--align-weight", type=float, default=None)
    p.add_argument("--align-warmup-epochs", type=int, default=None)
    p.add_argument("--sgm-weight", type=float, default=None)
    p.add_argument("--sgm-start-epoch", type=int, default=None)
    p.add_argument("--sgm-warmup-epochs", type=int, default=None)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="svtrv2", description=f"SVTRv2 digital meter OCR v{__version__}")
    sub = parser.add_subparsers(dest="command")

    _add_train_args(sub.add_parser("train", help="train a model"))

    vp = sub.add_parser("val", help="evaluate a checkpoint")
    vp.add_argument("--ckpt", required=True)
    vp.add_argument("--data", required=True)
    vp.add_argument("--batch", type=int, default=None)
    vp.add_argument("--device", default=None)

    pp = sub.add_parser("predict", help="read one image or a folder")
    pp.add_argument("--ckpt", required=True)
    pp.add_argument("--source", required=True, help="image file or directory")
    pp.add_argument("--device", default=None)
    pp.add_argument("-b", "--batch", type=int, default=32,
                    help="batch size for directory input (default 32)")
    pp.add_argument("-o", "--out", default=None, help="CSV path for directory input")
    pp.add_argument("--min-conf", type=float, default=0.90,
                    help="rows below this are flagged REVIEW (default 0.90)")

    ep = sub.add_parser("export", help="export ONNX")
    ep.add_argument("--ckpt", required=True)
    ep.add_argument("--output", default=None)
    ep.add_argument("--opset", type=int, default=18)
    ep.add_argument("--bin", default="medium", choices=["short", "medium", "long"])

    ip = sub.add_parser("info", help="show a variant's shape and parameter count")
    ip.add_argument("--model", default="svtrv2-m")

    bp = sub.add_parser("bins", help="measure a dataset and suggest MSR bin edges")
    bp.add_argument("--data", required=True)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0

    print(f"\n  svtrv2 v{__version__}\n")

    if args.command == "train":
        return _cmd_train(args)
    if args.command == "val":
        return _cmd_val(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "export":
        from .export import run_export

        run_export(args.ckpt, args.output, args.opset, args.bin)
        return 0
    if args.command == "info":
        return _cmd_info(args)
    if args.command == "bins":
        return _cmd_bins(args)
    return 0


def _cmd_train(args) -> int:
    variant = resolve_model_name(args.model)
    if variant not in MODELS:
        print(f"error: unknown model {args.model!r}; expected one of {', '.join(VARIANTS)}",
              file=sys.stderr)
        return 1

    from .engine import fit

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("command", "model", "data", "config", "project", "name",
                              "no_ema_eval", "no_compile")}
    if args.no_ema_eval:
        overrides["eval_ema"] = False
    if args.no_compile:
        overrides["compile"] = False
    cfg = load_config(args.config, **overrides)
    fit(variant, args.data, cfg, args.project, args.name)
    return 0


def _cmd_val(args) -> int:
    import torch

    from .dataset import build_loaders
    from .engine import evaluate, load_checkpoint, resolve_device
    from .text import CTCCodec

    device = resolve_device(args.device)
    net, ckpt = load_checkpoint(args.ckpt, device)
    cfg = dict(ckpt.get("config") or load_config())
    if args.batch:
        cfg["batch"] = args.batch

    # Evaluate on the very splits this checkpoint was trained with, when the run
    # directory is still beside it.
    split_dir = Path(args.ckpt).parent / "splits"
    _train, val_loader, test_loader, meta = build_loaders(
        args.data, cfg, CTCCodec(), str(split_dir) if split_dir.is_dir() else None)

    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    for split, loader in (("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        m = evaluate(net, loader, device, cfg, CTCCodec(), ctc)
        print(f"  {split:>5}:  n={int(m['n']):,}  loss={m['loss']:.4f}  "
              f"exact={m['exact']:.1%}  char={m['char_acc']:.1%}  cer={m['cer']:.4f}")
    return 0


def _cmd_predict(args) -> int:
    from .engine import (load_checkpoint, predict_dir, predict_image,
                         resolve_device, transform_for)
    from .text import CTCCodec

    device = resolve_device(args.device)
    net, ckpt = load_checkpoint(args.ckpt, device)
    codec = CTCCodec()
    # Serve with the same preprocessing the checkpoint was trained under.
    transform = transform_for(ckpt)
    src = Path(args.source)
    if not src.exists():
        print(f"error: source not found: {src}", file=sys.stderr)
        return 1

    acc = ckpt.get("val_acc")
    print(f"  {ckpt.get('model_name', '?')}  |  epoch {ckpt.get('epoch')}"
          f"{f'  |  val {acc:.1%}' if isinstance(acc, float) else ''}"
          f"  |  device {device}\n")

    if src.is_file():
        text, conf = predict_image(net, src, codec, device, transform)
        flag = "  REVIEW" if conf < args.min_conf else ""
        print(f"  {src.name} -> {text or '<empty>'}  (conf {conf:.3f}){flag}")
        return 0

    results = predict_dir(net, src, codec, device, transform, batch=args.batch)
    if not results:
        print(f"error: no images found in {src}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else src / "predictions.csv"
    flagged = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "reading", "confidence", "flag"])
        for name, text, conf in results:
            flag = "REVIEW" if conf < args.min_conf else ""
            flagged += bool(flag)
            w.writerow([name, text, f"{conf:.4f}", flag])
    print(f"  {len(results):,} images  |  {flagged:,} flagged below {args.min_conf:.2f}")
    print(f"  wrote {out_path}")
    return 0


def _cmd_info(args) -> int:
    import torch

    from .config import CHARSET, MSR_BINS, NUM_CLASSES
    from .model import SVTRNet

    variant = resolve_model_name(args.model)
    if variant not in MODELS:
        print(f"error: unknown model {args.model!r}", file=sys.stderr)
        return 1
    spec = MODELS[variant]
    net = SVTRNet(num_classes=NUM_CLASSES, **spec).eval()
    total = sum(p.numel() for p in net.parameters())
    sgm = sum(p.numel() for n, p in net.named_parameters() if n.startswith("sgm."))

    print(f"  name:     {variant}")
    print(f"  params:   {total:,} total  |  {total - sgm:,} at inference "
          f"(SGM is train-only)")
    print(f"  dims:     {spec['dims']}")
    print(f"  depths:   {spec['depths']}")
    print(f"  heads:    {tuple(max(1, d // 32) for d in spec['dims'])}")
    print(f"  mixers:   {' '.join(''.join(m[0] for m in stage) for stage in spec['mixers'])}")
    print(f"  charset:  {CHARSET!r}  ->  {NUM_CLASSES} classes (incl. CTC blank)")
    for b in MSR_BINS:
        with torch.no_grad():
            out = net(torch.randn(1, 3, int(b["height"]), int(b["width"])))
        print(f"  msr {b['name']:>6}:  {b['height']}x{b['width']}  ->  "
              f"{out.shape[1]} timesteps x {out.shape[2]} classes")
    return 0


def _cmd_bins(args) -> int:
    from .config import MSR_BINS
    from .dataset import measure_bins

    stats = measure_bins(args.data)
    if not stats.get("n"):
        print(f"error: no readable images in {args.data}", file=sys.stderr)
        return 1

    print(f"  images:   {stats['n']:,}")
    print(f"  aspect:   min {stats['min']:.2f}  mean {stats['mean']:.2f}  max {stats['max']:.2f}")
    print("  percentiles: " + "  ".join(f"p{p}={v:.2f}" for p, v in stats["percentiles"].items()))
    ll = stats["label_len"]
    print(f"  label len: min {ll['min']}  mean {ll['mean']:.1f}  max {ll['max']}")
    print(f"\n  current bin edges: {[b['max_ar'] for b in MSR_BINS[:-1]]}")
    lo, hi = stats["suggested_edges"]
    print(f"  suggested edges (terciles): [{lo:.2f}, {hi:.2f}]")
    print("\n  Set width ~= aspect * height for each bin so the canvas is mostly signal,")
    print("  and keep timesteps = width // 8.")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
