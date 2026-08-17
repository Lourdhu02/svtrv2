"""Training / evaluation / inference engine.

Single-process by design.  The distributed and sharded-IO machinery the earlier
version carried was unused at this data scale and made every code path harder to
follow; one GPU driven by an MSR batch sampler saturates fine.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from .config import DEFAULTS, MODELS, NUM_CLASSES
from .dataset import build_loaders, save_splits
from .losses import SGMLoss, UniformAlignmentLoss
from .model import SVTRNet
from .text import CTCCodec
from .transforms import MSRTransform, select_msr_bin


# ------------------------------------------------------------------ plumbing


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device and device != "auto":
        d = torch.device(device)
        return torch.device("cpu") if d.type == "cuda" and not torch.cuda.is_available() else d
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelEMA:
    """Exponential moving average of the weights, used for eval and serving.

    CTC training weights are peaky epoch to epoch; the average is smoother and
    consistently scores better, so ``best.pth`` stores it by default.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        import copy

        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            for ep, mp in zip(self.ema.parameters(), model.parameters()):
                ep.mul_(d).add_(mp.detach(), alpha=1 - d)
            for eb, mb in zip(self.ema.buffers(), model.buffers()):
                eb.copy_(mb)


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    warmup = cfg.get("warmup_epochs", 5)
    total = cfg["epochs"]
    min_ratio = cfg.get("min_lr", 1e-6) / cfg["lr"]

    def lr_fn(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        prog = (epoch - warmup) / max(total - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


def loss_weights(epoch: int, cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Weights for (CTC, alignment warmup, SGM) at a given 1-based epoch.

    CTC is the anchor and holds its configured weight throughout.  The alignment
    warmup decays linearly to zero: it exists only to stop the early all-blank
    collapse, and leaving it on would fight CTC's own alignment later.  SGM ramps
    in from ``sgm_start_epoch``, once the visual path is already emitting digits.
    """
    ctc_w = float(cfg.get("ctc_weight", 1.0))

    warm = int(cfg.get("align_warmup_epochs", 40))
    align_w = float(cfg.get("align_weight", 0.5))
    align_w = align_w * max(0.0, 1.0 - (epoch - 1) / max(warm, 1)) if warm > 0 else 0.0

    start = int(cfg.get("sgm_start_epoch", 5))
    ramp = int(cfg.get("sgm_warmup_epochs", 10))
    sgm_w = float(cfg.get("sgm_weight", 1.0))
    if epoch < start:
        sgm_w = 0.0
    elif ramp > 0:
        sgm_w *= min(1.0, (epoch - start + 1) / ramp)
    return ctc_w, align_w, sgm_w


def _amp_dtype(cfg: Dict[str, Any]) -> torch.dtype:
    if cfg.get("amp_dtype", "bf16") == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# -------------------------------------------------------------------- metrics


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compute_metrics(preds: List[str], gts: List[str]) -> Dict[str, float]:
    """Exact-match, character accuracy and CER.

    Character accuracy is edit-distance based rather than positional, so one
    dropped digit costs a single edit instead of misaligning the whole tail.
    """
    n = max(len(gts), 1)
    exact = sum(p == g for p, g in zip(preds, gts)) / n
    edits = sum(_levenshtein(p, g) for p, g in zip(preds, gts))
    chars = max(sum(len(g) for g in gts), 1)
    return {"exact": exact, "char_acc": 1 - edits / chars, "cer": edits / chars,
            "n": float(len(gts))}


# --------------------------------------------------------------- train / eval


def train_epoch(net: SVTRNet, loader, optimizer, device, cfg, epoch, ema=None,
                ctc_loss=None, align_loss=None, sgm_loss=None, scaler=None,
                fwd_features=None, fwd_sgm=None) -> Dict[str, float]:
    net.train()
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    dtype = _amp_dtype(cfg)
    ctc_w, align_w, sgm_w = loss_weights(epoch, cfg)
    fwd_features = fwd_features or net.forward_features
    fwd_sgm = fwd_sgm or net.forward_sgm

    sums = dict(total=0.0, ctc=0.0, align=0.0, sgm=0.0)
    steps = 0
    try:
        from tqdm import tqdm

        loader = tqdm(loader, leave=False, desc=f"  epoch {epoch}",
                      bar_format="{l_bar}{bar:24}{r_bar}")
    except Exception:
        pass

    for images, targets, lengths, _labels, padded, _bin in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        padded = padded.to(device, non_blocking=True)

        with autocast(device.type, dtype=dtype, enabled=use_amp):
            feats = fwd_features(images)
            # fp32 head: keeps the CTC log-probs numerically stable under AMP.
            log_probs = net.head(feats.float()).float().log_softmax(2)

            b, t, _ = log_probs.shape
            in_len = torch.full((b,), t, dtype=torch.long, device=device)
            loss_ctc = ctc_loss(log_probs.permute(1, 0, 2), targets, in_len, lengths)
            loss = ctc_w * loss_ctc

            loss_align = torch.zeros((), device=device)
            if align_w > 0:
                loss_align = align_loss(log_probs, padded, lengths, in_len)
                loss = loss + align_w * loss_align

            loss_sgm = torch.zeros((), device=device)
            if sgm_w > 0:
                loss_sgm = sgm_loss(fwd_sgm(feats.float(), padded), padded)
                loss = loss + sgm_w * loss_sgm

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            # fp16 only: gradients underflow without loss scaling, and they must
            # be unscaled before clipping or grad_clip is applied to the scaled
            # values and effectively does nothing.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), cfg.get("grad_clip", 5.0))
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped = scaler.get_scale() >= before
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg.get("grad_clip", 5.0))
            optimizer.step()
            stepped = True
        # A skipped step means the scaler rejected the batch; folding those
        # weights into the EMA would poison it.
        if ema is not None and stepped:
            ema.update(net)

        sums["total"] += float(loss.detach())
        sums["ctc"] += float(loss_ctc.detach())
        sums["align"] += float(loss_align.detach())
        sums["sgm"] += float(loss_sgm.detach())
        steps += 1

    return {k: v / max(steps, 1) for k, v in sums.items()}


@torch.no_grad()
def evaluate(net: SVTRNet, loader, device, cfg, codec: CTCCodec, ctc_loss) -> Dict[str, float]:
    net.eval()
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    dtype = _amp_dtype(cfg)
    preds: List[str] = []
    gts: List[str] = []
    total = 0.0
    steps = 0

    for images, targets, lengths, labels, _padded, _bin in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device.type, dtype=dtype, enabled=use_amp):
            log_probs = net(images).float()
        b, t, _ = log_probs.shape
        in_len = torch.full((b,), t, dtype=torch.long, device=device)
        total += float(ctc_loss(log_probs.permute(1, 0, 2), targets.to(device),
                                in_len, lengths.to(device)))
        idx = log_probs.argmax(2).cpu().numpy()
        preds.extend(codec.decode(row) for row in idx)
        gts.extend(labels)
        steps += 1

    metrics = compute_metrics(preds, gts)
    metrics["loss"] = total / max(steps, 1)
    return metrics


def _save(path: Path, net: nn.Module, ema: ModelEMA, variant: str,
          model_cfg: Dict[str, Any], cfg: Dict[str, Any], epoch: int,
          metrics: Dict[str, float], best_uses_ema: bool = True, **extra) -> None:
    ckpt: Dict[str, Any] = dict(
        model_name=variant, model_config=model_cfg, model_state=net.state_dict(),
        ema_state=ema.ema.state_dict(), best_uses_ema=best_uses_ema,
        charset=CTCCodec.charset, epoch=epoch, metrics=metrics,
        val_acc=metrics.get("exact"), config=cfg,
    )
    # Anything with a state_dict is stored under "<key>_state"; plain values
    # (best, wait, history) keep their own name, because that is what resume
    # reads them back by.
    for k, v in extra.items():
        if hasattr(v, "state_dict"):
            ckpt[f"{k}_state"] = v.state_dict()
        else:
            ckpt[k] = v
    torch.save(ckpt, path)


def fit(variant: str, data_dir: str, cfg: Dict[str, Any],
        project: str = "runs", name: str = "exp") -> str:
    device = resolve_device(cfg.get("device"))
    seed_all(cfg.get("seed", 42))
    codec = CTCCodec()
    run_dir = Path(project) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    split_dir = run_dir / "splits"

    train_loader, val_loader, test_loader, meta = build_loaders(
        data_dir, cfg, codec, str(split_dir) if split_dir.is_dir() else None)
    save_splits(meta["splits"], split_dir)

    model_cfg: Dict[str, Any] = dict(num_classes=NUM_CLASSES, **MODELS[variant])
    for k in ("img_h", "img_w", "resize_mode", "pad_mode", "blank_bias"):
        model_cfg[k] = cfg.get(k, DEFAULTS[k])
    model_cfg["msr"] = True
    net = SVTRNet(**model_cfg).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = build_scheduler(optimizer, cfg)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    align_loss = UniformAlignmentLoss()
    sgm_loss = SGMLoss()
    ema = ModelEMA(net, cfg.get("ema_decay", 0.999))
    eval_ema = bool(cfg.get("eval_ema", True))

    use_amp = cfg.get("amp", True) and device.type == "cuda"
    # A GradScaler is required for fp16 and must stay disabled for bf16: bf16 has
    # fp32's exponent range, so scaling buys nothing and only adds skipped steps.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and _amp_dtype(cfg) == torch.float16)

    best, wait, history = -1.0, 0, []
    best_path, last_path = run_dir / "best.pth", run_dir / "last.pth"
    start_epoch = 1

    if cfg.get("resume"):
        if not last_path.exists():
            raise FileNotFoundError(
                f"--resume was given but {last_path} does not exist")
        ck = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(ck["model_state"])
        ema.ema.load_state_dict(ck["ema_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        if "scaler_state" in ck:
            scaler.load_state_dict(ck["scaler_state"])
        start_epoch = int(ck["epoch"]) + 1
        best = float(ck.get("best", -1.0))
        wait = int(ck.get("wait", 0))
        history = list(ck.get("history", []))
        print(f"  resumed from epoch {start_epoch} (best score {best:.4f}, patience {wait})")

    # MSRBatchSampler never mixes bins inside a batch, so the compiled graph sees
    # exactly three static shapes.  Compilation is attempted, not assumed: if
    # Inductor cannot handle this torch build we fall back to eager rather than
    # failing the run.
    fwd_features, fwd_sgm = net.forward_features, net.forward_sgm
    if cfg.get("compile", True) and device.type == "cuda":
        try:
            fwd_features = torch.compile(net.forward_features, dynamic=False)
            fwd_sgm = torch.compile(net.forward_sgm, dynamic=False)
            print("  compile: on (first batch per MSR bin pays the compile cost)")
        except Exception as exc:
            fwd_features, fwd_sgm = net.forward_features, net.forward_sgm
            print(f"  compile: unavailable ({type(exc).__name__}) - running eager")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n  {variant}  |  {n_params:,} params  |  device {device}")
    print(f"  data: {data_dir}  |  split {meta['counts']}")
    print(f"  amp: {cfg.get('amp_dtype', 'bf16') if use_amp else 'off'}"
          f"  |  output: {run_dir}\n")

    hdr = (f"  {'Epoch':>8}  {'Loss':>7}  {'CTC':>7}  {'Align':>7}  {'SGM':>7}  "
           f"{'VLoss':>7}  {'Exact':>7}  {'Char':>7}  {'CER':>7}  {'LR':>9}  {'P':>3}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        sampler = getattr(train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        tr = train_epoch(net, train_loader, optimizer, device, cfg, epoch, ema,
                         ctc_loss, align_loss, sgm_loss, scaler,
                         fwd_features, fwd_sgm)
        scheduler.step()

        eval_net = ema.ema if eval_ema else net
        val = evaluate(eval_net, val_loader, device, cfg, codec, ctc_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, lr=lr,
                            **{f"train_{k}": v for k, v in tr.items()},
                            **{k: val[k] for k in ("loss", "exact", "char_acc", "cer")}))

        # Rank on exact-match, tie-broken by character accuracy, so best.pth
        # still tracks a genuinely improving model before the first exact match
        # appears -- and early stopping does not fire during that phase.
        score = val["exact"] + 0.01 * val["char_acc"]
        mark = ""
        if score > best or not best_path.exists():
            _save(best_path, eval_net, ema, variant, model_cfg, cfg, epoch, val,
                  best_uses_ema=eval_ema)
        if score > best:
            best, wait, mark = score, 0, " *"
        else:
            wait += 1
        _save(last_path, net, ema, variant, model_cfg, cfg, epoch, val,
              optimizer=optimizer, scheduler=scheduler, scaler=scaler,
              best=best, wait=wait, history=history)

        print(f"  {epoch:>4}/{cfg['epochs']:<3}  {tr['total']:>7.4f}  {tr['ctc']:>7.4f}  "
              f"{tr['align']:>7.4f}  {tr['sgm']:>7.4f}  {val['loss']:>7.4f}  "
              f"{val['exact']:>6.1%}  {val['char_acc']:>6.1%}  {val['cer']:>7.4f}  "
              f"{lr:>9.2e}  {wait:>3}{mark}")

        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if wait >= cfg.get("patience", 60):
            print(f"\n  early stop @ epoch {epoch}")
            break

    if test_loader is not None and best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model_state"])
        test = evaluate(net, test_loader, device, cfg, codec, ctc_loss)
        print(f"\n  test  |  exact {test['exact']:.1%}  char {test['char_acc']:.1%}  "
              f"cer {test['cer']:.4f}  (n={int(test['n'])})")
        (run_dir / "test.json").write_text(json.dumps(test, indent=2), encoding="utf-8")

    return str(best_path)


# ------------------------------------------------------------------ inference


def load_checkpoint(path: str, device: torch.device, prefer_ema: bool = True
                    ) -> Tuple[SVTRNet, Dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = SVTRNet(**ckpt["model_config"])
    state = ckpt["ema_state"] if (prefer_ema and "ema_state" in ckpt) else ckpt["model_state"]
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    net.eval().to(device)
    return net, ckpt


def transform_for(ckpt: Dict[str, Any]) -> MSRTransform:
    """Inference transform matching how this checkpoint was trained.

    `pad_mode` is a preprocessing choice baked into the weights: a model trained
    against replicated edges and served with black padding sees a hard border it
    never saw in training.  It is stored in the checkpoint precisely so serving
    cannot drift from training.
    """
    pad_mode = (ckpt.get("model_config") or {}).get("pad_mode", "edge")
    return MSRTransform(training=False, aug_level="none", pad_mode=pad_mode)


def read_rgb(path) -> np.ndarray:
    """IMREAD_COLOR always yields 3 channels, so a single-channel file on disk
    still satisfies the RGB-only contract."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _infer_autocast(device: torch.device):
    """bf16 where the GPU supports it, fp16 otherwise, off on CPU.

    Hardcoding bf16 crashes on pre-Ampere cards, which is exactly the hardware a
    field deployment is likely to run on.
    """
    if device.type != "cuda":
        return autocast(device.type, enabled=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return autocast("cuda", dtype=dtype, enabled=True)


@torch.no_grad()
def predict_image(net: SVTRNet, path, codec: CTCCodec, device: torch.device,
                  transform=None) -> Tuple[str, float]:
    transform = transform or MSRTransform(training=False, aug_level="none")
    tensor = transform(image=read_rgb(path))["image"].unsqueeze(0).to(device)
    with _infer_autocast(device):
        log_probs = net(tensor).float()
    text, conf, _ = codec.decode_with_conf(log_probs[0].cpu().numpy())
    return text, conf


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


@torch.no_grad()
def predict_dir(net: SVTRNet, source_dir, codec: CTCCodec, device: torch.device,
                transform=None, batch: int = 32) -> List[Tuple[str, str, float]]:
    """Read every image in a folder, batching within each MSR bin.

    Images of different aspect ratios land on different canvases and cannot share
    a batch, so files are grouped by bin first and the results re-sorted into
    filename order.  One-at-a-time inference wastes most of the GPU.
    """
    files = sorted(f for f in Path(source_dir).iterdir()
                   if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if not files:
        return []
    transform = transform or MSRTransform(training=False, aug_level="none")

    groups: Dict[str, List[Tuple[int, Path, torch.Tensor]]] = {}
    results: List[Optional[Tuple[str, str, float]]] = [None] * len(files)
    for i, f in enumerate(files):
        try:
            img = read_rgb(f)
        except FileNotFoundError:
            # An unreadable file yields an empty, zero-confidence read rather
            # than killing the whole batch run.
            results[i] = (f.name, "", 0.0)
            continue
        bin_name = select_msr_bin(img.shape[0], img.shape[1])["name"]
        groups.setdefault(str(bin_name), []).append((i, f, transform(image=img)["image"]))

    for items in groups.values():
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            stacked = torch.stack([t for _, _, t in chunk]).to(device)
            with _infer_autocast(device):
                log_probs = net(stacked).float()
            for (idx, path, _), lp in zip(chunk, log_probs):
                text, conf, _ = codec.decode_with_conf(lp.cpu().numpy())
                results[idx] = (path.name, text, conf)

    return [r for r in results if r is not None]
