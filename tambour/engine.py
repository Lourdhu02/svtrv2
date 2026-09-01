"""Training/eval engine: device & DDP, EMA, scheduler, per-domain metrics, fit loop.

Portable across CPU / single-GPU / multi-GPU (DDP) / Colab. AMP autocast and the
GradScaler activate only on CUDA; everything degrades cleanly to CPU.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .config import MODELS, NUM_CLASSES
from .data import build_loaders
from .losses import CenterLoss, build_criterion
from .models import build_model, model_config
from .text import CTCCodec


# --- setup -------------------------------------------------------------------
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


def init_distributed() -> Tuple[bool, int, int, int]:
    """Init DDP from env (torchrun). Returns (is_dist, rank, world_size, local_rank)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world <= 1 or not torch.distributed.is_available():
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return True, rank, world, local


def is_main(rank: int) -> bool:
    return rank == 0


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.ema = copy.deepcopy(self._core(model)).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _core(m):
        return m.module if hasattr(m, "module") else m

    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        core = self._core(model)
        with torch.no_grad():
            for ep, mp in zip(self.ema.parameters(), core.parameters()):
                ep.mul_(d).add_(mp.detach(), alpha=1 - d)
            for eb, mb in zip(self.ema.buffers(), core.buffers()):
                eb.copy_(mb)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup = cfg.get("warmup_epochs", 3)
    total, min_ratio = cfg["epochs"], cfg.get("min_lr", 1e-6) / cfg["lr"]

    def lr_fn(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        prog = (epoch - warmup) / max(total - warmup, 1)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# --- metrics -----------------------------------------------------------------
def _amp_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def ctc_loss_inputs(log_probs, device):
    b, t, _ = log_probs.shape
    return log_probs.permute(1, 0, 2), torch.full((b,), t, dtype=torch.long, device=device)


def compute_metrics(preds: List[str], gts: List[str], domains: Optional[List[str]] = None) -> Dict:
    import editdistance
    n = max(len(gts), 1)
    exact = sum(p == g for p, g in zip(preds, gts)) / n
    edits = sum(editdistance.eval(p, g) for p, g in zip(preds, gts))
    chars = sum(len(g) for g in gts)
    out = {"exact": exact, "char_acc": 1 - edits / max(chars, 1), "cer": edits / max(chars, 1), "n": len(gts)}
    if domains is not None:
        per: Dict[str, Dict] = {}
        for p, g, d in zip(preds, gts, domains):
            slot = per.setdefault(d, {"correct": 0, "n": 0})
            slot["correct"] += int(p == g)
            slot["n"] += 1
        out["per_domain"] = {d: {"exact": v["correct"] / max(v["n"], 1), "n": v["n"]} for d, v in per.items()}
        out["worst_domain_exact"] = min((v["exact"] for v in out["per_domain"].values()), default=exact)
    return out


# --- train / eval ------------------------------------------------------------
def train_epoch(net, loader, optimizer, scaler, criterion, device, cfg,
                ema=None, center=None, scheduler=None):
    net.train()
    core = net.module if hasattr(net, "module") else net
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    sgm_w, center_w = cfg.get("sgm_weight", 0.0), cfg.get("center_weight", 0.0)
    log_interval = cfg.get("log_interval", 50)
    total = 0.0
    try:
        from tqdm import tqdm
        loader = tqdm(loader, leave=False, desc="  train", bar_format="{l_bar}{bar:24}{r_bar}")
    except Exception:
        pass

    for step, (imgs, targets, lengths, _, domain_ids) in enumerate(loader):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        lengths, domain_ids = lengths.to(device), domain_ids.to(device)
        with autocast(device.type, dtype=_amp_dtype(), enabled=use_amp):
            log_probs = net(imgs, domain_ids)
            lp, in_len = ctc_loss_inputs(log_probs, device)
            loss = criterion(lp, targets, in_len, lengths)
            if (sgm_w > 0 and core.sgm_head is not None) or center_w > 0:
                feats = core.forward_backbone(imgs)
                if center_w > 0 and center is not None:
                    loss = loss + center_w * center(feats.float(), log_probs)
                if sgm_w > 0 and core.sgm_head is not None:
                    sgm_lp, sgm_in = ctc_loss_inputs(core.sgm_head(feats).float().log_softmax(2), device)
                    loss = loss + sgm_w * criterion(sgm_lp, targets, sgm_in, lengths)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(net.parameters(), cfg.get("grad_clip", 5.0))
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= before:
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(net)
        total += float(loss.item())
        
        if step % log_interval == 0 and step > 0:
            print(f"    step {step}/{len(loader)} | loss {loss.item():.4f}")
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(net, loader, criterion, device, codec, id2domain=None, use_amp=False) -> Dict:
    net.eval()
    preds: List[str] = []
    gts: List[str] = []
    doms: List[str] = []
    total = 0.0
    for imgs, targets, lengths, labels, domain_ids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast(device.type, dtype=_amp_dtype(), enabled=use_amp):
            log_probs = net(imgs, domain_ids.to(device))
            lp, in_len = ctc_loss_inputs(log_probs, device)
            total += float(criterion(lp, targets.to(device), in_len, lengths.to(device)).item())
        idx = log_probs.argmax(2).cpu().numpy()
        preds.extend(codec.decode(row) for row in idx)
        gts.extend(labels)
        if id2domain is not None:
            doms.extend(id2domain[int(d)] for d in domain_ids)
    metrics = compute_metrics(preds, gts, doms or None)
    metrics["loss"] = total / max(len(loader), 1)
    return metrics


def _save(path, net, ema, variant, cfg, meta, metrics, epoch, full=False):
    core = net.module if hasattr(net, "module") else net
    ckpt = dict(model_name=variant, model_config=model_config(variant, len(meta["domain_to_id"]), cfg.get("sgm_weight", 0) > 0),
                model_state=core.state_dict(), ema_state=ema.ema.state_dict(),
                domain_to_id=meta["domain_to_id"], img_h=cfg["img_h"], img_w=cfg["img_w"],
                charset=CTCCodec.charset, epoch=epoch, metrics=metrics)
    torch.save(ckpt, path)


def fit(variant: str, data_dir: str, cfg: Dict, project="runs", name="exp") -> str:
    is_dist, rank, world, local = init_distributed()
    device = resolve_device(cfg.get("device")) if not is_dist else torch.device(
        f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.get("seed", 42) + rank)
    codec = CTCCodec()
    run_dir = Path(project) / name
    if is_main(rank):
        run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, meta = build_loaders(data_dir, cfg, codec, rank, world)
    id2domain = {v: k for k, v in meta["domain_to_id"].items()}
    n_domains = len(meta["domain_to_id"])

    net = build_model(variant, NUM_CLASSES, num_domains=n_domains, sgm=cfg.get("sgm_weight", 0) > 0).to(device)
    if is_dist:
        net = nn.parallel.DistributedDataParallel(
            net, device_ids=[local] if torch.cuda.is_available() else None, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.05))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    criterion = build_criterion(cfg)
    center = (CenterLoss(NUM_CLASSES, MODELS[variant]["dims"][-1]).to(device)
              if cfg.get("center_weight", 0) > 0 else None)
    use_amp = cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=use_amp)
    ema = ModelEMA(net, cfg.get("ema_decay", 0.9995))

    if is_main(rank):
        n_params = sum(p.numel() for p in net.parameters())
        print(f"  {variant} | {n_params:,} params | device {device} | domains {n_domains} "
              f"| split {meta['counts']}")

    best, wait, history = -1.0, 0, []
    best_path = run_dir / "best.pth"
    for epoch in range(1, cfg["epochs"] + 1):
        tr = train_epoch(net, train_loader, optimizer, scaler, criterion, device, cfg,
                         ema=ema, center=center, scheduler=None)
        scheduler.step()
        # Validate / select / early-stop on the EMA weights: they are what we serve at
        # inference and are far smoother than the peaky live CTC weights (which bounce
        # epoch-to-epoch and make a healthy run look stuck).
        val = evaluate(ema.ema, val_loader, criterion, device, codec, id2domain, use_amp)
        if not is_main(rank):
            continue
        history.append(dict(epoch=epoch, train_loss=tr, **{k: val[k] for k in ("loss", "exact", "char_acc")}))
        wd = f" | worst-dom {val.get('worst_domain_exact', val['exact']):.3f}" if n_domains > 1 else ""
        print(f"  epoch {epoch:>3}/{cfg['epochs']} | loss {tr:.4f} | val_loss {val['loss']:.4f} "
              f"| exact {val['exact']:.4f} | char {val['char_acc']:.4f}{wd}")
        _save(run_dir / "last.pth", net, ema, variant, cfg, meta, val, epoch)
        # Rank by exact-match, breaking ties on char accuracy, so best.pth tracks a
        # genuinely improving model (and early-stop doesn't fire) before the first
        # exact match appears.
        score = val["exact"] + 0.01 * val["char_acc"]
        if score > best or not best_path.exists():
            _save(best_path, net, ema, variant, cfg, meta, val, epoch)
        if score > best:
            best, wait = score, 0
        else:
            wait += 1
            if wait >= cfg.get("patience", 20):
                print(f"  early stop @ epoch {epoch}")
                break

    if is_main(rank):
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        if test_loader is not None and best_path.exists():
            core = net.module if hasattr(net, "module") else net
            core.load_state_dict(torch.load(best_path, map_location=device)["ema_state"])
            test = evaluate(net, test_loader, criterion, device, codec, id2domain, use_amp)
            print(f"  test | exact {test['exact']:.4f} | char {test['char_acc']:.4f}")
            with open(run_dir / "test.json", "w") as f:
                json.dump(test, f, indent=2)
    if is_dist:
        torch.distributed.destroy_process_group()
    return str(best_path)
