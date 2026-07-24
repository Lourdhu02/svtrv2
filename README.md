# tambour

**Production OCR engine for analog mechanical rolling-counter meters** — the kind
where the reading sits on rotating digit drums (a *tambour*). Given a pre-cropped
strip, it recognizes the digit string (e.g. `088687`) with an SVTRv2-style CTC
recognizer, built for scale (millions of images, multiple meter domains) and for
the signature hard case: the **rolling/half digit** caught mid-rotation.

```bash
pip install -r requirements.txt
python -m tambour sweep                       # take all 4 variants through once
python -m tambour train --model tambour-b --data /path/to/dataset --device auto
```

## Why this design (research-backed)

| Concern | Choice |
|--------|--------|
| Recognizer | **SVTRv2-style CTC** + FRM (alignment) + SGM (train-only guidance). CTC beats encoder-decoders on short digit strings — faster *and* more accurate. |
| Rolling/half digits | Label the **lowest visible digit**; optional **AugLoss** doesn't penalize reading the in-between wheel; **synthetic rolling** augmentation oversamples it. |
| Red fractional wheel | Excluded from the billed read (`billed_reading`); the `.` boundary is kept in labels. |
| Many domains (4+) | One shared backbone + **per-domain adapters** (FiLM, identity-init) + **domain-balanced sampling**. |
| Millions of images | **WebDataset tar shards** (sequential I/O), **DDP + bf16**, group-split by `meter_id`. |
| Trust | **Calibrated confidence + abstain** — route low-confidence reads to humans for ≥99.5% accepted-read precision. |
| Deploy | ONNX export (fixed width, dynamic batch) → TensorRT/OpenVINO; CTC decode outside the graph. |

## Model variants

| variant | params (approx) | use |
|---------|------|-----|
| `tambour-n` | ~1.5M | edge / smoke / distill student |
| `tambour-s` | ~5M | light |
| `tambour-b` | ~9M | default |
| `tambour-l` | ~20M | top accuracy / teacher |

Charset `0123456789.` → 12 classes (incl. CTC blank). Input letterboxed (MSR,
aspect-preserving) to 3×48×192 (≈ the 4:1 digit-strip aspect; tune with `--img-w`).

## Dataset format

```
<data_dir>/
  images/
  labels.txt        # "filename<TAB>reading[<TAB>domain]" per line
```
`domain` is optional (inferred from the filename otherwise); `meter_id` is the text
before the first `-` and is used to keep all photos of one meter in a single split.

## CLI

```bash
python -m tambour train    --model tambour-b --data DATA --device auto --aug-loss
python -m tambour val      --ckpt runs/exp/best.pth --data DATA
python -m tambour predict  --ckpt best.pth --source img_or_dir --tau 0.9 --tta
python -m tambour calibrate --ckpt best.pth --data DATA --precision 0.995   # -> T, abstain tau
python -m tambour mine     --ckpt best.pth --source unlabeled/ --out to_relabel.txt
python -m tambour export   --ckpt best.pth --output tambour-b.onnx
python -m tambour shards   --data DATA --out shards/ --resize-h 48          # scale: pack to tar
python -m tambour sweep                                                     # all 4 variants once
python -m tambour agents   --model tambour-b --data DATA                    # health/latency/leakage
python -m tambour info     --model tambour-l
```

## Scale & portability

- **CPU / single-GPU / multi-GPU / Colab** from the same code: `resolve_device`
  falls back to CPU; AMP autocast (bf16) + GradScaler activate only on CUDA.
- **Multi-GPU:** launch with `torchrun --nproc_per_node=N -m tambour train ...`
  (DDP auto-detected from env). Non-balanced runs shard via `DistributedSampler`.
- **Millions of images:** build tar shards once (`tambour shards`) and stream them
  with `data.shards.ShardDataset` (stdlib `tarfile`, no hard dependency).

## Tests

```bash
python tests/test_engine.py        # or: python -m pytest tests
```
All tests run on CPU against synthetic rendered strips (no dataset needed): codec,
manifest + leakage-safe split, transforms, domain-balanced sampler, **forward for
all 4 variants** + domain conditioning, CTC/AugLoss backward, loaders, overfit,
confidence calibration + abstain, ONNX parity, shard round-trip, diagnostics.

## License

MIT — see [LICENSE](LICENSE).

## SVTRv2
Scene Text Recognition v2
