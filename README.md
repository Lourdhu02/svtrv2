# svtrv2

**OCR engine for digital utility meter displays** — seven-segment LCD and LED
registers, photographed through the meter's enclosure. Given a pre-cropped strip
it recognizes the digit string (e.g. `04872.6`) with an SVTRv2 CTC recognizer
built to the paper (arXiv 2411.15858v2).

```bash
pip install -e .
python -m svtrv2 info  --model m
python -m svtrv2 bins  --data /path/to/dataset        # fit the MSR bins first
python -m svtrv2 train --model m --data /path/to/dataset --device cuda
```

## Model variants

Four sizes. **S / M / L are the paper's T / S / B configurations verbatim**,
renamed; XL is the one extrapolation, and it keeps every structural rule the
paper sets (heads = groups = D/32, local blocks first, global last, stage 2 as
the crossover).

| variant | dims | depths | heads | permutation | params | inference params |
|---|---|---|---|---|---|---|
| `svtrv2-s` | 64 / 128 / 256 | 3, 6, 3 | 2, 4, 8 | `[L]6[G]6` | 8.48M | 6.36M |
| `svtrv2-m` | 96 / 192 / 384 | 3, 6, 3 | 3, 6, 12 | `[L]6[G]6` | 18.34M | 13.58M |
| `svtrv2-l` | 128 / 256 / 384 | 6, 6, 6 | 4, 8, 12 | `[L]8[G]10` | 27.27M | 22.52M |
| `svtrv2-xl` | 192 / 384 / 512 | 6, 9, 9 | 6, 12, 16 | `[L]8[G]16` | 65.37M | 56.93M |

Short aliases work: `--model s|m|l|xl`. The gap between the two parameter columns
is the semantic guidance module, which is **train-only** and contributes nothing
at inference.

Charset `0123456789.` → 12 classes including the CTC blank.

## Why this design

| Concern | Choice |
|---|---|
| Recognizer | SVTRv2 CTC. Each mixing block is local **or** global — never both — with local mixing as two consecutive grouped `Conv2d` and no norm between them. |
| Variable crop shapes | **MSR**: three canvases (`64x192`, `64x288`, `64x384`) picked from the raw aspect ratio. No positional encoding anywhere, which is what lets one set of weights take all three. |
| CTC blank collapse | A uniform-alignment warmup at 0.5 decaying to zero over 40 epochs, plus a `-2.0` initial blank bias. |
| Linguistic context | SGM ramps in from epoch 5 over 10 epochs, then fuses away at inference. |
| Glare, washout, bleed | A display-specific augmentation set — see below. |
| Split leakage | `meter_id` (filename before the first `-`) never spans two splits; repeat photos of one meter are near-duplicates. |
| Serving | ONNX export per MSR bin, with a torch-vs-onnxruntime parity check. |

## Augmentation

Derived from the reported failure modes for seven-segment meter capture —
specular reflection, contrast collapse at oblique angles, segment bleeding,
non-uniform illumination, auto-exposure swing, lens blur, day/night — rather than
from a generic image-classifier recipe. WBSEDCL's meters sit in a transparent
enclosure and their LCDs are specified for only a **35° viewing cone**, which
makes glare and washout dominant in West Bengal field photos.

| op | models |
|---|---|
| `glare` | saturating elliptical hotspot off the enclosure — clips to white, destroying lit/unlit contrast underneath |
| `viewing_angle_washout` | contrast collapse past the LCD viewing cone, as a vertical gradient |
| `backlight_bloom` | segment bleeding — **polarity-aware**: dark strokes spread on a reflective LCD, light glows outward on an LED |
| `segment_fade` | an aging or under-driven segment, faint but present |
| `moire` | interference from photographing a segment/pixel grid |

One display effect per sample (`A.OneOf`, p=0.65); stacking them produces images
no camera would ever produce.

**`segment_fade` attenuates, it never erases.** Deleting the lower-left segment of
an `8` turns it into a `9` while the label still reads `8` — that trains the model
to hallucinate. Every augmentation here leaves the label true.

## Dataset format

```
<data_dir>/
  images/
  labels.txt        # "filename<TAB>reading" per line
```

`meter_id` is the filename text before the first `-`.

## MSR bins are not final

The shipped bin edges (AR 3.5 / 5.0) are starting values reasoned from WBSEDCL's
"≥6 digits" register spec, **not** fitted to your images. Run:

```bash
python -m svtrv2 bins --data /path/to/dataset
```

It reports the aspect-ratio percentiles and label lengths, and suggests edges.
Set each bin's `width ≈ aspect × 64` and keep `timesteps = width // 8`.

## CLI

```bash
python -m svtrv2 train   --model m --data DATA --device cuda --name run1
python -m svtrv2 train   --model m --data DATA --name run1 --resume   # after a crash
python -m svtrv2 val     --ckpt runs/run1/best.pth --data DATA
python -m svtrv2 predict --ckpt runs/run1/best.pth --source img_or_dir -o readings.csv
python -m svtrv2 export  --ckpt runs/run1/best.pth --bin medium
python -m svtrv2 info    --model xl
python -m svtrv2 bins    --data DATA
```

`--resume` continues from `runs/<name>/last.pth` with the optimizer, scheduler,
EMA, gradient scaler, epoch counter and history intact, and reuses the persisted
splits so the val set never reshuffles under you. It fails loudly if there is no
checkpoint rather than silently restarting.

`torch.compile` is on by default on CUDA and falls back to eager if Inductor
can't handle your build. The first batch of each MSR bin pays the compile cost —
that is not a hang. Disable with `--no-compile`.

AMP is bf16 by default; `--amp-dtype fp16` switches on a `GradScaler`
automatically (bf16 does not need one, and skipped steps are excluded from the
EMA either way).

`predict` prints one line for a single image and writes
`filename,reading,confidence,flag` for a directory, flagging rows below
`--min-conf` (default 0.90) as `REVIEW`.

Confidence is the **minimum** per-digit probability, so one shaky digit sinks the
whole read. It catches hesitant errors, not confident ones — a glare hotspot that
turns an 8 into a 9 with no hesitation in the logits still scores high. Treat it
as triage, never as proof.

## Tests

```bash
python -m pytest tests -q      # or: python tests/test_engine.py
```

CPU-only and fully synthetic — it renders its own seven-segment images, so no
dataset is required.
