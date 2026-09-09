# Status snapshot — 2026-09-08 (v1.1.0, post-ARD session)

Read `plan.md` for phases and commands. This file says what is actually done, with
evidence, and what to pick up next. Update it as you work.

## Completed ✅ (this session)

| Item | Evidence |
|---|---|
| Architecture verified & fixed against official OpenOCR SVTRv2 | local mixing = two grouped **3x3** convs (`LNConvTwo33`); stage-2 merge does **not** downsample (`sub_k=[[1,1],[2,1]]`) -> timesteps = width//4; `use_pos_embed=False` confirmed. Tests: `test_local_mixing_is_two_3x3_grouped_convs`, `test_backbone_timesteps_are_width_over_four` |
| Dead code removed | `FocalCTCLoss`, `CenterLoss` deleted from `losses.py` (unused leftovers) |
| **ARD part 1: adaptive MSR routing** | `svtrv2/routing.py` (router net, straight-through routing, Bradley-Terry preference loss from exact-canvas CTC exploration); wired into `fit`/`train_epoch`/`predict_dir` behind `--route` (default off) |
| **ARD part 2: SGM -> CTC distillation** | `svtrv2/distill.py` (uniform + Viterbi alignment — Viterbi verified against exhaustive brute-force enumeration; KL+CE objective); behind `--distill` (default off) |
| Paper recipe preset | `--preset paper`: AdamW wd 0.05, OneCycleLR, 20 epochs, lr 6.5e-4, no-decay on bias/norm/embed (per official `svtrv2_tiny_rctc.yml`) |
| Test suite grown and green | **49/49** (`pytest tests -q`): 32 legacy + 17 novelty (`tests/test_novelty.py`), incl. end-to-end CPU fit with route+distill on |
| Docs | `research/method/method.md` (formal method + ablation grid); README ARD section; `research/summary.md`, `research/plan/proposed_plan.md`, `research/gaps/research_gaps.md` updated |

## Completed ✅ (earlier)

| Item | Evidence |
|---|---|
| GPU environment (RTX 5060 8 GB, sm_120) | `.venv` with torch 2.14.0+cu130; CUDA available; model fwd (bf16/fp32), CTC backward + optimizer step on GPU |
| Benchmark data pack (~12.9 GB, 70 files) | `data/` from HF `topdu/OpenOCR-Data` (pinned rev); verified 70/70 OK |
| LMDB data layer, benchmark command, docs | commits `84b7e79`, `07ae382`, `211560d`, `b9a7cc0`; pushed to `origin/main` |

## Left ⬜ (priority order)

1. **Extract LTB**: `tar -xf data/ltb.tar.xz`; check layout; wire into `benchmark`
   if not auto-discovered. (The long-text benchmark is the research differentiator.)
2. **Smoke train on GPU**: `python -m svtrv2 train --data data/Union14M-L-LMDB-Filtered
   --model t --preset paper --epochs 2 --batch 256 --device cuda --name smoke`
3. **Baseline lock**: `--model s --preset paper --name s-base` (early stop + EMA built in).
4. **ARD ablation grid** per `research/method/method.md` §4 (route/distill on/off,
   uniform vs viterbi, oracle routing, refit static edges).
5. **Benchmark the checkpoints** and compare vs paper tables
   (`research/papers/svtrv2_notes.md`); record protocol caveats in
   `research/datasets/tooling_notes.md`.
6. **Manuscript** from `research/method/method.md` + measured tables.

## Known caveats

- `data/` is git-ignored; anyone cloning needs `tools/repair_dataset.py --all`.
- Union14M-L scan caches (`.svtrv2_scan.npz`) are charset-keyed; delete them if
  the charset changes again.
- Changing geometry (timesteps W/4) invalidates any checkpoint trained before
  2026-09-08 — none exist yet, so nothing to migrate.
- HF download tooling was flaky on this network (proxy); `repair_dataset.py` is
  the reliable path for re-downloads.
