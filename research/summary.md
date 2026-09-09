# SVTRv2 Research Synthesis

Date: 2026-09-08 (updated)

This repo is now treated as a paper-first SVTRv2 research workspace.
The current codebase is faithful to the core SVTRv2 mechanism, and it now
carries a tested research extension on top:

- core paper mechanics are preserved and verified against the official
  OpenOCR implementation: MSR, FRM, SGM, local/global mixing rules, no
  positional encoding, sub_k downsample schedule, two 3x3 grouped-conv local
  mixing, W/4 timesteps
- the repo is benchmark-hardened: persisted splits, group-aware splitting,
  exact-match plus CER reporting, fixed-width ONNX export
- charset: 94 printable ASCII + space (96 classes), covering every English
  STR benchmark
- **ARD extension implemented and tested (default off):** adaptive MSR routing
  (`--route`) and SGM->CTC distillation (`--distill`), both inference-
  preserving; see `research/method/method.md` and `tests/test_novelty.py`

The key follow-up opportunity is no longer implementation but *evidence*: run
the Phase 3 baseline, then the ARD ablation grid in `research/method/method.md`,
on the planned benchmark suite.

Recommended primary thesis:

> Adaptive, inference-preserving SVTRv2: learn the resizing/routing policy
> instead of fixing MSR bins, and distill linguistic context so the test-time
> model stays CTC-only while improving robustness on irregular and long text.

Why this is the strongest direction:

- it stays close to the SVTRv2 design space
- it is easy to motivate from the current repo and paper
- it is benchmarkable on public STR sets
- it has a clear story for a publishable paper

Best secondary directions:

- multilingual scene text recognition
- long-text robustness
- irregular-text robustness on curved / perspective / occluded crops
- confidence-aware selective prediction

Research artifacts are split into:

- `research/papers/`
- `research/datasets/`
- `research/tools/`
- `research/gaps/`
- `research/modality/`
- `research/plan/`
