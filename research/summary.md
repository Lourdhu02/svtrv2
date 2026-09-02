# SVTRv2 Research Synthesis

Date: 2026-09-02

This repo is now treated as a paper-first SVTRv2 research workspace.
The current codebase is faithful to the core SVTRv2 mechanism, but it is also
intentionally narrower and more operational than the paper:

- core paper mechanics are preserved: MSR, FRM, SGM, local/global mixing rules
- the current repo is benchmark-hardened: persisted splits, group-aware splitting,
  exact-match plus CER reporting, fixed-width ONNX export
- the current harness is narrower than the paper: digits plus decimal point

The key follow-up opportunity is not to re-implement SVTRv2, but to make a new
paper on an extension that preserves CTC simplicity at inference while improving
robustness on harder benchmarks.

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
