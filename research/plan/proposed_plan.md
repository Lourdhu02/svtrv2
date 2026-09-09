# Proposed Plan

Status: **implemented behind flags** (`--route`, `--distill`), tested
(`tests/test_novelty.py`), experiments pending. See
`research/method/method.md` for the formal method and the ablation grid.

## Recommendation

Build the next paper around an adaptive, inference-preserving SVTRv2.

## Problem Statement

Fixed MSR bins and training-only context help SVTRv2, but they leave open the
question of whether the model can adapt its canvas budget more intelligently and
still keep CTC-style inference simplicity.

## Hypothesis

A learned resizing/routing policy plus distilled semantic guidance will improve
robustness on irregular and long text without adding inference cost.

## Proposed Method Sketch

- replace static MSR bucket choice with learned or semi-learned routing
- distill SGM into the backbone or a lightweight auxiliary head
- keep inference graph CTC-only
- evaluate on common, irregular, long-text, and special benchmarks

## Evaluation Plan

Primary:

- IIIT5K
- SVT
- ICDAR 2013
- ICDAR 2015
- SVTP
- CUTE80

Secondary:

- Union14M-Benchmark: Curve
- Union14M-Benchmark: Multi-Oriented
- Union14M-Benchmark: Artistic
- Union14M-Benchmark: Contextless
- Union14M-Benchmark: Salient
- Union14M-Benchmark: Multi-word
- Union14M-Benchmark: General
- LTB
- OST

## Research Sequence

1. Finalize the exact thesis.
2. Lock benchmark packs and evaluation rules.
3. Define the baseline reproduction target.
4. Decide whether the thesis is adaptive MSR, SGM distillation, or multilingual
   generalization.
5. Only then move into implementation.

## Current Recommendation

Primary thesis:

- adaptive MSR + inference-preserving SGM distillation

Secondary fallback:

- long-text / irregular-text robustness on a fixed benchmark suite

