# SVTRv2 Paper Notes

Base paper:
- `papers/SVTRv2_2411.15858v2.pdf`

Core contributions from the paper:
- MSR: multi-size resizing to reduce distortion
- FRM: feature rearrangement to match CTC reading order
- SGM: train-only semantic guidance, removed at inference

The paper’s benchmark framing is broader than the current harness:

- common short-text benchmarks
- hard irregular scene-text benchmarks
- long-text benchmark
- occluded text benchmark

The current repo keeps the architecture but is narrower in label space and
research framing, so a follow-up paper should widen the benchmark scope rather
than only tuning the current digit task.
