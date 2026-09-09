# Research Gaps

1. MSR is still hand-set. **ADDRESSED IN CODE**: `--route` learns per-sample
   canvas choice with loss-based preference supervision
   (`svtrv2/routing.py`); experiments pending.
2. SGM is train-only. **ADDRESSED IN CODE**: `--distill` transfers the SGM's
   linguistic context into the CTC head at aligned timesteps while keeping
   inference CTC-only (`svtrv2/distill.py`); experiments pending.
3. The current repo is narrow in charset and task framing, so it does not yet
   answer general scene-text generalization.
4. Long-text behavior is not studied systematically.
5. Irregular-text robustness is not isolated from benchmark effects.
6. Confidence is exposed but not calibrated as a scientific contribution.
7. The repo is benchmark-hardened, but the paper-level decomposition between
   architecture gain and benchmark hygiene is still open.

Best candidate contribution directions:

- adaptive MSR with learned routing
- SGM distillation into the visual backbone
- multilingual/open-vocabulary scene text
- long-text robustness with stronger length scaling
- selective prediction / confidence-aware recognition

Risk ranking:

- adaptive MSR: medium risk, strongest fit to existing architecture
- SGM distillation: medium risk, clear inference-preserving story
- multilingual/open-vocabulary: medium-high risk, needs bigger data and charset work
- long-text: medium risk, likely easier to justify experimentally
- selective prediction: lower novelty, but useful as an auxiliary contribution
