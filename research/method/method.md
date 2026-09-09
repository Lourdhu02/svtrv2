# ARD: Adaptive Routing and Semantic-Guidance Distillation for CTC-Only Scene Text Recognition

Status: **implemented behind flags** (`--route`, `--distill`), tested, awaiting
experiments. Baseline behavior is untouched when the flags are off (49/49
tests green, `DEFAULTS["route"] == DEFAULTS["distill"] == False`).

## 1. Motivation

SVTRv2 (ICCV 2025) wins with three components: MSR (multi-size resizing),
FRM (feature rearrangement), and SGM (train-only semantic guidance). Two of
its design decisions leave measurable value on the table:

1. **Static MSR routing.** The canvas a crop is resized onto is chosen from
   hand-set aspect-ratio edges. The rule is blind to what actually matters:
   which canvas lets the *recognizer* read the sample. Curved or sparse text
   with a moderate ratio often reads better on a wider canvas than geometry
   suggests, and a tight crop is over-padded on a wide one.
2. **SGM is discarded at test time.** The linguistic context SGM provides
   during training never reaches the inference-time CTC head; the two
   objectives optimize the same head independently.

ARD addresses both while keeping the deployment graph exactly CTC-only.

## 2. Method

### 2.1 Adaptive MSR routing (part 1)

A router network `R` sees a fixed-size probe of the crop — a 32x128 fit-pad,
normalized like a training tensor — plus the raw aspect ratio, and outputs one
logit per MSR bin:

    z = R(probe(x), ar(x))            in R^K,  K = 4 bins

Architecture: depthwise-separable conv stem (three 2x-downsample stages) ->
global pool -> concat with a small MLP embedding of the aspect ratio -> MLP
head. Under ~40k parameters; the probe forward is a tiny fraction of one
recognition pass.

**Loss-based preference supervision.** The router is not trained to imitate
geometry; it is trained to predict which canvas the *current recognizer* reads
best. Every `N` steps (`router_explore_every`), a small subset of the batch is
materialized on two candidate canvases — the router's pick A, and the static
AR rule's bin B (or the router's runner-up when they agree) — by exact fit-pad
from the original crop (never by re-warping an already-resized tensor). Each
candidate is forwarded through the model (eval mode, no grad) and scored by
per-character CTC loss. The router minimizes a pairwise Bradley-Terry logistic
loss

    L_pref = mean  softplus( -sign_i * (z_i,A - z_i,B) ),
    sign_i = +1  if  L_ctc(i|A) <= L_ctc(i|B)

plus a decisiveness entropy regularizer and an optional label-smoothed
geometric prior (both weighted; defaults `router_weight=0.5`,
`router_entropy=0.01`, prior off).

**Inference.** `z` replaces the AR rule for bin selection; the recognition
graph is unchanged and stays CTC-only. Serving cost is one probe + router
forward per sample.

### 2.2 SGM -> CTC distillation (part 2)

During training the SGM already predicts each target position from its
left/right context streams. We reuse that computation as a soft teacher for
the CTC head:

- Teacher: the mean of the two stream distributions, softened by temperature
  tau: `q = log softmax( (p_L + p_R) / 2 )` per label position.
- Alignment: label position i is mapped to the timestep t(i) where the model
  believes that character lives. Two modes:
  - `uniform` (default): midpoint of the label's run under the same monotone
    spread the alignment warmup uses; cheap and stable.
  - `viterbi`: first emission of each label on the single best CTC path,
    computed by DP over the extended blank/label alphabet
    (`svtrv2/distill.py::viterbi_positions`); verified equivalent to
    exhaustive path enumeration in the test suite.
- Objective over valid (non-pad) positions:

    L_distill = (1 - beta) * KL( q || p_head[t(i)] ) + beta * CE( y || p_head[t(i)] )

  with `beta = distill_ce_mix` (default 0.2) anchoring the student to the
  ground truth. The weight ramps in with the SGM schedule
  (`distill_start_epoch`, `sgm_warmup_epochs`).

The student graph — and therefore the exported model — is exactly the
baseline CTC network. Distillation changes the loss, never the architecture.

## 3. What is implemented

| Piece | File | Tests |
|---|---|---|
| Router net + straight-through hard routing | `svtrv2/routing.py` | `test_router_*` |
| Preference / entropy / prior losses | `svtrv2/routing.py` | `test_router_preference_learns_the_better_bin` |
| Exact-canvas exploration pass | `svtrv2/engine.py::_bin_losses` | end-to-end fit test |
| Uniform + Viterbi alignment | `svtrv2/distill.py` | incl. brute-force equivalence |
| KL+CE distillation objective | `svtrv2/distill.py::AlignmentDistillLoss` | `test_distill_loss_is_finite_and_masked` |
| Dataset emits probe/AR/original | `svtrv2/dataset.py` | `test_router_fields_flow_through_dataset_and_collate` |
| Engine wiring, ckpt save/load, resume | `svtrv2/engine.py::fit/train_epoch` | `test_fit_trains_with_routing_and_distillation_end_to_end` |
| Routed serving | `predict_dir(..., router=...)`, `--route` | `test_predict_dir_uses_the_checkpoint_router` |
| Paper recipe preset | `--preset paper` (AdamW wd .05, OneCycle, 20 ep, lr 6.5e-4) | — |

## 4. Experiment plan (to run)

Baseline lock first: `--preset paper` on Union14M-L, `svtrv2-t` then `svtrv2-s`,
benchmarked on the common six + Union14M-B + OST + LTB. Then:

| # | Run | route | distill | Claim tested |
|---|---|---|---|---|
| 1 | baseline | – | – | reproduction anchor |
| 2 | + distill (uniform) | – | yes | SGM knowledge transfers to CTC head |
| 3 | + distill (viterbi) | – | viterbi | exact alignment beats uniform |
| 4 | + route (train-only) | yes | – | router training signal regularizes |
| 5 | ARD full | yes | yes | combined effect |
| 6 | oracle routing | oracle | – | upper bound for the router |
| 7 | refit static edges (`svtrv2 bins`) | – | – | gain is not just better edges |

Report exact / 1-CER per set (macro + weighted), routing agreement with the
static rule, and per-bin CTC deltas. Ablate `router_eval_bs`,
`router_explore_every`, `distill_temperature`, `distill_ce_mix`.

## 5. Threats to validity

- Router supervision uses the *current* model's loss — early-training signal
  is noisy; the geometric prior and small eval subsets mitigate this.
- Exploration cost: two extra forwards on `router_eval_bs` samples every
  `router_explore_every` steps (~1-2% step-time at defaults).
- Viterbi mapping costs O(T * (2L+1)) per step once distillation is active.
- Full-recipe Union14M-L training on an 8 GB laptop GPU needs the reduced-
  epoch protocol documented in `agents/status.md`; report it honestly.
