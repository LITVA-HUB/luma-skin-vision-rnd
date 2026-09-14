# ChromaSeed AS v1: prospective architecture scaling screen

Registered before original TRAIN fitting. Latest user instruction: prioritize quality, freely use RTX 4060, increase to approximately five million parameters and retry earlier unsuccessful architectures at scale. This bounded screen implements that instruction. It does not establish a maximum achievable accuracy or complete the broad facial product goal.

## Data and selection

Only the immutable original TRAIN cache, SHA256 d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0, is opened. Load only color36, tokens64x18, native instrument Lab3 and patient/site/device: 966 observations, 24 people. No raw images, histogram arrays, new assets or exposed legacy validation/calibration/test. Preserve mixed, SLR→iPod and iPod→SLR roles and the same three inner person-disjoint camera-stratified folds. This repeated exploratory use and camera/person confounding limit inference.

Fit-only unweighted FP64 means/standard deviations exported FP32, standard-deviation floor1e-6. Sampling uses inherited equal-person/site/image weights. Every variant receives the exact matching inner/full NP warm parent for seed17/29/43, with the original upstream lineage and costs retained. NP is a frozen643-parameter color36→16ReLU→Lab3 anchor. The new trainable residual starts exactly zero. This differs from historical R's fit-only ridge anchor, so old R errors are context, not matched controls.

## Architectures

All dimensions and parameter counts are generated from the explicit specs in chromaseed_architecture_scale.py. Counts below include the frozen643 anchor parameters; FP32 normalizers are additional storage.

| Variant | Trainable mechanism | Deployed parameters |
| --- | --- | ---: |
| patch_small | R token18→32→24 SiLU, token mean plus color36, MLP60→96→64→48→3 | 17,374 |
| patch5m | Token18→384→256 SiLU, mean plus color36, MLP292→1792→1536→1024→3 | 4,962,566 |
| soft_small | Original R dimensions E24/state48, four shared refinement passes | 15,246 |
| soft5m | Token18→384→256, state1120, four shared refinement passes | 4,846,822 |
| dynamic_small | Same small recurrence with dynamic attention mask | 15,246 |
| dynamic5m | Same large recurrence with dynamic attention mask | 4,846,822 |
| pool5m | ReLU token18→512; mean/std/max plus color36; 1572→3072ReLU→3 | 4,851,846 |

Patch/pool output adds tanh residual to the frozen normalized NP answer. Recurrence: context from pooled tokens/color; key projection has no redundant bias; query depends on current state and prediction. Dot-product scores divided by sqrt(E). Soft uses all64 tokens; dynamic includes positive scores and at least top4, with straight-through sigmoid training gradient and .001 mean-sigmoid penalty. Both use state=.5state+.5tanh(update), shared two-layer update and shared head, four increments .25*tanh(head). Full four passes always execute; there is no adaptive exit selection. Dynamic token encoding and scoring remain dense, so active connections are not a sparse speed claim. Small/large members of each R-style family differ only in widths; the shared anchor/training/evaluation rules match.

Layers are independently initialized from CPU torch generators seeded by seed+CRC32(layer name), uniform ±1/sqrt(fan-in), output head zero. Paired rate slots have identical initialization. The conventional pool control uses a residual warm start, so it is not an exact function-preserving enlargement of WIDE's learned hidden head. No architecture novelty is claimed.

## Training and fixed search

Fixed six-slot bank: seeds17/29/43, each with rates1e-5 and1e-4, in that order. Batch64, replacement weighted draws, independent NumPy generators seed+830003. AdamW decay.01, per-slot gradient-norm clip5, FP32 parameters/optimizer/loss/network, TF32 off, deterministic algorithms. Deployment and candidate inference are FP32 CUDA or independent FP64 NumPy, without autocast. Sum per-slot losses; each slot uses .6 mean over pass MSE + .4 final-pass MSE; one-pass models reduce to MSE. Target normalization is fit-only. Cosine factor .1+.9*.5*(1+cos(pi*(step-1)/8192)); horizon8192 remains fixed even when stopping earlier. CUDA graph warmups reset all initial parameters and optimizer moments/counter/base rates before the first real update.

Hardware-only comparison before source freeze and any production fitting: synthetic FP32 preflight took5.48/14.62/14.63/3.77 seconds for64 steps of patch/soft/dynamic/pool six-slot banks, with1.22–1.24GB peak allocations. BF16 autocast with FP32 master parameters/optimizer/loss took5.59/14.77/14.78/3.80 seconds and did not accelerate this bank implementation. These are single synthetic hardware probes, not general precision benchmarks or quality evidence. Preserve both exact sources and receipts as preflight_fp32_source.py/preflight_fp32.json and preflight_bf16_source.py/preflight_bf16.json. Restore the tested FP32 source and receipt for the primary experiment. Both passed the four numerical tests. No production labels entered this decision.

First screen trains every inner bank through2048 and saves128/512/2048. This is intentionally bounded, not a claim that longer fitting was optimized. All parameters, all seeds and both slots remain present in full selected-bank replays. Never divide bank timing by the number of models to invent single-model training cost.

Seven variants × three roles × three folds =63 inner banks /378 trajectories /1134 positive checkpoint payloads /214200 OOF prediction vectors. Candidate predictions use the exported single model in CUDA FP32 with native output rescaling in FP64; independent NumPy FP64 arithmetic must agree within absolute .002 native Lab, rtol1e-6. No tolerance adjustment based on observed failures. Dynamic near-tie disagreements, if encountered, are recorded as failures requiring a separate diagnostic, not silent acceptance.

Six positive settings per variant plus frozen NP and WE overall control =44 candidates/role,132 scores. Rank by mean equal-person DeltaE00 across three separate seeds, then p90, numeric bytes, steps, learning rate. Three-seed scores are not ensemble predictions. Select one setting per variant and one overall policy per role:24 decisions. Freeze selections before any final fit or any held-role prediction. Preserve all curves. Controls import their already verified inner decisions and outputs; their search budget differs from AS. An overall selected control is a legitimate negative outcome.

Fit all21 chosen variant/role six-slot banks from the same matching warm parents. Only the three chosen rate slots are deployed. Expected84 total research banks /504 trajectories, with final duration determined by frozen inner choices. Record63 new final models plus18 imported NP/WE controls =81 records. Store all four pass outputs for recurrence to inspect whether additional passes help; no outer-selected exit is allowed.

## Verification, resources and interpretation

Before freeze: synthetic four-family64-step CUDA preflight, exact large dynamic8-step replay and4-step prefix tests, initial NP identity for every architecture, token sensitivity, isolated slot gradients, actual dynamic gating gradient and exported CPU/CUDA parity. Preserve the preflight receipt. No production labels enter hardware/configuration decisions. Require at least35GB disk free before starting and15GB headroom before each bank. Large checkpoint archives are uncompressed to avoid CPU compression bottlenecks; no existing files are deleted.

After primary terminal success: independently reconstruct person/fold/normalizer/source bindings and all candidate metrics/decisions. Verify every exported checkpoint's predictions with independent NumPy FP64 arithmetic; recompute final metrics and actual one-example consumer outputs. Replay complete selected six-slot training at least once for every variant/role, compare all slots bitwise, report construction cost including original warm lineage separately from pure continuation. Time actual single-example CPU consumers after warmup, including token encoding; image extraction and phone runtime excluded. Replays and timing run without another GPU fit. Negative results and per-pass regressions stay in the report.

No external publication, messages, delegation or new assets. The user has already authorized local experiments. This evidence is instrumental skin-color regression on existing acquisition data, not independent ordinary-phone facial validation, cosmetic shade-match accuracy, recognition of identity, patentability or Skolkovo qualification.
