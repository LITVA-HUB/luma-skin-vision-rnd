# ChromaSeed-GS: frozen-model color and gate stability

Registered 2026-09-13 before executing the diagnostic. Previous G goal turn is progress: model/metric/runtime/artifact bindings and terminal process state were rechecked. The full compact/fast/high-quality goal remains active. No delegation, publication or data acquisition.

## Frozen scope

Use original TRAIN only, SHA256 d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0; load color/target/patient/site/device arrays. Use G's72 selected models and their original query indices/predictions across three historically reused roles, eight families and seeds17/29/43. Bind G source/selection/results/verification and all144 model/prediction files before computing outputs. No fits, selection, threshold changes, legacy validation/calibration/test, images or tokens. These are synthetic sensitivity measurements, not new photographed subjects.

## Legal affine color transformations

For each encoded-RGB pixel v in [0,1]^3, define v'=(1-t)v+t*a with anchor a one of the eight RGB cube corners, in lexicographic order. Positive t values are exactly1/255,4/255,16/255,64/255. Identity occurs once:33 transformations total. The strongest dose is a stress condition; none of these doses is asserted to be a measured camera/noise distribution. We do not tune doses by outcomes.

This bounded contraction leaves every pixel legal without clipping. Quantiles and means transform by the same affine map, population standard deviations multiply by(1-t), and correlations stay unchanged because t<1. Apply in FP64 and store transformed features FP32 before the existing predictor's normalization. Finite legal features are required; no pixel clipping is approximated from quantiles. Constant channels retain their original zero correlations.

For every model/transform record prediction drift ΔE00 from its original answer and error ΔE00 against the original instrument target, plus image p90/max and equal-person means. The instrument target is unchanged under this synthetic digital corruption; scores do not imply newly verified target/image pairs. For each registered positive dose, take the rowwise worst drift and worst error across its eight anchors, then average within people and across people; also record camera subgroup person means, even though camera and participants are confounded. Seed summaries average errors/statistics, not predictions or independent people. Store every transform; no post-hoc best perturbation/family selection.

For active gates record original/perturbed scores and sign flips (also for soft gates, whose sign change is continuous). For inactive single-camera routed fallbacks, gate metrics are absent, not zero evidence of unknown-camera robustness. Preserve baseline/uniform/soft/hard controls for both losses in all roles.

## Boundary diagnostics

For each mixed hard model (six models, both losses ×3seeds), compare its matched soft model, shared base and uniform model at identical boundary inputs. Compute the gate score's affine slope along each of eight legal anchor directions, using baseline FP32-normalized coordinates and the exact FP64 feature-direction/sigma slope. Approximate root t=-score/slope is probed only when1e-4<t<64/255-1e-4. Evaluate actual predictors at t±1e-4 and report whether the hard gate really flips after FP32 rounding. Both transformations are legal pixel contractions, and their maximum possible per-pixel encoded-RGB difference is2e-4; this is a continuous-valued synthetic construction, not an8-bit/JPEG pipeline. Probe pairs may repeat people/rows/directions and are not independent samples. Report counts, ΔE00 jump distributions and all four matched output controls. No target labels guide these roots.

Separately project each mixed query to the nearest gate hyperplane in normalized36-feature Euclidean coordinates. Record RMS distance and the analytic hard-gate left/right output limits at that point. Report simple feature bounds/quantile-order/variance checks as necessary conditions only: this unconstrained projection need not correspond to a realizable image. Do not call the distance a certified pixel robustness radius. Soft has one continuous limit there; no claim that the hard jump occurs at a naturally observed input. Do not fit from these boundary probes.

## Integrity and independent checks

Tests before real data: bounded affine feature updates versus explicitly transformed synthetic pixels including constant channels, identity, invalid strengths/anchors, analytic affine gate crossing and nearest-hyperplane geometry, and a known hard-gate jump versus actual prediction. Freeze core/runner/tests/protocol and NumPy consumer plus inherited G sources before real execution.

Then independently reconstruct every transformed selected query with the previously verified standalone NumPy batch-one consumer and separate feature algebra; check all original controls, dose-wise worst/person/camera metrics, sign flips, boundary probe coordinates/outputs and unconstrained distances/limit jumps with direct kernels. Require native-Lab prediction drift<=2e-8 for ordinary transformations and<=0.001 for independently reconstructed boundary probes; report actual maxima and any rounding-related gate disagreement explicitly. The audit shares the verified CIEDE2000 formula and original row indexing; no unsupported claim of independent real-world validation.

No production runtime benchmark is needed because no inference algorithm or weights change. Record diagnostic wall time separately. Preserve every G and earlier lock and derived artifact. Report all doses and matched controls, with any synthetic robustness limitation, before deciding whether a later training remedy or broader person-split evaluation is justified.
