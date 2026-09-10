# V5 preflight, 2026-09-11

Status: implementation and integration verified; primary accuracy NOT YET MEASURED.
Protocol: [paired critic experiment](../research/cc_v5_protocol.md).

- Five added behavioral tests cover detached paired proposals, complete Adam/BN
  restoration without shared-state mutation, independent finite-difference
  physical derivatives, strict CUDA field/second-derivative backward, and the
  identity-pool change's output/gradient equivalence. CPU FP64 gradient tolerance
  is absolute1e-12/relative1e-10; the autograd accumulation order can differ.
- Complete repository suite:190 passed,14 existing warnings,33.38s. Ruff passed.
- First two-warmup/three-total-epoch integration run, `ccv5_pilot_s17`, failed
  at adaptive_avg_pool2d_backward_cuda after the point control. Its failure and
  source receipt are preserved. Point-only determinism alone was insufficient
  to verify the full field-training path.
- Revised `ccv5_pilot2_s17` completed all six arms. The first warmup epoch
  replayed with bitwise-identical full state and equal loss. All arms restored
  and preserved the same starting state. Full field and gradient training had
  finite outputs. This tiny pilot is pipeline evidence only; its approximately
 10.95-degree development errors do not rank scientific methods.
- V4 files and historical negative/positive milestones remain unchanged.

No V5 inference benchmark/export, independent test, camera-held-out evaluation,
calibrated coverage guarantee, physical surface DeltaE or facial validation exists.
Primary six-arm experiments at seeds17/29/43 are the next execution step.
