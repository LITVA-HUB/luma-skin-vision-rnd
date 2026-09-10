# V5 paired correction-critic training

Predeclared 2026-09-11, before V5 training. Continues the V4 next decision;
this is a source development experiment, not an independent benchmark.

Question: does grading the model's own selected corrections improve its actual
correction policy, compared with ordinary random-action supervision? Does
supervising the derivative of the physical cost help beyond supervising values?
No novelty follows from these established training concepts alone.

Use frozen V4 CorrectionEvidenceNet and the same SimpleCube++ linear source
cache, 1126 training and 119 repeatedly used validation images. No independent
test, calibration, residual-fitting or new INTEL image/GT rows enter this work.
Original SimpleCube++ CC BY 4.0 status remains unchanged. No external weights.

For each seed (17, 29, 43), train exactly one 20-epoch point-only warmup. Copy
all model tensors, BN buffers, AdamW and scheduler states to six arms:

| Arm | Router | Sample center | Additional gradient loss |
|---|---|---|---|
| point | posterior, diagnostic only | point | none; point loss only |
| posterior_random | fixed posterior | point | none |
| action_random | generic action-conditioned | point | none |
| transport_random | nonlinear corrected-color router | point | none |
| transport_policy | nonlinear corrected-color router | detached selected step2 | none |
| transport_gradient | nonlinear corrected-color router | detached selected step2 | 25 times sin-squared gradient MSE |

All arms continue through epoch120: batch32, AdamW lr.001/wd.0001,
cosine floor.00002 over120 epochs, clip5, FP32, no AMP/TF32. V4 point loss
and field ramp epochs21-40 are unchanged. Field loss is .05 angular-degree
MSE +25 sin-squared MSE. The gradient arm additionally compares the actual
scalar field's derivative with the analytic GT cost derivative; targets detach.
No GT-centered actions and no pseudo-GT. Sample33 actions: original point,
16 within +/-.4 of the declared center, 16 global U[-1.5,1.5], clamp[-2,2].
Every field arm executes the same detached two-stage selection for matched
proposal compute, even when its result is unused. Gradient backward costs more
and must be disclosed; point-only has less training compute.

Image order, horizontal flips, common exposure and action noise have independent
per-epoch deterministic generators. All arms receive identical image/augmentation
and base-noise sequences. No diagonal channel augmentation, EMA or external
teacher in this isolation screen; these remain separate subsequent experiments.
Enable strict deterministic PyTorch algorithms, CUBLAS workspace4096:8,
disable cuDNN benchmarking. Replay epoch1 twice from identical complete state;
require bitwise model/optimizer/scheduler equality before continuing. No silent
fallback to nondeterministic kernels. Initial optimizer states must be deep-copied.

Evaluate every epoch. Preserve both best epoch and final epoch measurements;
best is selected by mean step2 reproduction (point readout for point-only).
Archive predictions including stage1/2/4, recovery/reproduction summaries and
raw risk curves at100/95/90/80/70/60. Point-only has no trained reliability head;
its diagnostic risk must not be reported as a trained selective baseline.
The same119 validation images select checkpoints and assess development: no
significance/generalization claim from their reuse. Three seeds vary training,
not independent datasets. No universal, skin-DeltaE or SOTA claim.

Implementation sequence: test sampling/state restoration/physical gradient;
implement a separate V5 runner preserving all V4 executable bytes; strict CUDA
preflight and tiny integration run; run six arms at seed17 then seeds29/43;
independently recompute prediction metrics and compare both best/final outcomes.
Keep all failures. If selected-action training fails, do not add more passes as
a substitute for better evidence. Stronger scene context/teacher training and
FFCC reproduction remain open. Equal-query nonadaptive evaluation is required
before crediting sequential refinement itself. Freeze policy before separate
error-head fitting/calibration and the new group-audited camera experiment.
