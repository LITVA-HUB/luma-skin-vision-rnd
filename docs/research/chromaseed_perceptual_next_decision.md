# ChromaSeed-P: measured decision and next question

2026-09-13. **Progress, broader goal active.** Previous KF/KE progress was verified from its source/artifact receipt and terminal process state before this study. This P-series has completed fitting, independent audit and standalone runtime measurements; no P job remains. It does not establish ordinary-phone facial accuracy or full goal completion.

## What changed

The compact128-landmark readout now supports shared/per-target perceptual color weights and iterative robust correction during training, while retaining the seven-array20 284B one-pass predictor. All2916 registered readouts were fit in12 shared banks, with8859 coupled linear solves and40.04s main wall time. All324 normalized-ridge controls exactly match KF.15 numerical tests pass, and the separate analytic-tensor/SVD audit covers45 selected models and18 positive-step probes in addition to all stored inner predictions.

Shared weights produce5.3901 /8.6548 /8.3869 versus normalized5.4387 /8.5970 /8.7050 across mixed/forward/reverse roles. Local weights produce5.3842 /8.7874 /8.4135. This is a limited directional gain, not robust superiority. Older stronger per-role references are retained. Shared-weight standalone mixed fit22.58ms, prepared-feature inference16.47µs, same20KB numeric payload.

Both iterative families choose0 steps in every role. Even1 correction is worse in the current inner selection grid, and4/16 do not rescue it. All648 training-objective traces are monotone; an independent analytic/SVD positive-step replay agrees within1.63e-5Lab. This supports treating the observed failure as a result of the tested training design, not an identified arithmetic bug. It does not rule out other correction methods. These are training corrections, not new dynamic inference connections.

## Next registered question, not launched

Every one of the15 primary choices selects alpha0.1, the smallest tested penalty, and width1. The robust objective downweights large residuals while the ridge term remains fixed, so equal numeric alpha is not equal effective regularization across objectives. This is a reason to examine the penalty range, not evidence that a lower value will win.

Before any new fit, write a separate protocol and new sources. Keep rank128, include the exact current alpha0.1 controls, test an expanded logarithmic alpha grid below0.1 for normalized and perceptual readouts, preserve the0-step option and the existing person-disjoint selection/evaluation order. Reuse algebra where appropriate and measure selected full-fit cost. Keep the search budget explicit and report all methods/roles. Do not modify this primary lock or reinterpret the already exposed outer table as fresh validation.

The full objective remains a compact, quickly trained, quickly responding model with high real quality. Current data still comprise only historically used original TRAIN with24 people; no new ordinary-phone facial validation is available. Useful local work remains, so the goal is not blocked. A stronger local result would still require new appropriate confirmation before a deployment-quality claim.

## Resume facts

Run `experiments/runs/chromaseed_perceptual_v1`. Training unified exec session37639/PID39176 completed exit0; audit session50020 completed exit0; profiling returned exit0 directly. Source SHA256 `9ce75fddcfc3241b271318da51eead680d78a9e51877b346aed64716b5550e92`. Selection SHA256 `7f526a4f17e2afb53f4a81bd436b958bac58d703ee368afa3bdf76a9b95efcfb`. Preserve bound core/train/reference/tests/protocol and all earlier KE/KF/R/palette locks. No publication, delegation, new datasets or old held-out partitions were used.

[Report](../benchmarks/chromaseed_perceptual_v1/report.md) · [Audit](../benchmarks/chromaseed_perceptual_v1/audit.json) · [Verification](../benchmarks/chromaseed_perceptual_v1/verification.json) · [Model card](../architecture/chromaseed_perceptual_model_card.md) · [Goal ledger](chromaseed_active_goal.md).
