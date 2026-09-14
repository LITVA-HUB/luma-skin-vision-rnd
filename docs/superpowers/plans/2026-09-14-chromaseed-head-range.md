# ChromaSeed HR: matched residual output range

Current phase: CPU preparation verified while the unchanged AS primary uses RTX4060. Sixteen tests pass3.38s; receipt46bbefcf1155c85a8a2097409910cc95e35680c228684111ccee5b5f03a96b2d. The research objective is unfinished; the goal scheduler reported paused on2026-09-14 and was not changed. The already-running AS worker continues. Latest user priority is quality with approximately5million parameters.

The AS inner geometry diagnostic found27.15/34.37/26.61% subject-weighted targets outside the necessary +/-one-target-std residual box. Actual learned boundary use was much smaller. This motivates an experiment, not a causal claim or an accuracy gain. The architecture-scale experiment must complete under its original protocol.

- [x] Implement a separate head adapter for tanh(z),4*tanh(z/4), and z. Preserve unit slope at zero, all dimensions/parameters, four-pass mechanics, normalizers, warm starts and original optimizer. Unit mode calls the unchanged AS implementation directly.
- [x] Write CPU tests before implementation: initial-value/slope equality, exact unit-mode forward/gradients/exports, actual wider output support, and independent NumPy/Torch prediction parity. No CUDA context or production model fitting in this phase. Eleven adapter tests pass.
- [x] Adapt the fitter separately, retaining the original six-slot bank, sampling, schedule and optimizer; test CPU unit-mode trajectory identity against AS. Five fitter tests pass. The fused optimizer prototype stays separate.
- [x] Estimate storage for a full matched comparison before registering the GPU grid. Raw new theta arrays require29.562GB; lossless probes estimate24.629–27.364GB, with exact reconstructed bits but no guaranteed future ratio. D: has approximately353GB free. A new dedicated D:\Luma-RnD\chromaseed_head_range_v1 directory is linked from experiments/runs/chromaseed_head_range_v1 and a write/read check passed. Use uncompressed original arrays there; preserve all AS artifacts and all architecture comparisons.
- [ ] After AS primary/audit/runtime are terminal, run a CUDA preflight and freeze the final HR protocol/source grid before any original TRAIN fit. No simultaneous GPU work and no outcome claim from CPU tests.
- [ ] Complete inner selection, selected final fitting, independent verification and actual costs; compare all outcomes, retaining negative results. This phase is not yet launched.

The production HR runner, independent auditor and runtime report are not yet implemented. The numerical adapter/fitter and storage preparation alone do not constitute a launched experiment. See docs/research/chromaseed_head_range_readiness_2026-09-14.md.

Expected scientific contrast: widen the support without changing the initial derivative or adding parameters. Cap4 may still exclude extreme targets; linear removes this particular bound but may overfit or produce implausible outputs. A true WIDE enlargement with its linear head remains a relevant separate control. No original data/images/weights/packages/agents/publication added.
