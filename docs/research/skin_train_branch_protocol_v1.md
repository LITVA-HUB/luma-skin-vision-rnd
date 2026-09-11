# Follow-up: fitting-only spatial branches, native skin color

PREDECLARED FOLLOW-UP after the spatial source screen, not independent discovery
or a new held-out confirmation. Previous post-hoc step0 inference improved both
source camera-transfer directions while harming mixed-camera validation. The
same source endpoints have been observed; do not label them a fresh test.

36fits: graph_always/conv_always/graph_drop/conv_drop x seeds17/29/43 x mixed/
from_SLR/from_ipod. All methods ALWAYS deploy with zero branch steps and choose
epochs by zero-step same-camera source validation. Graph/conv_always use their
three-step branches on every fitting batch; drop variants independently enable
the branch with probability.5per fitting batch, without inverse-probability
rescaling. All fitting pixels, targets,80epochs, optimizer, site sampling and
initial parameter tensors are identical to the prior spatial experiment.
Conditional random draws use the seeded CPUtorch RNG; no drops during evaluation.
The always-arm final weights must reproduce the corresponding prior spatial
final weights exactly, proving that only checkpoint selection/inference changed.

Target: original instrument nativeLab, standardizedMSE with fitting-only scaling.
No spectral pseudo-labels, paired-output loss, camera ID, extra images, pretrained
model or test-time adaptation. The plain inference core has924,932active parameters;
unused training modules remain in checkpoint storage and are disclosed. Compare
against BOTH current plain and ordinary conv controls and prior capture-plainMSE
and conditional-mixture results under the same source protocol. A benefit over
a weaker baseline alone is insufficient. Preserve every negative result.

This tests whether the extra branch acts as a useful fitting constraint rather
than necessary inference computation. Random branch dropping has stochastic-
depth prior art and is not a novelty claim. Difference from conventional
stochastic depth here: zero branch at all deployment/selection endpoints, with
the entire source protocol fixed before this follow-up. No new independent
test evidence is available, even if the follow-up yields stronger source scores.

Fit-boundary/checkpoint/scalarDeltaE00/risk-coverage audits mirror the spatial
screen. Dispersion-based risk remains uncalibrated. Keep MSKCC TEST/CAL unopened
for this experiment and preserve existing independent results unchanged.
