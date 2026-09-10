# V5 and phone-validation engineering evidence, 2026-09-11

This interim addendum records engineering and development evidence, not a new
claim of inventive superiority or filing readiness. Broader positioning remains
compact adaptive color normalization and reliability estimation for color-sensitive
visual analysis, with facial skin/cosmetics as the intended first application.

## Implemented and verified

- Paired-state, deterministic comparison of six3.097M-parameter correction
  estimators/critics. Two seeds completed; third in progress. All24 saved
  best/final results of the completed seeds independently rescored from real
  SimpleCube++ illumination references and replayed from weights on CPU.
- A matched-query control isolates sequential recentering from the number of
  correction candidates evaluated. At51 queries the adaptive and fixed searches
  are identical for the ten completed trained critics;103-query differences
  are small and mixed. This is negative evidence for claiming a recurrence
  advantage in the current architecture.
- Licensed phone-data acquisition, provenance and loader/reference pipeline
  for Samsung S21 Plus/Oppo Find X5 Pro. Six official TRAIN captures passed
  development preparation;44 paired test scenes are reserved. Correct handling
  of already demosaiced, quantized camera RGB and reference saturation is
  documented. The code does not substitute phone camera AWB for reference GT.

## Measured on public real data

V5 results concern119 reused source-development images, not an independent
benchmark. No consistent large improvement from the new selected-action or
gradient-supervision mechanism is established. The completed V2 camera-transfer
milestone remains the strongest positive component evidence, including its
source/Canon regressions and competitive cheap controls. It is preserved intact.

Phone classical numbers presently concern only six loader-development captures;
they validate the evaluation pipeline and must not populate final phone efficacy
claims. No held-out phone result is asserted. A small iPhone SE2/XS Max CC0
object-photo archive is acquired, but appropriate absolute-color GT has not
been verified for it. Its auxiliary role is explicitly separated.

## Still unvalidated

Universal camera/scene/processing accuracy; ordinary smartphone JPEG/HEIC skin
colorimetry; clinically or instrument-validated facial color; physical surface
DeltaE accuracy; production cosmetic-recommendation outcomes; patent novelty
and formal legal classification. Optional DINOv2 teacher acquisition is not an
accuracy result, and teacher distillation itself has prior art.

Evidence: [execution status](../research/cc_v5_execution_status.md),
[two-seed measured table](../benchmarks/cc_v5/two_seed_screen/report.md),
[CPU replay](../benchmarks/cc_v5/two_seed_cpu_replay.json),
[equal-query ablation](../benchmarks/cc_v5/two_seed_search_control/report.md),
[phone loader findings](../data/phone_loader_findings.md),
[original data rights](../data/smartphone_benchmark_plan.md),
[V2 component result](cc_v2_evidence_addendum.md).
