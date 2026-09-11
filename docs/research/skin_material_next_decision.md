# Decision after 45 real-image material fits

Do not promote or export the nonlinear material decoder. Its physical prior
does not produce a universal improvement in instrument-referenced skin Lab.
All results are exploratory source cohorts that have been inspected repeatedly.
The previously exposed independent MSKCC test remains an evaluation archive.

Mean DeltaE00 across three separate seeds, not an ensemble:

| Method | Mixed | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Ordinary direct | 3.4406 | 5.0193 | 5.9635 |
| Matched tangent | 3.5112 | 4.9531 | 5.4509 |
| Nonlinear material | 3.5141 | 4.9083 | 6.1311 |
| Historical training-only graph | 3.6394 | 5.8339 | 4.9736 |

The small SLR-to-iPod material/tangent difference has a descriptive patient
bootstrap interval crossing zero. The reverse direction is worse. No tested
matched contrast improves all three seeds. Uncalibrated hypothesis dispersion
often makes the mean error worse at 80% coverage. This is not a calibrated
selective-system result. [All results](../benchmarks/skin_material_image_v1/report.md).

## Attack on the representation hypothesis

A separately recorded post-hoc TRAIN diagnostic supplied the true native Lab
to a bounded nonlinear solver. With eight coefficients allowed within +/-6
ISSA training standard deviations, all 248 TRAIN site references were represented
to DeltaE00 below 0.01 (mean achieved 1.96e-11). With bounds +/-1 and +/-3,
mean achieved errors were 3.7694 and 0.1059. This is known-target reconstruction,
not image prediction; it does not identify the true spectrum or prove the
validation gamut. Nonzero solver residuals do not prove global infeasibility.

Consequently, insufficient single-code TRAIN color range is not a necessary
explanation for the failed image model. Increasing the decoder range alone is
not the next experiment. [Diagnostic](../benchmarks/skin_material_image_v1/feasibility_train.json)
and [744-case scalar audit](../benchmarks/skin_material_image_v1/feasibility_audit.json).

## Next competing mechanism: conditional native-color distributions

Cancel the assumption that one regressed Lab point adequately represents an
ambiguous photograph. Compare an ordinary mean regressor, a single Gaussian,
and a compact mixture density head. Separately compare the same learned
distribution's mean with a decision minimizing its estimated expected DeltaE00.
The prediction and uncertainty refer to instrument skin color, not illumination.

Mechanism: multimodal appearance ambiguities might be retained until the final
color decision. Key assumption: this small real training cohort supports useful
conditional densities. Advantage: one distribution supplies color alternatives
and downstream error ranking. Likely failures: component collapse, overconfident
extrapolation, poorly estimated tails, and worse fitting than ordinary regression.
Cheapest falsifier: matched source fits and frozen inference-rule comparison;
only then consider subject-held-out error calibration and a fresh benchmark.

Mixture density networks and Bayes decisions are established prior art, not our
invention. A rival branch is a forward appearance model with feasible color sets;
its identifiability and camera assumptions must be tested before implementation.
No new foundation model, camera identity input, or test-time calibration is implied.

## Evidence and rights

All 135 color prediction arrays replay exactly; 7,920 independent scalar color
cases, 270 coverage rows and 7,920 curve points pass. The material feasibility
diagnostic independently replays 744 scalar cases. Existing full test run: 318
passed with 14 historical warnings; this is software verification, not accuracy.

ISSA original data is CC BY 4.0. Original CIE constants are CC BY-SA 4.0; adapted
buffers keep attribution/share-alike. A checkpoint containing them is not an
unrestricted proprietary artifact. No participant files or model weights are
published. Ordinary facial phone accuracy and the product target remain unmet.
