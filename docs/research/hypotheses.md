# Pre-registered research questions, hypotheses, and gates

Status: **PLANNED. No gate has passed; real results are NOT MEASURED.** Freeze thresholds and analysis choices before opening test results.

## Research questions and falsifiable hypotheses

| ID | Question and hypothesis | Required comparison/evidence |
|---|---|---|
| RQ1 | Naïve facial color is materially unstable across cameras and lighting. | A0 within-subject ΔE00 and variance components across the capture matrix. |
| RQ2 | Deterministic color methods remove part, but not all, of that error. | Paired A0 vs A1/A2 errors by subject, camera, and light. |
| RQ3 | Compact learned baseline C outperforms the best deterministic method. | Subject-paired test ΔE00 distributions and full coverage curve at 100% coverage. |
| RQ4 | Measurement-aware ROI selection improves error beyond matched model C+. | Proposed vs C+ with identical backbone/training budget; ROI ablations. |
| RQ5 | Photometric ambiguity features predict held-out real ΔE00. | Out-of-fold residual prediction, rank association, calibration, and feature ablations. |
| RQ6 | The dedicated error model ranks/calibrates failure better than conventional confidence scores. | Compare predicted error/probability against detector confidence, generic quality, regression magnitude, ensemble/standard uncertainty baselines. |
| RQ7 | At equal coverage, the proposed selector reduces accepted-set ΔE00. | Full paired risk–coverage curves plus pre-registered 80% coverage primary comparison. |
| RQ8 | Benefits persist on locked devices and/or lights unseen during fitting. | Separate untouched device-held-out and light-held-out reports with no threshold refitting. |
| RQ9 | Useful behavior transfers to a production candidate under approximately 10M parameters. | Teacher/student or direct compact model comparison, size, latency, memory, and equal-coverage error. |
| RQ10 | ONNX/FP16/INT8 preserve acceptable predictions and selection. | Paired numerical equivalence, ΔE00/calibration/coverage deltas, latency and memory; INT8 only if justified. |

## Stage gates

- **G1 — Target repeatability:** real repeated cheek measurements meet the pre-registered repeatability tolerance. Failure blocks training on real targets and triggers protocol repair.
- **G2 — Real problem:** A0 shows enough cross-camera/light error to justify correction while the target remains repeatable.
- **G3 — Learned value:** C beats the best validated classical baseline without leakage.
- **G4 — Proposed value:** Proposed beats strong C+ in the primary matched comparison and survives mechanism ablations.
- **G5 — Selective value:** Proposed lowers risk at equal coverage, primarily at 80%, with useful behavior across the full coverage range and locked domains.
- **G6 — Deployment:** a compact exported model preserves the accepted scientific behavior within pre-registered tolerances on target hardware.

Each gate needs an explicit numeric criterion set before its evaluation data are inspected. A failed gate is a result, not a reason to change the criterion after the fact.

## Primary estimand and uncertainty

The primary comparison is Proposed versus C+ at 80% accepted-image coverage on the subject-held-out test set, using mean accepted-region ΔE00 and a paired difference. Report the complete risk–coverage curve as co-primary context so a single operating point cannot hide poor behavior. Thresholds are chosen on calibration data and applied unchanged to test data.

Use a subject-cluster paired bootstrap: resample subjects with replacement, retain all their sessions/images/cheeks, compute both methods on the same resample, and report the paired difference interval. Pre-register replicate count and interval method. Also report median, upper quantiles, coverage achieved, failure/unsupported counts, and per-device/per-light results. Do not treat cheek rows or image repeats as independent subjects.

## Sample size rule

The 10–15-person capture is a variance and workflow pilot, not a powered confirmatory study. Use its subject-level paired differences, repeatability, failure rate, and between-domain variance to simulate or analytically estimate the number of independent subjects needed for a predeclared minimum relevant ΔE00 improvement and desired interval precision. Preserve clustering and anticipated attrition/domain strata. Report assumptions and sensitivity ranges. Do not publish false power numbers before these variances exist.

