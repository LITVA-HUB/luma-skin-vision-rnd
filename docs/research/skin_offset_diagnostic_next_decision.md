# Decision: a shared color offset is not sufficient

The privileged-reference diagnostic is complete on90 existing source prediction
arrays,324 excluded-person folds and180 corrected arrays. It uses other evaluation
people's actual instrument references and is NOT a calibration-free model result.
No new model, original prediction, independent test or deployment score changed.
[Full diagnostic report](../benchmarks/skin_offset_diagnostic_v1/report.md).

## Measured and interpreted separately

Image-uniform model SLR-to-iPod mean DeltaE00 worsens5.4877 to6.2298 at full
excluded-person offset; the combination worsens6.1545 to7.4325. In reverse,
image6.5602 becomes6.0229 and combination6.4074 becomes5.4434. These reference-aided
figures must not be advertised as achieved single-image accuracy. Both fixed
strengths0.5 and1.0, every model and original predictions are retained.

The shared-offset hypothesis is inadequate across the source directions. The
residual varies across people and images, and a correction inferred from other
people can over-correct the held person. This does not identify whether biology,
capture, camera, color support or reference variability causes the differences.
The small known/unseen source populations remain observational and reused.

A separate Euclidean-Lab identity quantifies why shared corrections can harm.
For combined SLR-to-iPod, shared residual energy is6.258 versus between-person
energy24.201; full correction increases hierarchical squared-Lab error23.993.
For reverse, shared19.047 versus between-person2.084 yields change-16.441.
These are squared Euclidean native-Lab quantities, NOT CIEDE2000, a new accuracy
target, noise floor, or an analytic explanation of every DeltaE00 change.
All180 identities match direct calculation within4.27e-14. The actual color
accuracy findings above are independently scalar-verified CIEDE2000.

## Next mechanism: whether updates transfer across people

Do not build another global offset head from this evidence. Examine whether
training updates that improve one person's examples damage other people's
examples. Use original TRAIN only and frozen source models, no VALIDATION
gradients or references. Start with image-uniform and combined seed17 models
for mixed/SLR/iPod TRAIN protocols; keep these choices fixed before analysis.

Measure per-person gradients for color-only and original color+mode objectives.
Compare with shuffled group assignments preserving group sizes and the same
underlying examples. Verify selected one-step loss changes against gradient
inner-product predictions and restore exact checkpoint states after interventions.
Do not promote transient training-loss changes as skin accuracy improvements.

The key counterargument is mathematical: near a stationary pooled objective,
group gradients can oppose each other even when learning is correct. Therefore
negative gradient cosine alone is not evidence of camera shortcuts or a reason
to force invariance. Random-group controls and finite-update checks must precede
a meta-learning or gradient-agreement training experiment. Color-range differences
between people are legitimate signals and must not be erased merely to improve
an agreement statistic.

Possible advantage: TRAIN-only updates that transfer across people could reduce
dependence on person/acquisition-specific correlations without extra inference
inputs. Likely failure: suppressing useful variation or worsening the ordinary
color objective. Cheapest falsifier: this gradient/update audit before new fits.
Methods such as MLDG, Fish and Fishr are established prior art, not our inventions;
use them as relevant baselines if the audit justifies pursuing this direction.

## Integrity and product boundaries

324 own-reference perturbation checks, explicit zero own-person matrix blocks,
unit row sums,42768 independent scalar color cases and1620 fixed-coverage rows.
361 tests pass,14 historical warnings in33.37s. No new model-size/latency claim.
Original MSKCC CC-BY; no new outside data/weights, proprietary collection, cloud
resources or publication. The source labels used by this diagnostic do not alter
the single-image/no-test-camera-calibration goal.

Independent MSKCC remains primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447.
No strongest independent-baseline win, universal phone/face accuracy, calibrated
refusal guarantee or demonstrated novel technology. Goal remains active and unmet.
