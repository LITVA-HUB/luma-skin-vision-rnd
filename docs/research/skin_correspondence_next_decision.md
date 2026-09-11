# Decision: shared bias diagnostic and inverted consistency experiment

The endpoint is actual native skin Lab / CIEDE2000. This cycle produced a
source-only correspondence audit of 314 sites and 27 frozen prediction sets,
then 36 predeclared fits on real instrument-referenced photographs. No exposed
MSKCC TEST/CAL or UMINHO held-out endpoint was accessed. Source validation has
been repeatedly explored; no new independent accuracy result is claimed.

## What the diagnostic establishes

Triplicate pairwise discrepancy averages 2.5757 TRAIN / 2.6139 VALIDATION
DeltaE00. Each reading differs from its own three-reading mean by 1.5195 /
1.5354 on average. These quantities cannot be subtracted from model error or
called an irreducible accuracy floor. References are exactly repeated across
the views of a site, not independent new measurements for each photograph.

For the strong mixture model, shared site bias accounts for 71.65% of squared
native-Lab error on mixed-camera validation, 81.59% for SLR-to-iPod and 58.64%
for iPod-to-SLR. This is an algebraic decomposition of Euclidean squared Lab,
not of DeltaE00 and not causal attribution to cameras or measurement noise.
Even averaging the site's multiple predictions leaves errors 2.8719 / 4.6598 /
4.1870 DeltaE00. Multiple views are a diagnostic, not an allowed inference claim.

## Inverted assumption and falsification

We inverted the assumption that more paired-image agreement is always useful.
For a genuine pair, individual squared error I = shared-mean error G +
disagreement D. Identical models/batches/budgets compare G+2D (consistency),
G+D (ordinary), G+0.5D (shared-half), and G (shared-only). The important failure
mode is opposite errors canceling under G while each single-image answer is
wrong. All actual model results below use ONE image.

| Mean over three separate seed scores | Mixed | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Ordinary mixture control | 3.4406 | 5.0193 | 5.9635 |
| Strong ordinary plain control | 3.4771 | 5.8301 | 5.5089 |
| Increased consistency | 3.4178 | 5.3455 | 5.9443 |
| Shared-half | 3.5246 | 5.2770 | 5.8533 |
| Shared-only | 3.7398 | 6.1655 | 6.3318 |

Increased consistency improves mixed mean in all three seeds by only 0.0228
on average. The descriptive six-person bootstrap interval is [-0.0650, 0.0254],
crossing zero. It worsens p95 (6.9574 vs 6.9316) and 80% error (3.4848 vs
3.4407), and worsens the SLR-to-iPod mean. It is not a selective or universal
victory. Shared-half helps against its own weak reverse-direction control but
loses against strong plain in every seed there. Shared-only loses all protocols.

Reject the proposed common-bias reweighting as an improvement in this screen.
Do not combine it with prior one-direction wins or launch a finer coefficient
search on the same six validation people. Pair consistency alone is not the
missing mechanism. All 36 fits and all unfavorable strata remain archived.

## Integrity and limits

Same CaptureColor mixture architecture, 929,297 stored parameters and 80-epoch
budget. No camera/reference/mode inputs at inference. Source model audit:
108 exact color arrays, 36 gate/hypothesis replays, 6,336 independent scalar
DeltaE00 checks, 216 fixed-coverage checks, and nine bitwise-identical historical
control final weight sets. Diagnostic: 30,150 independent scalar checks and
reference/decomposition identities. Hypothesis dispersion is uncalibrated.
No deployment optimization or new latency claim. Peak allocated CUDA memory
for fit plus selection is 108.36 MiB, excluding CPU caches and CUDA reservation.

## Next different representation, not another consistency coefficient

Before distilling another model, cheaply test whether the already acquired
licensed frozen DINOv2 teacher contains complementary source skin information.
Compare matched-budget supervised readouts of ordinary absolute-color features,
teacher features, and both together; also include a shuffled-feature control
if a gain appears. Fit/readout selection must use only source roles and same-
camera selection in transfer screens. Preserve native Lab targets and every
strong existing compact baseline. Teacher-only invariance cannot substitute
for absolute lightness or chromaticity.

Mechanism: representation learned from diverse images may describe surface,
texture and capture artifacts absent from hand-engineered token summaries.
Assumption: that representation retains useful residual information about
instrument color. Failure mode: it mostly identifies capture or texture and
discards color, or improves familiar people/cameras only. Cheapest falsifier:
frozen-feature source readout before any new teacher/student training sweep.
This is PLANNED, not extracted or fitted in this cycle. A teacher readout would
be a research diagnostic exceeding the deployment size, not a product model.
Use the existing original license/provenance; pretraining overlap is unknown.

The previous independent skin test remains 4.4570 full / 4.1591 at 80% for the
primary system, and 4.3005 / 4.1447 for the stronger ordinary fusion. Proposed
has no convincing superiority. Ordinary facial iPhone/Android accuracy and
the working median<=2 / p95<=5 at>=80% product aspiration remain unvalidated.

[Correspondence](../benchmarks/skin_correspondence_v1/report.md),
[36-fit results](../benchmarks/skin_shared_bias_v1/report.md),
[fresh source-method check](skin_correspondence_sources_2026_09_11.md),
[existing teacher rights/provenance](../ip/dinov2_teacher_adoption.md).
