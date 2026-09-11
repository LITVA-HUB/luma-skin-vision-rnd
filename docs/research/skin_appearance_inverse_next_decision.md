# Decision after conditional appearance inversion and support controls

Actual skin Lab / DeltaE00 remains the endpoint. This CPU statistical falsifier
does not earn a new accuracy, novelty or universal phone claim. The 108 original
fits and 72 post-hoc support endpoint instances are preserved in the
[report](../benchmarks/skin_appearance_inverse_v1/report.md). All use the existing
real MSKCC source photographs and native instrument references, not synthetic
illuminants or an invented surface-color target. Source populations are reused.

## What the changed direction achieved

With all 18 mean patch statistics, the mixed-source nonlinear four-mode inverse
has mean 3.8663 versus direct quadratic regression 3.9046. Its descriptive
patient interval for the difference is [-0.2316, 0.1458], crossing zero. A direct
control constrained to the same TRAIN color hull scores 3.8947. The tiny mixed
gain is not strong evidence. The archived compact image baseline is better at
3.4406 and has access to all 64 patches, rather than their average statistics.

At common 80% coverage, the inverse scores 3.7307 versus direct 3.7645; its own
posterior-error ranking is worse at 3.8248. Posterior uncertainty is not shown
to improve selection over simple input novelty and is not calibrated C+.

The same stats/mode-degree2/posterior-mean family, with alpha selected on each
same-camera validation only, scores 5.5627 for SLR-to-iPod and 7.4273 for reverse
transfer. Historical stronger controls are 4.8328 and 4.9736 respectively. Other
inverse families sometimes beat these two new scores, but do not select them
using the held-out-camera errors or present that as an achieved universal model.

## Attack the favorable interpretation

The inverse's empirical color prior forbids extreme extrapolation. The initial
direct comparator did not have that constraint. Therefore a separate frozen
post-hoc screen projected every direct candidate onto the same TRAIN color hull
or nearest actual color atom, then reselected alpha on same-camera validation.

These constraints explain some, but not all, of the inverse/direct transfer gap.
For example stats/degree2 direct reverse error falls from 15.9210 to 13.7872 with
hull projection, while the global-degree2 inverse scores 6.9891. That comparison
is still between weak small-statistic models, not a victory over the stronger
image baseline. The inverse is not just the uninformative prior: constant TRAIN
site-mean color errors are 9.9678, 9.2982 and 11.8522 across the three protocols.

Nearest-true-target palette diagnostics have means 1.1043 / 1.8851 / 1.6807.
They use true reference color and are not image predictions. They show that the
finite color palette alone does not explain this inverse's large errors; they
do not prove that an arbitrary skin color is identifiable from one JPEG.
Covariance estimated from 2,880 person-excluded forward fits still does not
guarantee correct uncertainty under an unseen acquisition distribution.

## Next mechanisms and what must be falsified

Retain two independent paths:

1. Combine the earlier training-only graph regularization with observed-patch
   support under matched raw/plain controls. The prior support gain versus plain
   occurred in both transfer directions, but harmed mixed accuracy. Check spatial
   assumptions before composing the inputs. Do not allocate the whole search to
   this incremental combination.
2. Replace mean appearance with a distribution over observed patches. Before
   building a neural renderer, test whether a non-Gaussian conditional patch
   likelihood preserves useful information about skin color that averaging loses.
   The latent nuisance should not be defined solely by the known camera/mode.

There is a useful algebraic trap in the second path. For Gaussian patch noise
with covariance independent of color, average patch log likelihood equals the
log likelihood of the patch mean minus half the trace of precision times patch
scatter. With one component that scatter term is independent of candidate skin
color and cancels from the posterior. A product rather than average likelihood
changes effective temperature, not the sufficient statistic. With multiple
fixed covariances it can reweight modes, but still supplies no extra within-mode
color statistic. Merely feeding all patches into that same Gaussian model is
therefore not a new information source for color inference.

The distinct candidate needs non-Gaussian patch structure or justified
color-dependent distributions. Mechanism: distinguish material/color hypotheses
using the full distribution of observed regions while marginalizing nuisance.
Assumption: that distribution contains transferable information about site Lab.
Potential benefit: retain heterogeneity without requiring camera identity.
Likely failure: patch correlations produce false confidence, or a flexible
nuisance model explains every candidate color. Cheapest test: a small conditional
patch mixture with explicit mean/Gaussian and likelihood-temperature controls,
using only actual observed TRAIN patches. Native Lab labels describe the site,
not per-pixel measured colors. No new neural fit is claimed by this decision.

## Verification and claim boundary

108 exact closed-form/OOF refits; 360 exact prediction arrays; 31,680 independent
evaluation color cases; 1,944 coverage rows; 57,024 curve points. Independent
checks cover 3,096 normal equations, 357,120 Gaussian densities and 238,146 palette
costs, plus 98,736 prior/representation scalar cases. The support control adds
2,356,992 scalar nearest-atom distances, 25,344 color cases and 1,575 constrained
projection instances with KKT checks. These are computational cases, not new
people or independent samples. 343 tests pass, 14 historical warnings, 33.56 s.

All jobs terminal. No new external data, weights, cloud use, publication or
export. Original MSKCC CC-BY remains the data provenance. The exposed independent
TEST/CAL archive is unchanged: primary mean 4.4570 / 80% 4.1591; ordinary fusion
4.3005 / 4.1447. Ordinary phone facial color accuracy and product targets remain
unproved. Prior synthetic and real negatives remain intact. Goal active/unmet.
