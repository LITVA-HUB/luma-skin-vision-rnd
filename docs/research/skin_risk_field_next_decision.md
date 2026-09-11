# Decision after color/risk crossing and candidate-loss fields

Keep actual instrument skin DeltaE00 as the endpoint. No new method earns a
universal accuracy or deployment claim. The independent400-image MSKCC result
is unchanged. All experiments here use heavily inspected source development.

## Crossing useful components: partial, inconclusive

At80% mixed-source coverage, a fixed ordinary+Gaussian pair with covariance
ranking scores3.1954 versus3.3625 for an ordinary pair with dispersion ranking.
Active parameter counts are comparable:1,864,750 versus1,858,594. Its descriptive
patient-bootstrap interval for the difference is[-0.4692,0.1323], crossing zero.
The reverse transfer is worse:5.9368 versus5.6588. SLR-to-iPod mixed-pair color
mean5.1263 is worse than ordinary-pair4.8747. Do not promote the favorable mixed
coverage number as camera independence or as a calibrated C+ comparison.

Ninety frozen method/seed/protocol endpoints are preserved, with368,640scalar
integral cases,15,840scalar color cases,540coverage checks and complete curves.
[Crossing report](../benchmarks/skin_risk_cross_v1/report.md).

## Changing the target representation: no stronger result

36matched image fits compared ordinary regression, soft categorical labels,
simplex-constrained color-loss fields and affine loss fields without positivity.
The shared full Gram factor preserves uniform candidate-risk MSE, rather than
discarding spectral components. Derived loss targets use only actual TRAIN Lab.

| Primary method | Mixed | SLR to iPod | iPod to SLR |
|---|---:|---:|---:|
| Direct | 3.4406 | 5.0193 | 5.9635 |
| Soft CE | 3.7810 | 5.5461 | 7.4636 |
| Simplex field | 4.0426 | 5.6813 | 9.5897 |
| Affine field | 3.7598 | 4.9641 | 8.9582 |

Affine beats soft CE in all three SLR-source seeds, but not the strongest
ordinary controls universally. It uses negative atom weights on about20-27%
of coefficients; selected risks happened to remain nonnegative in this screen.
Removing probability constraints does not solve camera transfer.

The post-hoc grid control projects ordinary predictions without using true
references: mean3.5335/5.0652/5.9965, still better than most field outputs.
Known-target nearest-grid oracle errors0.7264/0.7121/0.6926 are representation
diagnostics, not image accuracy. Thus discretization contributes error but does
not explain the large reverse-transfer failure. This does not rule out every
continuous conditional-risk architecture.

All108color arrays replay;12,672scalar color cases,432coverage rows and12,672curve
points pass. Palette63,488scalar cost cases and full-Gram identities pass. Additional
2,112scalar grid-control cases pass. [Field report](../benchmarks/skin_loss_field_v1/report.md).

## Attack the current shared assumption

Changing only output uncertainty or output geometry has not yielded robust
transfer. A shared unverified assumption is that the existing source capture
variations teach the backbone a sufficiently broad appearance-to-color mapping.
The validation population is tiny; more searches on it do not create independent
evidence. Neither confident Gaussian output nor a rich candidate table adds
information to the input photograph.

Next independent mechanism to falsify: learn appearance variations from real
same-site captures and use them for training support expansion, keeping the
instrument reference fixed. Compare actual same-site interpolation against
ordinary training and a matched augmentation control. First audit the paired
capture geometry; do not assume registered pixels or label-preserving arbitrary
RGB transforms. Any derived augmentation is a training regularizer, not a new
independent photographed subject or a newly measured color reference.

Mechanism: actual paired capture differences might expose nuisance variation
without estimating the camera identity. Key assumption: within-site capture
variation covers useful deployment perturbations. Advantage: changes the input
learning problem instead of another output head. Likely failure: wrong pair
alignment or source-only appearance shifts reinforce camera bias. Cheapest
experiment: TRAIN-only paired-transform diagnostics, then a frozen small matched
fit. Do not expand to a large synthetic accuracy cycle.

Retain Gaussian-risk crossing as a secondary lead for subject-held-out calibrated
C+ work, not a proven invention. Prioritize actual color accuracy as well as
accepted-image error; abstention alone cannot meet the product goal.

## Evidence and limits

327tests passed,14historical warnings,35.60s. All jobs terminal. No export or
latency optimization. No new externally acquired skin dataset, no external
publishing/contact. Fresh search reconfirmed ENCoDE credentialed access and
already-known skin prior art; no new permissive paired-phone source was cleared.
Ordinary phone facial validation and a technically distinctive positive result
remain outstanding. Goal active, not complete or blocked.
