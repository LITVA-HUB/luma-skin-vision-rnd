# Decision after observed-patch support experiments

The product endpoint remains instrument-referenced skin native Lab / CIEDE2000.
Illuminant angular accuracy is component evidence and cannot substitute for skin
color accuracy. The user's renewed clarification changes neither this endpoint
nor the preserved independent test.

## Measured evidence

45 matched fits on original MSKCC TRAIN/VALIDATION photographs compare five
training arms, three seeds and three source protocols. References are actual
site-level instrument measurements; derived training bags do not create new
photographed scenes or new measurements. All inference models have 929,297
parameters. [Full report](../benchmarks/skin_capture_support_v1/report.md).

Paired stratified sampling improves the SLR-to-iPod source mean from 5.0193 to
4.8328 DeltaE00, but mixed-source mean worsens from 3.4406 to 3.5619 and reverse
transfer remains 5.9520 versus 5.9635. The historical reverse graph control is
stronger at 4.9736. The favorable direction does not win in every seed. These
small, heavily reused development populations cannot support confirmation.
There is no universal accuracy gain or matched calibrated C+ victory.

TRAIN has 1,421 same-site pairs with exactly matching native Lab, but no
cross-camera pairs. Variation includes capture mode; patch pattern similarity
does not establish spatial registration. Real patches alone do not guarantee
that a newly composed bag represents a physically possible single photograph.

The subsequent six-model diagnostic supplies true source capture labels to
route each patch on those TRAIN virtual bags. Known local routing fails to
rescue frozen experts. This is teacher-assisted TRAIN evidence, not a deployment
result. Weighted-sum supervision need not identify individually correct color
experts. The observation is consistent with that concern, but does not prove
the cause of failure or exclude differently trained local routing.

## Next falsifiable alternatives

1. Remove the mixture and mode gate. Compare a plain native-Lab regressor with
   and without observed-patch sampling under the same data and training budget.
   Mechanism: avoid unsupported conditional decomposition. Key assumption:
   shared features retain sufficient color information. Failure mode: multimodal
   capture ambiguity requires specialization. Cheapest test: a fixed source-only
   factorial screen, then both transfer directions if promising.
2. Explicitly supervise conditional experts with real TRAIN capture labels and
   native Lab, while withholding those labels from inference. Mechanism: make
   the individual heads meaningful before testing routing. Assumption: capture
   modes describe useful conditional mappings. Failure mode: clinical mode
   specialization harms unseen devices. Cheapest test: compare conditional-head
   supervision to the existing weighted-sum model, with matched capacity/budget.

These are competing unverified hypotheses, not a promise of novelty or gain.
Do not automatically add a local router after its diagnostic failed. Retain
simple RGB/patch controls and test whether the extra components are necessary.
No new fit is reported by this decision document.

## Product success and boundaries

The working aspiration is median skin DeltaE00 <=2 and p95 <=5 at >=80% accepted
coverage on new people and unseen ordinary phones under a specified capture
protocol. These are engineering targets, not established universal perceptual
thresholds or an achieved result. A research victory also requires a reliable
gain against the strongest matched baseline on an untouched evaluation.

The independent 400-image, 10-person MSKCC result remains primary mean 4.4570,
80% mean 4.1591; ordinary fusion achieves 4.3005 and 4.1447 respectively. Both
camera families were seen during development. Clinical/dermoscopic photographs
do not validate ordinary facial selfies, makeup matching or universal camera
independence. Exposed TEST/CAL remain evaluation archives; do not tune on them.

No new data, external weights, export or publication. Original MSKCC CC-BY
provenance remains applicable. Prior synthetic failure is preserved. Goal unmet.
