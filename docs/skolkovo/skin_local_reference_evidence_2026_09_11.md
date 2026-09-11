# Local-reference skin-color evidence: partial gain, transfer limitation

IMPLEMENTED: reproducible single-query global/local reference regressions with
fixed appearance and estimated-color affinities, exact person exclusion and
camera-held-out source banks. No query reference or camera metadata at inference.

MEASURED ON PUBLIC REAL DATA: native MSKCC skin Lab. Person-excluded source
mean5.4259 becomes4.5860 DeltaE00, with21/24 people improved and p9511.4364 to
10.0157. At common80% coverage4.9899 becomes4.2031. These are exploratory source
results of standard local regression, not proof of a novel neural method.

LIMITATION MEASURED: new source camera-transfer systems remain worse than strong
previously locally reproduced compact neural comparators. Best new mixed3.9343,
forward7.1177,reverse6.6968 versus historical3.4406/4.8328/4.9736. Coverage
rejection can worsen unseen-camera error. Reference-bank capacity/runtime must
be counted separately from neural weights.

STILL UNVALIDATED: ordinary phone facial accuracy, calibrated refusal, cosmetic
shade decisions and a technically distinct improvement over matched C+. The
new user capacity allowance adds at most200,000 neural parameters to929,297;
it is planned engineering budget, not achieved technology or approved legal
classification. Original MSKCC CC-BY provenance remains unchanged.

Independent400-image skin evidence remains primary4.4570/80%4.1591 versus
ordinary fusion4.3005/4.1447. No new independent accuracy claim.

[Report](../benchmarks/skin_local_reference_transfer_v1/report.md),
[decision](../research/skin_local_reference_next_decision.md),
[capacity budget](../research/skin_capacity_budget_2026_09_11.md).
