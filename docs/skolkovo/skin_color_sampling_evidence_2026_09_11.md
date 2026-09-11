# Actual skin-color allocation: evidence addendum

Implemented: image, person/site, site and measured-color training distributions,
bounded importance correction, and person-mass-preserving confounding controls.
Single-image compact inference does not require camera identity. Training uses
actual native instrument Lab; no invented skin-color ground truth.

Measured:21 models on an internal original-TRAIN split. Color allocation mean
6.0012 DeltaE00 versus image-uniform6.1082 and simple person/site6.0084. Modest
tail improvements and within-person permutation contrasts are exploratory.
The strongest simple matched control is essentially tied; no novelty victory.
[Decision and full report](../research/skin_color_sampling_next_decision.md).

Evidence:21 exact color arrays, three exactly reproduced prior control states,
four additional exact full refits,4872 scalar color cases,750141 TRAIN scalar
density pairs,126 coverage rows and108 person-mass identities. Core929,297 params;
maximum measured fit allocation288.35MiB. Existing independent test is unchanged.

Still unvalidated: new ordinary facial-phone accuracy, robust unseen-camera
precision, calibrated refusal, cosmetics matching, technical novelty and product
readiness. This is an honest partial R&D result, not approved legal classification.
