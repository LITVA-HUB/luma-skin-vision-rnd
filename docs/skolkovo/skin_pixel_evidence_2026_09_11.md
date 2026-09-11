# Implemented versus measured versus future: direct skin pixels

Implemented: a local single-crop skin-color estimator trained from scratch
on legitimate original MSKCC CC-BY JPEGs and SkinColorCatch Lab measurements.
It needs no supplied camera ID, cloud or foundation model at inference.
The preferred current small candidate has924932parameters. Its code implements
established patch-set/context/confidence operations; novelty is unverified.

Measured development evidence: three seeds give meanDeltaE00 approximately
3.49–3.51 on264images of66sites from6people excluded from model fitting but
used for epoch/architecture selection. Standard CNN results are3.81–4.11;
approximately matched global MLP3.73–3.90. These are source validation numbers,
not independent final-test or clinical/consumer efficacy evidence.

The intended iterative special mechanism did not robustly beat its matched
plain variant. Negative Huber/context-removal/large histogram experiments are
retained. A six-model blend gives3.3906mean but is much larger; do not report
its accuracy with the small model's resource measurements.

Measured compute for seed17small model on RTX4060:0.4915ms median GPU inference
on prepared descriptors, about108.4MiB training allocation,13.73MiB inference
allocation. Complete current original-JPEG preparation is separate,91.79ms
median CPU over10training images and includes all feature extraction. These
are research measurements, not a deployment service-level guarantee.

Future validation: calibrated expected color error; full matched C+ risk head;
sealed10-person final test; truly new camera/phone; ordinary facial selfies;
cosmetic shade accuracy and clinical/real-use reliability. Both study devices
were used in source development, with author-calibrated acquisition conditions.
No proprietary collection, paid cloud, external publication or legal
classification was performed. The product positioning remains a candidate.

See [decision and detailed evidence](../research/skin_mskcc_pixel_decision.md).
