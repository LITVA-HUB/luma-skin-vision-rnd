# Pre-test reference-completeness amendment

The original pre-calibration lock was SHA256
`f4c286c74b79e0643f6a50766cae0d8298e04df109b7ee54a2462cc15739b10f`.
It and its loader are preserved under the selective benchmark's
`reference_amendment_archive`. Do not rewrite the original OOF protocol or receipt.

The first authorized calibration load failed before decoding calibration pixels
or fitting calibrators because three image rows of one skin site had empty
second and third instrument repetitions. Subsequent inspection was restricted
to the six CALIBRATION subjects: 205 image rows have three complete Lab readings;
three have one. None has a partial Lab reading or lacks all readings. Means of
available complete readings agree with the published averages within 0.006667
Lab channel units. Calibration numerical reference completeness has therefore
been inspected; it is incorrect to claim no calibration endpoints were opened.
No calibration model predictions, error ranking or performance were inspected.
TEST endpoints and pixels remain unopened at this amendment.

Before test, replace the loader's requirement of exactly three readings with:
use the arithmetic mean of one to three actually published COMPLETE native Lab
instrument readings, checking agreement with the published average at the
existing absolute tolerance 0.051 per channel. Preserve absent repetitions as
NaN in provenance only; never impute measurements or use NaN as a target.
Reject incomplete individual Lab vectors, nonfinite readings, missing all
readings or inconsistent means by failing the run, without silent exclusion.
Report the count of images with one, two and three readings separately.
This rule applies identically to calibration and test. Existing training and
validation targets (all three readings) remain bitwise unchanged.

All images and patient roles are retained. All color checkpoints, 24 risk heads,
selected configurations, feature definitions, normalization, fitting budgets,
calibration rules and evaluation metrics remain unchanged. This amendment is
solely reference parsing, discovered on calibration and declared before test.
The amended pre-calibration lock binds this document, the updated loader and
the archived originals. The final lock must precede TEST access as before.

An average of one measurement has less repeatability support than an average
of three; neither is proof of instrument trueness or of exact correspondence
between a point instrument footprint and every pixel of the photographed site.
