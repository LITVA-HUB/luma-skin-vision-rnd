# ISSA measured-material prerequisite, version 1

Frozen before reading participant spectral/colorimetric numbers. Metadata and
workbook header/constants/formulas were inspected first. This is a bounded
prerequisite to an image-model material prior, not a skin-photo benchmark.
No MSKCC TEST/CAL or UMINHO held-out data is read.

## Source and identity

Use original ISSA v4 workbook SHA256
7396aa7608f1a6f91ec70bf8f8421252a9d81f71a2bca066eb7e98155b9d1161.
Original CC BY4.0 and acquisition are archived. Preserve raw workbook.
15,256 records and 2,107 subject labels were counted locally, whereas the paper
reports 2,113 subjects. No subject label occurs in multiple origins. Three labels
have conflicting gender codes and one conflicting age code. Keep all records
of each label together. Do not infer missing identities or repair demographics.
Disjoint labels cannot prove that differently labelled records are different
physical people. Report this identity limitation explicitly.
The metadata-only parser initially rejected blank demographics, before a split
or endpoint access. Inspection found 2,716 absent age cells and nine absent
gender cells. These fields remain absent, with aggregate counts reported;
they are not required to assign subject roles and are never imputed.

Origins 9, 10 and 11 are reserved whole, as separate future source-held-out
cohorts, chosen by original source order before inspecting endpoints.
Within each origin 1 through 8, sort subject labels by SHA256 of the UTF-8
string `LumaISSA1|origin|subject`. First floor(0.70*n) are TRAIN, next
floor(0.15*n) VALIDATION, remaining KNOWN_SOURCE_TEST. Every record of a subject
gets the same role. This is a custom protocol, not an author benchmark split.
Save record-to-role manifest only under ignored data/processed/skin_issa_v1.
Commit aggregate counts and manifest/code/protocol hashes before endpoints.

## Extraction and support audit

First decode numeric endpoints for TRAIN only, after checking manifest/lock.
Audit declared support, actual missing/invalid cells, formula patterns and
cached colorimetric consistency. Missing values remain missing. Do not execute
arbitrary workbook formulas. Independently implement only the explicitly
verified XYZ and Lab formula family. Header colorimetric constants are common
public metadata; the workbook white uses 400-700nm at 10nm intervals.
Do not label that white or its derived Lab as full-visible CIE integration.
Do not mix this 2-degree reference convention with MSKCC D65/10-degree targets.

No validation/test endpoints are decoded in the initial schema audit.
If training data is coherent, the first material screen may read VALIDATION,
once its executable bytes and fixed settings are locked. TEST and all reserved
source endpoints remain unread. Fit every basis/statistic on TRAIN only.

## First fixed material screen (if audit permits)

Compare PCA in measured reflectance, optical density (-log reflectance), and
logit reflectance, with latent widths 2,3,4,6,8. Use common 31 measured bands,
400-700nm. Compare an ordinary linear spectral basis first. Use equally
weighted subjects, not equal records, to fit the basis. No validation-based
tuning of transforms or latent widths. Report all 15 configurations.
Before transforms reject nonfinite/out-of-(0,1) measurements if present and
record the exclusion; no hidden clipping of original data. Reconstructed
reflectance may be clipped to [0,1], report its fraction and unclipped error.

Inputs to this screen are full measured spectra. These are oracle compression
controls, NOT RGB reconstruction or image color accuracy. Output measures:
spectral RMSE, relative L2 error, and only after source formula verification,
DeltaE00 relative to supplied TRAIN/VALIDATION Lab under the workbook's actual
common-band white convention. Report mean, median, p95 and worst source mean.
Do not fit risk ranking or call this a selective image model. A low dimensional
basis preserving reference color is necessary evidence for a proposed material
constraint, never sufficient evidence of camera-independent recovery.

## Mechanism and alternatives

Hypothesis: a measured skin-material basis may remove implausible outputs from
a compact image model. Key assumption: the allowed material family preserves
real color variation across people, body sites and sources. Failure: prior
shrinks unusual real skin toward common colors, especially across acquisition
geometries, observers and camera processing. Cheapest falsification is the
oracle compression control above. A useful basis must next be compared against
matched ordinary image heads on actual instrument skin color; do not promote
it merely because spectra compress well.

Skin optical models, low dimensional reflectance and skin color manifolds are
prior art, not established novelty. An alternative is learning only an
ambiguity/rejection constraint rather than forcing color onto a narrow locus.
An opponent should test whether any benefit needs skin-specific geometry or
is explained by an ordinary low dimensional bottleneck/regularization.
