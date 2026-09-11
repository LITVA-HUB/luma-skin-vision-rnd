# Direct instrument skin-color evidence — bounded source pilot

Implemented: original-source metadata acquisition, exact image/site/person
joins, subject-disjoint train/validation/calibration/test partition, ten compact
color-summary regressors, instrument-native CIEDE2000 and independent scalar
metric/replay audit. Real JPEG acquisition is a separate step and does not
make the summary experiment an end-to-end image pipeline.

Measured: 24 training people /966images and6validation people /264images.
The source-selected MLP produced meanDeltaE00 4.3702, median3.8038,
p959.7150. At80% coverage by a fixed feature-distance score mean4.3694;
the score failed to meaningfully reduce mean error and worsened the tail.
These are model-selection results, not an independent final test, publication-
quality efficacy claim, clinical validation or validated cosmetic matching.

Audit: ten exact model/target replays,60independent fixed-coverage cases and
34Sharma reference pairs. Maximum scalar discrepancy4.89e-15.
Underlying source release is original MSKCC/ISIC CC-BY; version not specified
on the landing page, with individual API license checks during acquisition.
No externally pretrained dermatology weights have been used.

Not yet validated: local image feature/pixel pipeline; proposed novel mechanism;
calibrated color-error rejection; completely unseen-camera transfer; everyday
phone facial color; deployment latency/VRAM for a skin image model. Ten test
people remain numerically unopened. Do not claim facial skin accuracy from the
older illuminant experiments or from the present clinical close-up dataset.

Candidate technology positioning remains: compact adaptive color normalization
and reliability estimation for analysis of color-sensitive visual objects, with
facial skin analysis and cosmetics recommendation as the intended first
application. This is candidate technical wording, not approved legal classification.

Evidence: [benchmark](../benchmarks/skin_mskcc_summary_v1/report.md),
[frozen protocol](../research/skin_mskcc_protocol_v1.md),
[next decision](../research/skin_mskcc_next_decision.md),
[original release](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/).
