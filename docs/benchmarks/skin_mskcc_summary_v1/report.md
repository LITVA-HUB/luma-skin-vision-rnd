# MSKCC direct skin-color source pilot v1

**MEASURED SOURCE VALIDATION ONLY.** 966 training images / 24 people; 264 validation images / 6 different people.
Ten test participants / 400 images and six calibration participants / 208 images remain numerically unopened.

Input: author-supplied three image-median Lab features. Target: mean of three real SkinColorCatch Lab readings at that skin site.
DeltaE00 is in the native instrument convention; no illuminant angle or invented skin reference is used.

|Locally fit control|Mean DeltaE00|Median|p95|Mean at 80% diagnostic coverage|
|---|---:|---:|---:|---:|
|constant|9.9879|10.4777|17.8445|9.7425|
|affine_a0.01|5.0504|4.6774|9.6788|5.0318|
|affine_a1|5.0495|4.6809|9.6734|5.0309|
|affine_a100|5.0217|4.6710|9.0369|5.0079|
|poly2_a0.01|4.4994|4.0181|9.1291|4.4328|
|poly2_a1|4.4994|4.0279|9.1093|4.4331|
|poly2_a100|4.5701|4.2037|8.8142|4.5307|
|mlp64x32_a0.1|5.1635|4.4246|11.8584|5.0076|
|mlp64x32_a1|4.3702|3.8038|9.7150|4.3694|
|knn5|4.3924|3.5527|10.8903|4.3935|

Source selection by patient-balanced mean: **mlp64x32_a1**.

The selection score has been used to choose the model and is not an independent final performance estimate.
Coverage score is fixed 5-neighbor distance, not calibrated expected color error. Rejection may fail.

TRAIN-only repeatability: 248 distinct sites / 744 pairwise measurement differences; median DeltaE00 2.3333, p95 5.7964.
This reflects within-site repeated measurements, not a certified physical accuracy floor.

No novel proposed architecture, actual pixel pipeline, unseen-camera result, phone-selfie accuracy or cosmetic matching accuracy has been demonstrated here.
All ten controls, predictions, warning logs and exact model replay checks are retained. Original image acquisition is separate.

Sources: [original MSKCC release](https://api.isic-archive.com/doi/mskcc-skin-tone-labeling-dataset/),
[manufacturer instrument convention](https://store.delfintech.com/products/skincolorcatch).
Attribution: Memorial Sloan Kettering Cancer Center; original CC-BY data.
