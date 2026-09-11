# Real-image measured-material decoder screen

All predictions use one image. Ground truth is actual MSKCC native instrument
Lab (D65/10degree); DeltaE00 below is skin color error, not illuminant angle.
These are heavily explored SOURCE development cohorts, not new independent
tests. MSKCC TEST/CAL, ISSA reserved origins and UMINHO held-outs were not read.

Five arms x three seeds x three protocols. All share the same parameter
shapes, initialization, image data/resolution and80epoch fitting budget.
Material/tangent have the same ISSA TRAIN-only prior. Tangent is its fixed
first-order expansion, so the residual pair tests the nonlinear decoder
against matched linear color geometry. Capture mode is never an input.

Each table entry averages three separate seed scores, not ensemble predictions.
The p95 column is the mean of three individual p95 values.

| Protocol | Arm | Mean DeltaE00 | Mean seed p95 | Mean at80% | Seed means |
|---|---|---:|---:|---:|---|
| mixed | direct | 3.4406 | 6.9303 | 3.4407 | 3.3785, 3.4698, 3.4735 |
| mixed | tangent | 3.5112 | 7.3823 | 3.5393 | 3.5084, 3.4928, 3.5323 |
| mixed | material | 3.5141 | 7.2257 | 3.4992 | 3.5291, 3.5002, 3.5131 |
| mixed | tangent_residual | 3.4510 | 7.3864 | 3.4953 | 3.3890, 3.4773, 3.4866 |
| mixed | material_residual | 3.4529 | 7.2327 | 3.4787 | 3.4448, 3.4397, 3.4742 |
| from_SLR | direct | 5.0193 | 9.7826 | 5.1088 | 4.9977, 5.2416, 4.8187 |
| from_SLR | tangent | 4.9531 | 9.1153 | 5.0204 | 5.0433, 4.8575, 4.9584 |
| from_SLR | material | 4.9083 | 9.1436 | 4.9778 | 4.9087, 4.9035, 4.9126 |
| from_SLR | tangent_residual | 5.0311 | 9.7962 | 5.3053 | 4.9601, 5.2824, 4.8510 |
| from_SLR | material_residual | 5.0100 | 9.8046 | 5.3498 | 4.8415, 5.3424, 4.8461 |
| from_ipod | direct | 5.9635 | 11.0899 | 5.7914 | 5.7634, 6.4147, 5.7124 |
| from_ipod | tangent | 5.4509 | 10.1641 | 5.2781 | 5.6321, 5.1510, 5.5695 |
| from_ipod | material | 6.1311 | 10.5748 | 6.4299 | 6.3720, 6.2309, 5.7906 |
| from_ipod | tangent_residual | 6.0128 | 10.4162 | 5.9151 | 5.6676, 6.0342, 6.3367 |
| from_ipod | material_residual | 6.1110 | 10.5066 | 6.2489 | 5.6713, 6.4799, 6.1817 |

## Strong historical controls (reproduced locally earlier)

| Protocol | Method | Mean DeltaE00 |
|---|---|---:|
| mixed | plain_mse | 3.4771 |
| mixed | mixture_mse | 3.4406 |
| mixed | training_only_graph | 3.6394 |
| from_SLR | plain_mse | 5.8301 |
| from_SLR | mixture_mse | 5.0193 |
| from_SLR | training_only_graph | 5.8339 |
| from_ipod | plain_mse | 5.5089 |
| from_ipod | mixture_mse | 5.9635 |
| from_ipod | training_only_graph | 4.9736 |

## Matched nonlinear mechanism comparisons

Negative difference favors material. Intervals are descriptive patient-cluster
bootstraps on repeatedly inspected cohorts, without multiplicity correction.
They do not establish a confirmatory improvement.

| Protocol | Arm vs control | Image-mean difference | Patient-mean95% interval | All3seeds improve |
|---|---|---:|---|---|
| mixed | material vs tangent | 0.0030 | [-0.0403,0.0431] | False |
| mixed | material_residual vs tangent_residual | 0.0019 | [-0.0434,0.0403] | False |
| mixed | material_residual vs direct | 0.0123 | [-0.0802,0.1022] | False |
| from_SLR | material vs tangent | -0.0448 | [-0.1765,0.0644] | False |
| from_SLR | material_residual vs tangent_residual | -0.0211 | [-0.1402,0.1384] | False |
| from_SLR | material_residual vs direct | -0.0094 | [-0.4700,0.2464] | False |
| from_ipod | material vs tangent | 0.6803 | [0.0927,1.0750] | False |
| from_ipod | material_residual vs tangent_residual | 0.0981 | [0.0519,0.1348] | False |
| from_ipod | material_residual vs direct | 0.1475 | [-0.7704,0.6300] | False |

## Color interface and scope

The prior uses measured ISSA reflectance, not its2degree Lab labels. The
decoder uses original CIE10degree CMFs and D65, with explicit interpolation
and constant endpoint extrapolation of missing skin tails. This is a model
assumption, not new spectral ground truth. On already-read TRAIN spectra
with extra measured bands, the derived tail sensitivity is mean0.00159,
p950.00518DeltaE00; neither branch is an instrument10degree validation.
Residual variants can depart from the material model. Strong real-photo
controls determine utility; low oracle spectral error does not.

Original ISSA is CC BY4.0. Original CIE tables are CC BY-SA4.0; adapted
integration constants retain attribution/share-alike. Do not describe a
checkpoint bundling those buffers as unrestricted proprietary output.
[CIE provenance and metadata discrepancy](../../data/provenance/cie_material_v1/README.md).

## Uncalibrated risk and compute

Ranking uses four-hypothesis dispersion, not calibrated expected error.
A matched C+ error head and post-hoc calibration have not been demonstrated
by this screen. Fixed coverage is not a deployed threshold guarantee.

![Skin error and accepted coverage](risk_coverage.png)

The CSV contains complete curves, including below50% shown outside the plot.
All model states store937,521parameters. Direct uses929,297active; hard
tangent/material934,437; residual arms937,521. No foundation model is needed.
Largest measured fit-process allocation: 108.83MiB.
Largest model file: 3,759,067bytes.
Fit wall times include ordinary process overhead and some concurrent tests;
they are not batch1latency. No new inference VRAM, ONNX or latency claim.

## Verification

135exact color arrays,45gate/hypothesis/free-coordinate sets,
7920independent scalar color checks,
270coverage rows and7920curve points verified.
Frozen prior buffers, fitting-only target scales and same-camera epoch
selection are checked. All matched initial state hashes agree.

Independent MSKCC result remains unchanged: primary full4.4570/80%4.1591;
ordinary fusion4.3005/4.1447 is stronger. Ordinary phone facial skin
accuracy and a novel selective-system advantage remain unvalidated.
