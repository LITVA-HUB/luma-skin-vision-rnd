# Latent capture conditioning and perceptual objective

All54fits completed on REAL original MSKCC images and instrument-native Lab labels. Results are REPRODUCED LOCALLY, but use previously studied source TRAIN/VALIDATION people. The independent400-image TEST and208-image CALIBRATION caches/labels/predictions were not loaded. These scores are not new final-test accuracy or universal facial-phone validation.

## Exact factorial and provenance

Three architectures x two objectives x three seeds x three camera protocols. Every model has929,297stored parameters, identical per-seed initialization and80epochs. Plain and uniform average four color hypotheses. Mixture weights them using a gate inferred from the image. Uniform and mixture receive identical0.1-weighted capture-mode supervision; plain does not. Thus uniform is the strict equal-capacity/supervision comparator, while plain tests whether the added supervision itself helps. The gate uses no provided camera or capture metadata at inference.

The auxiliary gate has2,052parameters and is unnecessary for plain/uniform color inference. Mixture uses it. Keeping unused parameters in a checkpoint is not extra functional capacity for plain/uniform. Four linear hypothesis outputs averaged uniformly are expressively reducible to one output head; the conditional combination is the mechanism being tested.

MSE is standardized native-Lab squared error. DE2 is actual CIEDE2000 squared/25, with native double-precision instrument targets. Full evaluation uses independently checked NumPy/scalar CIEDE2000. No angular-to-color conversion, artificial skin labels, new pretrained weights or large inference model is used.

## All source results

Values are means over three separate seed scores, NOT an ensemble result. p95 is likewise averaged over seed p95 values.

| Protocol | Architecture | Objective | Mean DeltaE00: seed17 /29 /43 | Mean over seeds | Mean p95 | Mode accuracy |
|---|---|---|---|---:|---:|---:|
| mixed | plain | mse | 3.4957 / 3.5284 / 3.4072 | 3.4771 | 7.0733 | 24.6% |
| mixed | uniform | mse | 3.5242 / 3.5189 / 3.4576 | 3.5002 | 7.2727 | 65.0% |
| mixed | mixture | mse | 3.3783 / 3.4698 / 3.4736 | 3.4406 | 6.9316 | 60.5% |
| mixed | plain | de2 | 3.4835 / 3.5571 / 3.4391 | 3.4932 | 7.4447 | 25.4% |
| mixed | uniform | de2 | 3.5204 / 3.5608 / 3.4493 | 3.5102 | 7.1286 | 61.4% |
| mixed | mixture | de2 | 3.4442 / 3.5692 / 3.4394 | 3.4843 | 7.2796 | 43.3% |
| from_SLR | plain | mse | 5.6802 / 6.0602 / 5.7499 | 5.8301 | 11.2459 | 27.0% |
| from_SLR | uniform | mse | 5.8079 / 6.0563 / 5.3514 | 5.7385 | 10.7133 | 40.2% |
| from_SLR | mixture | mse | 4.9975 / 5.2417 / 4.8187 | 5.0193 | 9.7825 | 44.9% |
| from_SLR | plain | de2 | 5.2945 / 5.9691 / 5.8270 | 5.6969 | 10.5457 | 23.2% |
| from_SLR | uniform | de2 | 5.2978 / 5.7979 / 5.7131 | 5.6029 | 10.4527 | 37.6% |
| from_SLR | mixture | de2 | 5.2773 / 5.7605 / 5.3011 | 5.4463 | 9.9017 | 45.2% |
| from_ipod | plain | mse | 5.3087 / 5.6826 / 5.5353 | 5.5089 | 9.4018 | 27.8% |
| from_ipod | uniform | mse | 5.8259 / 7.2178 / 7.2009 | 6.7482 | 11.4012 | 47.0% |
| from_ipod | mixture | mse | 5.7634 / 6.4146 / 5.7124 | 5.9635 | 11.0896 | 42.7% |
| from_ipod | plain | de2 | 7.1997 / 6.4725 / 6.7385 | 6.8036 | 11.5086 | 29.8% |
| from_ipod | uniform | de2 | 7.4360 / 6.5850 / 6.8899 | 6.9703 | 11.8014 | 41.7% |
| from_ipod | mixture | de2 | 7.8231 / 7.8796 / 6.8225 | 7.5084 | 12.3615 | 42.9% |

Mixed:24TRAIN people/966images;6selection/evaluation people/264images. FromSLR:8TRAIN people/323images and3same-camera selection people/132images; opposite-camera3people/132images evaluated after checkpoint selection. FromiPod:16TRAIN people/643images and3same-camera selection people/132images, opposite3people/132images evaluated after selection. No opposite-camera observations or target scales enter a fit. All54configurations were fixed before these experiments; no camera-selected challenger subset. Historical source exposure still limits independence.

## Matched mechanism contrasts

Negative favors mixture. These are descriptive paired contrasts on small previously exposed source cohorts, not claims of statistical significance.

| Protocol | Objective | Mixture minus uniform | Mixture minus plain | Seedwise mixture minus uniform |
|---|---|---:|---:|---|
| mixed | mse | -0.0596 | -0.0365 | -0.1458 / -0.0491 / +0.0161 |
| mixed | de2 | -0.0259 | -0.0090 | -0.0762 / +0.0084 / -0.0099 |
| from_SLR | mse | -0.7192 | -0.8108 | -0.8103 / -0.8146 / -0.5326 |
| from_SLR | de2 | -0.1566 | -0.2506 | -0.0205 / -0.0374 / -0.4120 |
| from_ipod | mse | -0.7847 | +0.4546 | -0.0625 / -0.8032 / -1.4885 |
| from_ipod | de2 | +0.5381 | +0.7049 | +0.3871 / +1.2947 / -0.0674 |

Compare against both controls and both directions. A gain over the auxiliary-task control alone is insufficient if the ordinary plain estimator is stronger. A gain for one camera direction does not establish universal camera independence. Mode accuracy is a diagnostic of the auxiliary task, not skin-color accuracy or calibrated reliability.

## Verification and limitations

54fits, 162exact checkpoint prediction-array replays, 54gate/hypothesis replays, 9504independent scalarDeltaE00 comparisons; maximum metric gap4.88e-15. All target scales and same-camera epoch-selection boundaries checked; initial states match. Numerical loss independently agrees on10,000random color pairs within2.14e-14; all34rounded Sharma fixtures pass5e-5 tolerance, with finite-difference/neutral-gradient tests. Formula checks are not synthetic skin-accuracy evidence.

The exact CIEDE2000 formula is piecewise and has hue discontinuities; using an autograd implementation does not remove these. Its direct optimization is not guaranteed to improve generalization or the tail. All gradient norms were checked finite without clipping. Stored model files and measured training allocation are recorded per fit; no new batch1/ONNX/TensorRT or full-pipeline latency claim is made.

These models and all prior negative arms remain archived. The previously measured independent skin benchmark stays authoritative and unchanged. No new source result is promoted to final test, cosmetics accuracy, clinical utility or a patent-novelty claim. [Frozen protocol](../../research/skin_capture_protocol_v1.md).
