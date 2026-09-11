# Independent instrument skin-color evidence

Candidate technical positioning, not approved legal classification: compact
adaptive color estimation and reliability technology for color-sensitive visual
objects, with facial skin analysis and cosmetics recommendation as the intended
first commercial application.

## Implemented

Local JPEG preprocessing, compact patch-based native-Lab skin-color prediction,
subject-held-out expected-DeltaE00 supervision, calibrated rejection, immutable
pre-test artifact locks, comparative baselines and independent metric audits.
No foundation model or cloud inference is required by these experiments.

## Measured on real public data

Original MSKCC CC-BY images paired with actual instrument readings. Independent
400-image/105-site/10-person test, after24TRAIN/6VALIDATION/6CALIBRATION people.
Primary2.775M-parameter color model mean4.457 DeltaE00; at80% acceptance,
Proposed4.159 versus standard C+4.333,95%paired difference interval crosses zero.
The ordinary6-model fusion is stronger: full4.300 and density80%4.145. A special
mechanism advantage is not established. This is direct skin-color error, not an
illuminant-angle proxy. Raw source, reference amendment and negative results
are retained. [Full evidence](../benchmarks/skin_mskcc_selective_v1/report.md).

GPU color-only batch1 median1.470ms on RTX4060 from prepared descriptors;
preprocessing and risk processing are additional. Do not present this as total
photo-to-answer latency. Neither ONNX nor TensorRT skin deployment is claimed.

## Still to validate

Ordinary facial selfies, unseen iPhone/Android models, unsupported illumination,
skin localization, repeated regional accuracy and cosmetics shade decisions.
MSKCC uses specialized capture hardware/settings and both device families were
seen during development. Current accepted80%median3.883/p958.097 fails the
working <=2/<=5DeltaE00 product target. Calibration does not give a per-image
error guarantee. No clinical or universal camera-independence claim is supported.

This evidence supports an implemented, reproducibly evaluated research
component and an honest negative/partial hypothesis result. It does not by
itself establish patent novelty, a finished Luma product or Skolkovo eligibility.
