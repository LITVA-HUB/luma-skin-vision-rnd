# Technology overview — technical draft, 2026-09-10

Candidate positioning: **Compact adaptive color-normalization and reliability estimation technology for camera-independent analysis of color-sensitive visual objects, with the first commercial application in facial skin analysis and cosmetics recommendation.** This is proposed technical positioning, not an approved legal classification, novelty determination or Skolkovo admission decision.

The engine estimates illumination normalization from one linear camera image, predicts residual normalization error, and supports accept/refuse before downstream color measurement. Current inference uses a compact locally trained CNN, without a foundation model, cloud inference, test-camera CCM, extra test images or adaptation. Raw sensor-level decoding metadata is still required; arbitrary processed smartphone JPEG input is not validated.

**Already implemented:** licensed public-data import, linear preprocessing and target masks, four classical estimators, compact learned C/C+, matched mixture, held-out residual regression, source calibration, selection, reproducible records and hardware profiling. Earlier facial/synthetic infrastructure and its negative result remain preserved.

**Measured on public real data:** SimpleCube++ official 462-image test, three seeds; held-out 600D protocol 931 images; external Sony 30 pilot. See [results](../benchmarks/public_benchmark_report.md). The standard compact estimator improves within-source illumination accuracy over classical methods. The special mixture has no established selective advantage over the strongest matched control and loses to Gray World on the external sensor pilot. This evidences a working research component and remaining technical problem, not an original superior technology.

**Still to validate for facial skin:** region registration, instrument-linked physical color targets, repeatability, skin-specific reliability, camera/ISP and mixed-light domain, cosmetics catalog measurements and user outcomes. Skin-specific colorimetric validation remains future work requiring a facial dataset with appropriate reference measurements. Its present unavailability does not stop public photometric-core research.

Distinctiveness must be judged against FC4, Reweight-CC, C5, CCMNet, uncertainty 2025, VLM-CC2026 and other reviewed work. Ensembles, confidence heads and rejection are established techniques. Current results do not establish patentability.
