# Luma ChromaSeed-X model card

**Luma ChromaSeed** is the working research family. X denotes the fit-only covariance projection study. The cosmetics e-commerce application is an integration/demo wrapper; this component predicts skin-region color and does not establish production or Skolkovo qualification.

## Contract and mechanism

Input remains one finite FP32 color36 vector from an already prepared skin region: nine encoded-sRGB RGB quantiles [.01,.05,.10,.25,.50,.75,.90,.95,.99], RGB means, population RGB standard deviations, RG/RB/GB correlations, in that order. Source pixels are encoded RGB in [0,1]. The [feature definition](../../scripts/chromaseed.py) is unchanged. A 16-dimensional projection does not remove the need to extract all36 input statistics; the continuous acquisition gate still uses them.

Output is native instrument-reference Lab in the dataset's D65/10° convention. This is not a whole-face detector, identity recognizer, ethnicity classifier, skin diagnosis or validated cosmetic shade recommendation.

Original fit rows define the same unweighted FP32 input/target moments as A. Their person/site-balanced covariance supplies an unsupervised eigenspace. The stored FP32 mean36 and projection36×d map normalized features into FP32 latent coordinates before FP64 kernel arithmetic. Thirteen representations were registered: identity and dimensions8/16/36 × shrinkage0/.1/.5/1. Width and up to128 RPCholesky centers are fit anew in each representation. Each model evaluates either a static readout or joint [Z,s(x)Z] coefficients under analytic normalized/perceptual ridge fitting.

The gate is trained on original, unprojected fit features using the frozen G procedure. A single-camera fit has no active gate and exactly aliases its corresponding projected static readout. This describes fit-domain availability, not an unknown-camera detector. There is no query label, target, batch adaptation or prediction-time fitting loop. No color augmentation is used in X training.

## Measured representative candidate

Mixed role, perceptual joint soft, compact policy, d16/τ.5/α.1:

| Quantity | Measured value |
|---|---:|
| Person mean ΔE00, average of3 basis errors | 5.206762 |
| Matching raw A clean | 5.272642 |
| Numeric payload | 14,181 bytes |
| Raw active payload | 21,973 bytes |
| Reduction in payload | 35.46% |
| Cached NumPy arrays | 29,056 bytes |
| Response, median of seed medians | 13.5 µs |
| Full fit734 rows, seed17 median of3 after warmup | 30.559 ms |
| Fit / query people | 18 / 6 |
| Fit / query rows | 734 / 232 |

CPU Ryzen9 7900X, one thread, NumPy2.5.3/Windows. Full fit includes weights/moments/covariance/projection/exact distances/landmarks/gate/readout. Image preparation, imports, I/O and search are excluded. Cache accounting excludes caller dictionaries, temporary arrays, Python/NumPy and process memory. Actual NPZ archive bytes are recorded separately. Same-run raw control response is about12µs; projection makes inference slightly slower despite fewer stored bytes. Earlier A clean fit34.600ms was a different measurement run, not a new paired training-speed baseline. No GPU or phone timing claim.

The mixed compact comparison improves4/6 people, with descriptive fixed-prediction bootstrap interval[-.251568,.139354] for its −.065880 mean difference versus A. It includes zero and does not account for the long architecture-selection history. These966 rows/24 people and role splits have been reused and overlap.

Across the full study,19/24 same-family/policy/role comparisons worsen. Perceptual compact SLR→iPod chooses8 unshrunk coordinates and gives11.029335 versus raw8.654805; reverse chooses8 unwhitened directions and gives8.770553 versus8.386889. Quality policy also worsens both transfers. Each role independently selected its training setting; these are different models, not one checkpoint evaluated across all roles. Inner quality/p90 guards are not outer guarantees. X is not promoted as a universal replacement.

## Local use

[Representative seed17 weights](../../experiments/runs/chromaseed_projection_v1/selected/mixed/perceptual_joint_soft_compact_s17.npz) are a reproducible example, not a seed selected as best. Published quality averages three seed errors, not this single file or an ensemble.

```python
import sys
import numpy as np

sys.path.insert(0, "scripts")
from chromaseed_projection_numpy import Predictor

path = "experiments/runs/chromaseed_projection_v1/selected/mixed/perceptual_joint_soft_compact_s17.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({key: archive[key] for key in archive.files})

# Supply matching, independently validated skin-region features:
# native_lab = predictor(color36)
```

[Standalone consumer](../../scripts/chromaseed_projection_numpy.py) imports NumPy only. It validates complete fields/shapes/dtypes, finite values and positive scales. It supports original, projected, static, continuous and constant control formats. All111 models×33 transforms match stored outputs to1.95e-12 Lab maximum.15 numerical tests pass. Independent SVD/dense/QR refits cover72 selected instances and12 positive probes, maximum2.29e-5 Lab; all112 timing fits reproduce arrays exactly.

[Report](../benchmarks/chromaseed_projection_v1/report.md) · [Verification](../benchmarks/chromaseed_projection_v1/verification.json) · [Protocol](../research/chromaseed_projection_v1_protocol.md) · [Next decision](../research/chromaseed_projection_next_decision.md).

Ordinary-phone facial accuracy, product shade matching and end-to-end preprocessing remain unvalidated. No new images/data acquisition, legacy validation/calibration/test, participant identifier publication or external upload was used. Commercial rights to dataset-derived weights and name clearance are not established by this implementation; no novelty claim is made.
