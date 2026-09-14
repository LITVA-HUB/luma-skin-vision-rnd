# Luma ChromaSeed-A model card

**Luma ChromaSeed** is the research family name. A denotes the analytical joint-learning and affine-augmentation study. The Luma cosmetics application remains an integration/demo layer. The measured component predicts skin-region color; it is not a production-qualified facial analysis system or proof of Skolkovo eligibility.

## Contract

Input: one finite FP32 vector of 36 prepared skin-region statistics. Nine encoded-sRGB quantiles (.01, .05, .10, .25, .50, .75, .90, .95, .99), each ordered R/G/B, occupy indices 0:27; mean RGB 27:30; population standard deviations 30:33; RG/RB/GB correlations 33:36. Source pixel RGB is in [0,1]. This feature definition is inherited from [color36](../../scripts/chromaseed.py). Face detection, skin masking and consistent preprocessing are separate requirements.

Output: three native dataset instrument-reference Lab values, D65/10° convention. A prediction is not a biometric identity, skin diagnosis, ethnicity label or demonstrated product shade match. No query camera label or observed Lab is required.

## Mechanism

Original fit rows define FP32 input/target normalizers, the exact kernel width, 128 landmarks and whitening. Static readout uses Z; joint soft readout jointly fits [Z,s(x)Z]. The acquisition signal s is a clipped continuous affine function learned on original fit rows using the frozen G procedure. Both color coefficient blocks are fitted together by an analytical ridge solve. There is no prediction-time learning loop or growing connectivity. The consumer evaluates the shared 128 kernels once, then combines both readouts.

The saved joint model has seven ordinary arrays plus a correction matrix, 37 gate coefficients, rho=1 and soft mode=1. A fit containing one acquisition group exports an exact alias of the newly fitted static model at that same alpha/eta. This is not an inference-time unknown-camera safeguard.

The registered augmentation creates 16 bounded affine RGB variants per source record. Its target remains the original instrument Lab as a synthetic assumption; total weight per source stays unchanged. Copies never cross person folds. Normalization, landmarks, gate and perceptual geometry use original fit rows only.

## Measured representative configuration

Perceptual joint soft, clean inner selection, alpha=.1, eta=0:

| Quantity | Measured value |
|---|---:|
| Mixed person mean ΔE00, average of three seed errors | 5.272642 |
| Exact static / old G soft controls | 5.390066 / 5.295118 |
| SLR→iPod / iPod→SLR person ΔE00 | 8.654805 / 8.386889 |
| Mixed fit / query rows | 734 / 232 |
| Mixed fit / query people | 18 / 6 |
| Numeric payload, active joint | 21,973 bytes |
| Cached NumPy arrays, active joint | 44,640 bytes |
| Batch-one response, median of seed medians | 11.8 µs |
| Full mixed fit, seed17, median of three after warmup | 34.600 ms |
| Corresponding eta=.75 full mixed fit | 88.481 ms |

Ryzen 9 7900X, one CPU thread, Windows/NumPy 2.5.3. Response uses ready color36, excluding image preparation, archive load and imports. Full fit includes preparation from existing feature arrays, weights, normalization, exact width, landmarks, gate and readout. Cache bytes exclude retained caller dictionaries, transient arrays, Python/NumPy and process memory. Actual NPZ archive size is stored separately in results.json. Single-camera static aliases have 20,284 numeric bytes and 41,272 cached-array bytes. No GPU or phone inference claim.

Every clean policy chose eta=0; every guarded policy chose eta=.75. The guarded mixed stress error improves 6.278089→6.232160 but SLR→iPod clean error worsens 8.654805→8.743031 and stress worsens 9.234482→9.309668. The inner allowance .05 does not bound outer degradation. Positive augmentation is therefore an experimental variant, not a universal improvement.

The mixed joint result improves four of six people against G soft, but the descriptive paired fixed-prediction interval includes zero. Joint inner error is worse than G soft and static. These are repeatedly reused, overlapping TRAIN roles: 966 rows, 24 people, three basis seeds rather than independent participant replications. The older guided RBF remains stronger on reverse transfer. Ordinary-phone facial accuracy remains unvalidated.

## Local numerical use

The [representative seed17 payload](../../experiments/runs/chromaseed_affine_v1/selected/mixed/perceptual_joint_soft_clean_s17.npz) is one reproducible model, not a winner selected among seeds. Reported quality averages three seed errors and does not describe this single file's error or an ensemble.

From the worktree root:

```python
import sys
import numpy as np

sys.path.insert(0, "scripts")
from chromaseed_gated_numpy import Predictor

path = "experiments/runs/chromaseed_affine_v1/selected/mixed/perceptual_joint_soft_clean_s17.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({key: archive[key] for key in archive.files})

# Supply one finite float32[36] from matching skin-region preprocessing:
# native_lab = predictor(color36)
```

The unchanged [NumPy-only consumer](../../scripts/chromaseed_gated_numpy.py) validates fields/shapes/dtypes/scales and reproduces saved model outputs. All 111 final cases were checked at 33 transformations; 72 selected new models and 12 fixed augmentation probes were independently reconstructed by QR/SVD, maximum difference 9.63e-6 Lab. All 112 timing fits reproduced arrays exactly. This is numerical reproducibility, not confirmation of future phone-camera accuracy.

[Full report](../benchmarks/chromaseed_affine_v1/report.md) · [Verification](../benchmarks/chromaseed_affine_v1/verification.json) · [Protocol](../research/chromaseed_affine_v1_protocol.md) · [Next study decision](../research/chromaseed_affine_next_decision.md).

No raw images, direct participant identifiers, new data acquisition, legacy validation/calibration/test, external uploads or publication were used. Dataset/derived-weight commercial rights and name clearance are not established by this experiment; no novelty claim is made.
