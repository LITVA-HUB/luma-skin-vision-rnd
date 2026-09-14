# Luma ChromaSeed FG — research model card

Purpose: estimate instrument-referenced native D65/10° skin Lab from prepared regional color statistics. This study changes the input feature subset of the existing analytic 128-landmark RBF model. It is not face identification, skin segmentation, a photo encoder or a cosmetic shade-match service.

Inputs: finite one-row float32 `color36`, encoded-sRGB quantiles (.01,.05,.1,.25,.5,.75,.9,.95,.99), quantile-major RGB at 0:27, mean 27:30, population std 30:33, correlations RG/RB/GB at 33:36. `mean3` gathers [27,28,29], `median3` [12,13,14], central9 [9:18], mean_std6 [27:33], quant27 [0:27], no_corr33 [0:33]. All 36 caller values must currently be finite. Discarded finite coordinates cannot influence subset predictions. X16 retains all 36 inputs and its original projection. Output is one three-component Lab estimate, without camera labels, query targets, other query rows, error confidence or calibration guarantees.

Training: fit-only population moments, balanced person/site/image weights, exact width, RPCholesky maximum 128 centers, alpha .1/1/10, analytic static/joint and normalized/perceptual readouts. Gate uses only selected fit features/camera labels; one-camera joint exports exact static schema. Twelve banks and three seeds. Ninety-six alpha/group settings and 24 policies fixed before final fits. Original TRAIN only, hash d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0. All 24 people and overlapping evaluation roles are reused exploratory data. Data provenance and permission remain those of the existing cache; no additional data/weight licence is inferred from this implementation.

Deployment: [NumPy consumer](../../scripts/chromaseed_feature_groups_numpy.py), with original projection/gated/kernel NumPy dependencies available from the same scripts directory. Example, run from the worktree with its existing environment:

```python
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
from chromaseed_feature_groups_numpy import Predictor

# Demonstration of an evaluated candidate, not a production-selected model.
path = Path("experiments/runs/chromaseed_feature_groups_v1/selected/mixed/perceptual_joint_soft_mean3_s17.npz")
with np.load(path, allow_pickle=False) as archive:
    model = {k: archive[k] for k in archive.files}
predict = Predictor(model)
lab = predict(color36)  # Supply the measured region's real features.
```

Mixed mean3: 4,684 numeric B, 10,323 cached-array B, 12.6 us response, 27.89 ms complete 734-row fit, one CPU thread. Static transfer model is 3,127 B. NPZ archive overhead, runtime dependencies, temporary arrays, Python/process memory and image preprocessing are separate. Reduced files store uint8 indices and FP32 parameters; cached center/readout arithmetic is FP64. No quantization or GPU acceleration claim.

Mixed mean3 error 6.511623 versus raw 5.272642 DeltaE00. The same input group improves both camera-transfer means, with overlapping/confounded people; this does not establish camera invariance. Frozen mixed joint quality/compact select central9, whose outer error is 5.903517. X16 5.206762 is preserved historical control. No selector was changed after seeing outer errors. All outcomes, adverse results and descriptive intervals are in the [report](../benchmarks/chromaseed_feature_groups_v1/report.md).

Independent audit checks every deployed consumer under 33 transformations and refits all 288 nonconstant models. [Protocol](../research/chromaseed_feature_groups_v1_protocol.md), [count erratum](../research/chromaseed_feature_groups_count_erratum.md), [verification](../benchmarks/chromaseed_feature_groups_v1/verification.json). No ordinary-phone facial quality, subgroup fairness, identity recognition, clinical use, patent novelty or Skolkovo eligibility is established by this series.
