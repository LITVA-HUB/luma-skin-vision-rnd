# Luma ChromaSeed-P128

Research candidates for instrument-native Lab regression from a prepared color36 skin-region vector. P means perceptual readout training. P128-G refers to `constant_de2`, P128-L to `local_de2`; the family name remains **Luma ChromaSeed**. Neither is adopted as a universal replacement for KE128. These are compact kernel regressors, not a newly established neural architecture.

**Interface and payload.** Float32 `(batch,36)` with the frozen encoded-sRGB statistics:9 RGB quantiles(.01,.05,.1,.25,.5,.75,.9,.95,.99), mean3, population std3, RG/RB/GB correlations. Output `(batch,3)` native Lab. The seven arrays are x_mean/x_std/y_mean/y_std/width/centers/coefficient.128 landmarks,20 284 numeric bytes. Storage FP32, arithmetic FP64. File headers, runtime libraries and temporary allocations are additional. Face/region selection, color management and feature extraction are external to the measured predictor.

**Training.** Exact KE width and on-demand weighted randomized-Cholesky columns, Nyström basis, then a coupled three-output ridge solution under shared or per-target local CIEDE2000 tensors. The tensors are finite-difference approximations, checked independently against analytic infinitesimal geometry. They are collapsed into output coefficients and are not needed during inference. Optional robust training corrections were evaluated for0/1/4/16 steps. All six iterative family/role selections chose0; the selected iterative artifacts are ordinary normalized-ridge controls.

**Measured quality.** Person-balanced ΔE00, means over three separately fitted seeds:

| Variant | mixed | SLR→iPod | iPod→SLR |
|---|---:|---:|---:|
| KE128 / normalized control | 5.4387 | 8.5970 | 8.7050 |
| P128-G / shared weights | 5.3901 | 8.6548 | 8.3869 |
| P128-L / local weights | 5.3842 | 8.7874 | 8.4135 |

Shared weights improve two roles but worsen one. Older guided RBF reverse7.7193 and ChromaSeed-R mixed5.3551 remain stronger references on those individual roles. No robust overall win is established. All15 primary choices selected alpha0.1, the lower grid boundary, and width1; a weaker-regularization follow-up remains planned.

**Measured costs.** On734 prepared mixed fit rows, one CPU thread Ryzen9 7900X, shared weights22.58ms standalone fit versus current-study normalized control16.39ms. Shared-weight inference16.47µs, mean of three per-model batch-one medians; image processing excluded. One fit warmup/three timed refits,20 inference warmups/three passes.60 selected timing refits with warmups reproduced arrays exactly. Additional fixed16-step cost probes took46.96–112.78ms depending on role/family and actual early stopping; they add no new outer quality observations.

**Verification.**15 numerical tests pass, including34 published metric examples, independent analytic tensor comparison, augmented least squares, zero-step identity and monotone synthetic objectives.324 normalized bank controls exactly reproduce KF. Independent audit:2916 stored readouts,413 100 inner prediction rows,15 choices,45 final SVD refits/17 970 final prediction rows,18 positive-step probe readouts. Maximum final SVD query drift2.394e-5Lab. All648 logged regularized-objective trajectories are nonincreasing. This is numerical implementation evidence, not a fresh clinical/product/user validation.

**Evidence limits and provenance.** Only frozen original TRAIN SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`,966 records/24 people. Images/tokens were not loaded. Roles/folds have person separation but are historically explored and overlapping. Camera roles also change subject/color distributions. Legacy validation/calibration/test excluded. No new dataset, weights, dependencies or application integration. [Existing dataset provenance](../ip/skin_mskcc_provenance.md) and software/data/derived-model rights remain separate. No identity/ethnicity inference, diagnostic output, calibrated confidence or validated cosmetic shade matching. Ordinary-phone facial accuracy is unknown.

**Load example.** Seed17 is fixed for reproducibility, not selected by outer error:

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_kernel import predict_kernel

path = root / "experiments/runs/chromaseed_perceptual_v1/selected/mixed/constant_de2_s17.npz"
with np.load(path, allow_pickle=False) as data:
    model = dict(data)
# color_statistics must be finite float32(batch,36) in the frozen schema.
native_lab = predict_kernel(model, color_statistics)
```

[Full report](../benchmarks/chromaseed_perceptual_v1/report.md) · [Protocol](../research/chromaseed_perceptual_v1_protocol.md) · [Next decision](../research/chromaseed_perceptual_next_decision.md).
