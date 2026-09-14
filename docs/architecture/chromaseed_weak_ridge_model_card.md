# Luma ChromaSeed-W128

**Status:** expanded-penalty research candidates, not accepted as replacements for the previous P/KE candidates. W identifies the weaker-ridge study; the working family name remains **Luma ChromaSeed**. Only training selection changed. No new inference architecture or established scientific novelty is claimed.

**Interface.** Prepared float32 `(batch,36)` encoded-sRGB color statistics → native Lab `(batch,3)`. Nine RGB quantiles(.01,.05,.1,.25,.5,.75,.9,.95,.99), mean3, population std3, RG/RB/GB correlations. Same128 stored Gaussian landmarks and seven-array payload: x_mean/x_std/y_mean/y_std/width/centers/coefficient.20 284 numeric bytes, FP32 storage and FP64 kernels/readout. Runtime libraries, file headers, temporary allocations and image/face feature extraction are additional.

**Training.** Frozen P numerical helpers, exact condensed width and weighted randomized-Cholesky Nyström basis. Alpha grid0.0001/0.0003/0.001/0.003/0.01/0.03/0.1/1/10, widths0.5/1/2, seeds17/29/43. Same normalized, global/local perceptual and two robust correction families. Correction checkpoints0/1/4/16 are selected jointly with alpha/width on inner people. All corrections collapse into the existing coefficient array. A query does not execute these training iterations.

**Measured results.** Person-balanced ΔE00, means of three independently seeded models, not an ensemble:

| Method | mixed | SLR→iPod | iPod→SLR |
|---|---:|---:|---:|
| W normalized | 5.5447 | 9.5273 | 8.9189 |
| W global weights | 5.4995 | 9.5260 | 8.6863 |
| W local weights | 5.4844 | 9.6353 | 8.7300 |
| W fixed-geometry correction | 5.5616 | 9.4134 | 8.5290 |
| W updated-geometry correction | 5.5666 | 9.5237 | 8.5144 |
| Previous P global weights | 5.3901 | 8.6548 | 8.3869 |

All15 inner family scores improved;13/15 outer comparisons with each corresponding previous P family worsened. The two reverse iterative improvements compare against P's zero-step aliases, not against its stronger global-weight candidate. Some correction beats W's weaker normalized control, without establishing an overall improvement. Older guided RBF reverse7.7193 and ChromaSeed-R mixed5.3551 remain useful stronger references on those roles.

**Selected settings and costs.** All chosen alpha values are internal to the new grid(.001/.003/.01/.03). Width2 is chosen in mixed and SLR→iPod, width1 in reverse. Fixed correction chooses1/4/1 steps; updated correction1/16/16, with5 actual solves in the SLR16-step case after early rejection. Mixed global-weight full fit22.21ms and batch-one16.43µs on734 prepared fit rows, Ryzen9 7900X, one CPU thread. Reverse updated correction uses111.33ms training for16 executed solves and16.47µs response, still worse in quality than previous P global weights. Complete table in report; timings exclude image processing/I/O/search as specified there.

**Verification.**26 numerical tests pass. All2 916 matching P bank payloads independently checked exactly equal. All8 748 stored readouts and1 239 300 inner prediction rows audited;15 selections,45 final independent analytic-tensor/SVD refits,17 970 outer predictions and18 positive-step probes at alpha0.0001 checked. Max final refit query drift0.000369Lab, positive probes0.000520Lab, within the registered0.002 limit. All1 944 regularized training-objective traces are nonincreasing.60 selected timing fits with warmups reproduce arrays exactly;24 extra fixed16-step fits measure cost only. None of this creates new predictive validation.

**Data and application boundary.** Original TRAIN only, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`,966 rows/24 people. Only color,target,patient,site,device loaded; no image/token arrays, legacy validation/calibration/test or new dataset. Inner/outer people are disjoint per role, while historical roles overlap. Camera transfer changes people/color distributions too. Ordinary smartphone facial accuracy, cosmetic shade matching, confidence calibration and scientific novelty remain unestablished. No identity/ethnicity/medical output. Existing [data provenance](../ip/skin_mskcc_provenance.md) and separate code/data/derived-model rights remain unchanged. The original app and repository were not changed.

**Inspect a fixed artifact.** Example uses seed17, not a seed selected by outer error:

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_kernel import predict_kernel

path = root / "experiments/runs/chromaseed_weak_ridge_v1/selected/slr_to_ipod/local_irls_s17.npz"
with np.load(path, allow_pickle=False) as archive:
    model = dict(archive)
# color_statistics must have the finite float32(batch,36) frozen schema.
native_lab = predict_kernel(model, color_statistics)
```

[Report](../benchmarks/chromaseed_weak_ridge_v1/report.md) · [Protocol](../research/chromaseed_weak_ridge_v1_protocol.md) · [Next decision](../research/chromaseed_weak_ridge_next_decision.md).
