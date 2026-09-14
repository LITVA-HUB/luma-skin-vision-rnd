# Luma ChromaSeed K128-N / exact KE trainer

**Purpose.** Instrument-native Lab regression from a prepared skin-region color36 vector. The compact128-center predictor architecture is unchanged. `ChromaSeed-KE` names the exact optimized trainer; KE128 denotes its128-center artifact. This is an engineering research candidate, not a newly validated face model.

**Interface.** Float32 `(batch,36)` encoded-sRGB statistics:9 RGB quantiles [.01,.05,.1,.25,.5,.75,.9,.95,.99], mean3, population std3 and RG/RB/GB correlations. Output `(batch,3)` native Lab. Region/face selection, EXIF/color management, camera behavior and image-to-statistics processing are external. No identity recognition, ethnicity inference, diagnosis, calibrated confidence or demonstrated cosmetic shade-match outcome.

**Predictor.** Stored fit normalization →128 Gaussian landmarks →3 linear readouts → inverse target normalization. Numeric payload20 284B, including centers/coefficients/scalers/width; FP32 storage with FP64 kernels/readout arithmetic. Header/runtime/temporary-memory overhead is additional. Landmark coordinates are selected stored fit features, not thousands of gradient-trained neural weights. No FP16/INT8 claim for this series.

**Optimized training.** Exact median of all positive pair RMS distances via SciPy `pdist('sqeuclidean')` condensed upper triangle and in-place lower-median selection. On-demand person/site-weighted randomized-Cholesky columns, whitened Nyström ridge, same frozen hyperparameters. No pair subsampling, new teacher or new hyperparameter selection in KE. Distance storage remains O(N²), columns/readout use N×rank arrays. SciPy1.18.1 was already installed; explicit research dependency and uv.lock were updated without installing packages. Inference still uses the original NumPy predictor.

**Measurement.** On734 prepared real-cache rows/128 centers, paired seed17 median full fit68.08ms dense versus15.11ms optimized, with one warmup+three measurements. Includes balancing, normalization, width, centers and solve; excludes loading, selection, writing. Ryzen9 7900X, one CPU thread. Mean of three per-model batch-one medians16.73µs including normalization, excluding image/face feature extraction. Synthetic8192 rows:529.78ms and290.47MiB incremental tracked allocation peak; not process RSS, not real-skin evidence and not million-row scalability.

**Equivalence.** 81 inner and27 final models at64/128/256 centers reproduced the already independently audited `column_exact` KF models. All108 payload arrays exactly equal,26 082 checked query rows with0Lab drift,0rounded-width ULP difference. Final128 person-balanced ΔE00 remains5.4387 /8.5970 /8.7050 in mixed/SLR→iPod/reverse roles. This verifies implementation equivalence, not fresh predictive validity.

**Negative controls.** Sampling1024 pairs passed the registered internal guard but hurt mixed128 error to5.9161. Inner size selection chose64 and hurt mixed quality to5.9480. Increasing to256 gives5.4003 /8.5834 /8.8556 and40 252B, not a universal improvement. Previous guided RBF remains stronger in reverse transfer7.7193. Sampled policies, larger models and all earlier R/palette failures remain in the report.

**Data and rights.** Only original TRAIN SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`,966 records/24 people, without reading image arrays. Three historical roles/folds with person separation; roles overlap and are exposed. Legacy validation/calibration/test excluded. No new dataset or external pretrained model. [Dataset provenance/obligations](../ip/skin_mskcc_provenance.md); code, dependencies, derived models and source data have distinct rights. Working name, no trademark clearance. Standard SciPy/Nyström/RPCholesky implementation is not evidence of inventive novelty.

**Load a fixed-seed artifact.** Do not choose seeds by external error. This example uses seed17 for reproducibility, not because it is the best model.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_kernel import predict_kernel

path = root / "experiments/runs/chromaseed_condensed_exact_v1/selected/mixed/condensed_exact_k128_s17.npz"
with np.load(path, allow_pickle=False) as artifact:
    model = dict(artifact)
# color_statistics must match the frozen finite float32(batch,36) schema.
native_lab = predict_kernel(model, color_statistics)
```

**Lifecycle.** No Luma app/backend replacement occurred. All source/protocol locks remain. Next loss-function/iterative-readout hypothesis is planned, not running. Ordinary smartphone face accuracy remains unvalidated. [Full measured report](../benchmarks/chromaseed_fast_kernel_v1/report.md).
