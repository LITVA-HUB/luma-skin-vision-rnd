# ChromaSeed TG model contract

Research regressors from prepared color36 skin-region statistics to native instrument D65/10° Lab3. No identity recognition, face detection, skin segmentation, ordinary-phone color guarantee or cosmetic shade catalogue is included. Same deployment architecture for Adam, TAGI-diag and TAGI-full3: d→64 ReLU→3. Mean3 consumes coordinates27/28/29; raw36 consumes all36. 451/2563 network parameters; 1855/10564 numeric bytes including FP32 normalizers and mean3 uint8 indices. Output is deterministic; posterior variances are discarded and are not a confidence estimate.

Input is finite encoded-sRGB color36 in the existing prepared-feature contract, not linear RGB or arbitrary three-channel images. Mean3 consumer still validates all36 coordinates and ignores the finite unused ones. One row or a batch is accepted. Fit-only population moments are computed in FP64 and stored FP32; x normalization is performed in FP32, then the network computes with cached FP64 weights. Output is rescaled to native Lab. Cached arrays3659/20816 B exclude Python object and process overhead. Numeric storage, compressed NPZ bytes, optimizer state and actual process RAM are distinct quantities.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_gaussian_numpy import Predictor

path = root / "experiments/runs/chromaseed_gaussian_v1/models/mixed/adam_raw36_s17_h0_e64.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({k: archive[k] for k in archive.files})
# prepared_color36 must come from the existing skin-region preprocessing contract.
# lab = predictor(prepared_color36)
```

The example is a frozen individual model, not a production recommendation. Reported quality averages seed errors; inference does not ensemble. Original TRAIN only966 rows/24 people, reused and overlapping roles; no fresh phone-face evidence. Full3 avoids the observed diag variance-floor failure but does not establish a better quality/cost frontier. Most selected networks lose to the exact FG norm_static controls. Model names and implementation do not establish patent novelty, trademark clearance or Skolkovo admission.

[Report](../benchmarks/chromaseed_gaussian_v1/report.md) · [Protocol and formulas](../research/chromaseed_gaussian_v1_protocol.md) · [Receipt](../benchmarks/chromaseed_gaussian_v1/verification.json).
