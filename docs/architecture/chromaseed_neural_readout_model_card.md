# ChromaSeed NR model contract

Prepared skin-region color36→native instrument D65/10° Lab3. Network d→64 ReLU→3; mean3 uses coordinates27/28/29, raw36 uses all36.451/2563 network parameters,1855/10564 numeric B including FP32 normalizers and optional uint8 indices. Every hidden payload is preserved during analytic fitting; hidden standardization folds into w2/b2 and adds no deployed arrays. All seven representation bases share this capacity. No uncertainty, iterative self-correction or dynamic links are added by NR.

Use the unchanged `chromaseed_gaussian_numpy.Predictor`: one finite shape(36,) vector per call, Lab3 output. It validates every coordinate including finite ignored features. `chromaseed_gaussian.predict` is a separate N×36 batch helper. Inputs follow the existing prepared encoded-sRGB statistics contract, not arbitrary image RGB. The old TG model-card batch statement has a separate bound erratum.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_gaussian_numpy import Predictor

path = root / "experiments/runs/chromaseed_neural_readout_v1/selected/mixed/norm_random_raw36_s17_a2.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({k: archive[k] for k in archive.files})
# lab = predictor(prepared_color36)
# batch_lab = np.array([predictor(row) for row in prepared_batch])
```

The example is an individual audited random-basis model, not a selected production release. Reported errors average individual seed errors, not ensemble inference. FP32-normalize then FP64 compute; cached arrays20816/3659 B, separate from process RAM. NPZ container size and training design/system/tensor costs are reported separately. Actual face/skin extraction, phone imagery, product-shade mapping and end-to-end latency remain unvalidated.

Original TRAIN966 rows/24 people only; three reused overlapping roles, camera/person confounding. Most NR comparisons and every frozen policy lose to FG. Analytic readout often improves the unchanged short-trained network, but that does not prove superiority over the strongest reference or patent novelty. Preserve all controls and source/input/receipt bindings.

[Report](../benchmarks/chromaseed_neural_readout_v1/report.md) · [Protocol](../research/chromaseed_neural_readout_v1_protocol.md) · [Receipt](../benchmarks/chromaseed_neural_readout_v1/verification.json) · [TG interface correction](../research/chromaseed_gaussian_consumer_erratum.md).
