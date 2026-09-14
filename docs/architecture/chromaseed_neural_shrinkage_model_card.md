# Luma ChromaSeed NS: stronger regularization of the compact readout

Research family: Luma ChromaSeed. NS changes only the selected analytical output-layer penalty on the exact NR representations. Architecture and consumer remain d→64 ReLU→3. raw36:2563 learned parameters,10564 numeric bytes including normalizers; mean3:451 parameters,1855 bytes including normalizers/indices. These are array payloads, not NPZ size, process RAM or full application footprint.

Input is exactly one finite36-element vector of the project's encoded-sRGB skin-region statistics. mean3 selects positions27,28,29 internally. Output is native instrument Lab, D65/10-degree convention. Full image/face detection, skin-region selection and statistics extraction are separate and are not included in microsecond inference timing. This model does not identify a person.

Use the already verified NumPy consumer `chromaseed_gaussian_numpy.Predictor`. It takes one vector, not a batch. An example exported NS model is `experiments/runs/chromaseed_neural_shrinkage_v1/selected/mixed/norm_adam_e16_raw36_s17_a3.npz`.

```python
import numpy as np
from chromaseed_gaussian_numpy import Predictor

with np.load(model_path, allow_pickle=False) as saved:
    model = {key: saved[key] for key in saved.files}
predictor = Predictor(model)
lab = predictor(np.asarray(color36, dtype=np.float32))  # shape (36,)
```

Do not supply a photograph or arbitrary RGB triplet to this interface. For a batch, call the consumer once per36-vector or use the separate validated batch helper. The earlier TG batch-interface wording is corrected in [the preserved erratum](../research/chromaseed_gaussian_consumer_erratum.md).

Training: original966-row/24-person TRAIN only, exact NR bases; readout alpha.1/1/10/100/1000 chosen through person-disjoint inner folds. Three reused/confounded outer roles are exploratory. All12 frozen NS policies lose to corresponding FG;10/84 heads change alpha, five improve and five worsen versus their old NR head. No more compact or generally more accurate model is established by this regularization extension. Never call an exposed role a fresh independent test.

The cached sweep does not pay for hidden representation training; the separate270 full construction checks do. For learned bases, all selected representation epochs and preparation are included. All270 exports matched exactly. Error values are DeltaE00, not percentages. Storing Gaussian variance or Adam moments is unnecessary for inference, but their costs belong to training-state accounting. Current implementation timings are one CPU thread, not RTX4060/phone benchmarks.

[Full comparison](../benchmarks/chromaseed_neural_shrinkage_v1/report.md) · [Protocol](../research/chromaseed_neural_shrinkage_v1_protocol.md) · [Next decision](../research/chromaseed_neural_shrinkage_next_decision.md).
