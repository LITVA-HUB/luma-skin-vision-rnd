# Luma ChromaSeed-K v1 — research model card

**Purpose.** Compact regression of instrument-referenced native Lab from prepared skin-region color statistics. Research family: Luma ChromaSeed. Engineering follow-up candidate: **ChromaSeed-K128-N**, `nys_rpchol`, rank128. This designation follows exploratory engineering comparisons; it is not an independently validated production winner.

**Inputs.** A float32 array `(batch, 36)` of encoded sRGB region statistics: nine per-channel quantiles [.01,.05,.1,.25,.5,.75,.9,.95,.99], mean3, population standard deviation3, and correlations RG/RB/GB. Region selection, face localization, camera handling and image preprocessing are external. The series consumes the existing cache; it never reads its image array. `scripts/chromaseed.py::color36` implements the prepared RGB-sample representation.

**Outputs.** Three native Lab coordinates per example. The model does not identify a person, locate their face, infer ethnicity or provide a calibrated confidence estimate. Smartphone facial color accuracy and cosmetic product shade matching are unvalidated.

**Architecture.** Fit-only normalization → Gaussian kernels against128 stored landmarks → three linear channel readouts → inverse target normalization. Landmark selection uses randomized pivoted Cholesky of the person/site-weighted fit kernel, without labels. Nyström regression uses a whitened landmark span and ridge regularization. Width multiplier and ridge alpha are chosen on inner person-disjoint folds.

**Numerical storage.** 4608 center scalars +384 readout scalars +78 normalization scalars +1 width scalar =5071 FP32 scalars, **20 284 bytes**. Landmarks are stored feature vectors, not 4608 gradient-trained neural parameters. NPZ headers, Python/NumPy dependencies, FP64 runtime copies and temporary matrices add overhead. This K-series did not test FP16/INT8 payload compression; previous neural precision results do not apply automatically.

**Arithmetic/runtime.** FP32 normalization produces canonical coordinates, then kernels and readout products execute in FP64. Mixed-role warm CPU batch-one mean of per-seed medians16.50µs, Ryzen9 7900X, one thread. Includes normalization, excludes loading and image-to-feature computation. Standalone fit67.77ms on734 prepared rows includes balancing, normalization, exact distance median, full fit Gram matrix,128 landmarks and ridge solve. It excludes parameter search and persistence. The implementation currently materializes N×N fit arrays; million-example scaling is not demonstrated.

**Related variants.** Random/greedy landmarks,16/32/64/128 ranks, full weighted KRR, teacher projection, adaptive prefix projection, convex corrections with prior guided RBF. Teacher projection at128 stores the same20 284B but costs102.87ms with a freshly fitted CUDA teacher. Adaptive stores27 508B and returns at128 for every evaluated input under all three selected policies; no speed advantage. Its diagnostic bounds teacher approximation, not instrument error, and is not a formal floating-point certificate.

**Training/evidence.** Only previously acquired original TRAIN:966 records/24 people, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Mixed18fit/6held, SLR→iPod8/16, reverse16/8; roles overlap and are historically exposed. Seed repetitions are not new people. Legacy validation/calibration/test excluded. No synthetic palette or third-party pretrained weights in this series. Existing attribution and derived-artifact obligations remain: [MSKCC provenance](../ip/skin_mskcc_provenance.md). Code, dataset and derived-model rights are separate.

K128-N person-balanced native ΔE00:5.4387 /8.5970 /8.7050 in the three roles; full kernel5.3984 /8.5845 /8.9125. Earlier guided RBF5.8013 /10.0280 /7.7193 remains stronger in reverse transfer. Ordinary smartphone faces and scientific novelty remain unvalidated. Family-wide results and uncertainty: [report](../benchmarks/chromaseed_kernel_v1/report.md).

**Example, explicit fixed seed17.** This chooses an artifact by role/variant/seed for reproduction, not by picking the best outer score. Never feed a raw RGB triplet in place of color36.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed_kernel import predict_kernel

path = root / "experiments/runs/chromaseed_kernel_v1/selected/mixed/nys_rpchol_k128_s17.npz"
with np.load(path, allow_pickle=False) as artifact:
    model = dict(artifact)
# color_statistics: finite float32 (batch, 36), matching the frozen schema.
lab = predict_kernel(model, color_statistics)
```

**Lifecycle.** Keep the frozen primary protocol, sources, normalization and selected artifacts intact. Any sampled-bandwidth or matrix-free implementation belongs to a new prospective protocol. Do not automatically replace the Luma application backend, use the exposed roles as fresh confirmation, or apply numerical bounds as a guarantee of true skin color. Working name, trademark availability not checked. No series job remains; broad research goal remains active.
