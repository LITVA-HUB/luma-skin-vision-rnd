# Luma ChromaSeed-G model card

Working research family: **Luma ChromaSeed**. G is a compact conditional residual on an exact 128-center Nyström color model. It is a candidate component for Luma's skin-color technology; the cosmetic e-commerce app is a demonstration/integration layer. It has no verified production or Skolkovo qualification status.

## Input and output

One `float32[36]` vector from an already prepared skin region: nine encoded-RGB quantiles (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99; each R/G/B), RGB mean, population RGB standard deviation, then RG/RB/GB correlations. Original pixel values use encoded sRGB in [0,1]. The definition is preserved in [color36](../../scripts/chromaseed.py), but that training module is not required by the standalone consumer. Consistent skin-region preparation is still an integration requirement.

Output: three native instrument-reference Lab values in the MSKCC dataset's D65/10° convention. Do not silently relabel these as another Lab convention. Input is neither a whole face nor a biometric identity. The component does not localize skin, solve arbitrary illumination, estimate ethnicity, diagnose skin or validate a product shade match.

## Mechanism and fitting

The same 128 kernel evaluations feed the shared prediction and its residual. A fit-side acquisition gate (37 stored FP32 coefficients) controls the residual sign/strength from current color features. Soft clips the score to [-1,1]; hard takes its sign, with zero going to +1. The gate has no query camera label, query Lab or batch adaptation. Hard routing can be discontinuous near score zero; robustness is a pending diagnostic.

The base is frozen during analytic residual fitting. Residual ridge penalties 0.1/1/10 and strengths 0/0.25/0.5/1 are chosen by three inner person-disjoint folds. Width factor1/base alpha0.1 are fixed. A single-camera fit makes gated families exact base aliases, including all cross-camera roles in this study. This is not an inference-time unknown-camera safeguard. The uniform residual is the equal-capacity ungated control and collapses into the original static coefficients.

## Measured scope

| Quantity | Perceptual hard gate, mixed role |
|---|---:|
| Mean held-person ΔE00, average of seed errors | 5.285104 |
| Exact shared base / uniform control | 5.390066 / 5.361655 |
| Fit / held rows | 734 / 232 |
| Fit / held people | 18 / 6 |
| Numeric payload | 21,973 bytes |
| Cached arrays in standalone NumPy predictor | 44,640 bytes |
| CPU response, prepared color36 | 11.7 µs median across seed medians |
| Full fit, seed17, median of3 after warmup | 35.53 ms |

AMD Ryzen9 7900X, one CPU thread, Windows11/NumPy2.5.3. Initialization, image preprocessing, I/O, imports and mobile hardware are outside response timing. The model byte count excludes Python/NumPy, temporary arrays, archive overhead and retained caller dictionaries. The standalone implementation caches FP64 calculation arrays while preserving the original FP32 normalization arithmetic.

Mixed gain is approximately 1.95% against the matching base; four of six people improve, and the descriptive fixed-prediction person-bootstrap range for the difference includes zero. Soft is close at5.295118. Cross-camera gated results remain the base's8.654805/8.386889 by exact fallback; a prior guided RBF remains better in reverse at7.7193. Repeated seed fits and historically reused overlapping TRAIN roles are not new independent people or an ensemble prediction. These results do not establish ordinary-phone facial accuracy.

## Local use

Representative stored seed17, not a deployment-selected winner: [weights](../../experiments/runs/chromaseed_gated_v1/selected/mixed/perceptual_hard_s17.npz). Other families, roles and seeds remain in the same selected-model archive. Published quality above averages three seed errors, whereas this file is one model.

From the worktree root:

```python
import sys
import numpy as np

sys.path.insert(0, "scripts")
from chromaseed_gated_numpy import Predictor

path = "experiments/runs/chromaseed_gated_v1/selected/mixed/perceptual_hard_s17.npz"
with np.load(path, allow_pickle=False) as archive:
    predictor = Predictor({key: archive[key] for key in archive.files})

# color36 is one finite vector from matching, independently validated preprocessing.
# lab = predictor(color36)
```

[Standalone consumer](../../scripts/chromaseed_gated_numpy.py) imports NumPy only. It validates fields, shapes, finite values, dtypes and scales. Its outputs reproduce all72 selected models' stored predictions to below2e-8 native Lab. This is numerical parity, not a guarantee about new input acquisition. The caller supplies the prepared feature vector; no camera labels or observed target are needed.

## Evidence and intended next step

23 primary numerical tests and3 separate portable tests pass. Independent direct/analytic/SVD auditing reconstructs72 selected models plus12 fixed positive-route probes; maximum native-Lab drift2.46e-5, below0.001. All72 P bases and864 single-camera fallback copies are exact. Primary, audit and runtime are terminal. Full results include all families and controls in the [report](../benchmarks/chromaseed_gated_v1/report.md), with [runtime](../benchmarks/chromaseed_gated_v1/runtime.json) and [verification](../benchmarks/chromaseed_gated_v1/verification.json).

No new data, downloads, images/tokens, legacy validation/calibration/test or external publication were used. Dataset and derived-weight commercial rights are not cleared merely by implementing this predictor; no commercial clearance is claimed. Working name has no trademark clearance claim. Next: fixed-model gate-margin and small-color-perturbation diagnosis, with base/uniform/soft/hard controls. A real independent ordinary-phone facial benchmark is still needed for the user goal.
