# Luma ChromaSeed v1 — research model card

**Purpose:** compact regression of native instrument Lab from a prepared skin-region color36 representation, studying synthetic-palette initialization and frozen representation transfer. This is a research family with explicit controls, not a selected production release.

**Architecture:** fixed physical normalization →36→64 SiLU→3, plus36→3 linear skip.2671 learned scalars +78 normalizer scalars,10996 numeric bytes FP32. NPZ files additionally store metadata/container headers. The common architecture deliberately matches all controls. Library/runtime, face localization and feature extraction sizes are excluded.

**Inputs:**36 statistics over encoded sRGB samples in[0,1]:9 RGB quantiles[.01,.05,.1,.25,.5,.75,.9,.95,.99], mean3, population std3, upper correlations RG/RB/GB. Input region selection is external. `scripts/chromaseed.py::color36` computes this representation from batches of RGB samples; it does not locate a face.

**Outputs:** three native Lab numbers. No confidence calibration, identity recognition, ethnicity inference, diagnosis or demonstrated cosmetic shade-match outcome. Native target follows the existing MSKCC cache; synthetic pretraining uses canonical D65 Lab and is a deliberately limited proxy for that target.

**Training:** locally generated synthetic color surfaces; original, already acquired MSKCC CC-BY TRAIN-only data and instrument references for adaptation. No pretrained third-party weights and no new face datasets. Existing attribution duties remain: [MSKCC provenance](../ip/skin_mskcc_provenance.md). Synthetic renderer and new implementation are local; dependency licenses remain separate.

**Variants:** scratch, clean_palette, rendered_palette, shuffled_palette negative control, skin_long compute control; a separate experiment uses frozen random/clean/rendered/shuffled hidden layers with analytic heads. Each seed is a separate model. Do not present seed averaging of metrics as an inference ensemble.

**Evidence:** limited mixed-camera improvement over short scratch (5.959→5.760 ΔE00 for clean palette), approximately tied with longer real-only training (5.779), and no robust cross-camera transfer gain. Simulated-camera pretraining worsened reverse transfer to13.420 versus scratch11.508. All outer groups are historically reused and overlap across protocols; no fresh independent validation or ordinary-phone face guarantee.

**Model loading example:** choose the protocol/arm/seed explicitly; do not select a seed from outer error. The following shows fixed seed17, not a claim it is the best model.

```python
import sys
from pathlib import Path
import numpy as np

root = Path(r"C:\Users\dimal\Documents\просто\.worktrees\luma-local-search")
sys.path.insert(0, str(root / "scripts"))
from chromaseed import color36, predict

path = root / "experiments/runs/chromaseed_v1/mixed/final/clean_palette_s17/step1024.npz"
with np.load(path, allow_pickle=False) as artifact:
    model = {key: artifact[key] for key in artifact.files}
# samples: float RGB, shape(batch, pixels, 3), from already selected regions.
lab = predict(model, color36(samples))
```

**Lifecycle:** do not replace previous research candidates based on this run. Future simulator changes need a new frozen version, fit-only choices and new validation evidence. Keep all previous losses and controls. The name is provisional; trademark availability has not been checked.
