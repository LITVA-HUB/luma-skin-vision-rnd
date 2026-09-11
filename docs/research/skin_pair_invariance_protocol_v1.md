# Source-only paired-capture invariance experiment

Previous goal turn was PROGRESS: an immutable independent skin test and negative/
partial method evidence were produced. This new experiment does not load its
400-image TEST or208-image CALIBRATION caches, references or predictions.
Existing source validation has been reused in prior research; all new scores
are exploratory, not fresh independent confirmation.

## Mechanisms and falsifiers

Invert the independent-image assumption. TRAIN has248skin sites,232with four
capture modes,8with two,7with three,1with one. All images of each TRAIN site
have exactly the same actual instrument label. The same site is not a pixelwise
registration or a paired camera-model intervention. Learn capture invariance
without requiring a second image, site identity, capture mode or camera metadata
at inference.

A: output consistency. Penalize disagreement between predictions for actual
same-site captures, alongside true Lab supervision. Assumes differences arise
mostly from capture; can work by reducing capture-sensitive estimation variance.
Failure: averaging away real footprint differences or increasing color bias.
Cheapest falsifier: same model/sampling/budget with and without the pair loss.

B: joint representation consistency. Add a training-only64dim projection of
the512dim scene context and a variance-invariance-covariance objective. Assumes
a useful shared representation exists. Collapse or loss of necessary capture
information is the likely failure. Compare to output-only and plain models.

C: hard nuisance quotient. Standardize18patch descriptor channels from TRAIN,
estimate within-site covariance of image-mean token descriptors, and remove its
top3eigen-directions before the network. Assumes capture variation lies in a
small subspace distinct from true color. This is an intentionally severe,
testable inversion: remove the network's access to major capture directions.
Failure: those directions also carry real skin color. A standardized full-rank
control separates projection from mere scaling. No spectral/physical uniqueness
or camera independence follows from this empirical feature operation.

## Fixed source screen

Five arms: raw paired-sampling baseline; standardized full-rank control;
hard quotient3; raw output-consistency0.5; raw output0.5 plus projected-context
VICReg0.01*(25invariance+25variance+covariance). VICReg uses epsilon1e-4,
variance floor1, unbiased covariance, and mean squared off-diagonal covariance
sum divided by feature count. No pretrained weights or copied external code.

All use the existing924932parameter plain PatchVotes base, seeds17/29/43,
80epochs, AdamWlr.001/wd.01, cosine end.00001. Per step sample16TRAIN sites
uniformly and two different images per site when available (singleton repeated).
31steps/epoch = ceil(966/32), identical paired sampling across arms. Base
initialization matches per seed. Supervised standardized-LabMSE averages all32
images. Output consistency uses standardized predictions, mean squared pair
difference. Context objective applies to all pairs; true labels anchor outputs.
VICReg's Linear512->64 projector is training-only, with32832extra parameters.

Training target scales use24TRAIN people only. Standardization/quotient fitting
uses only their inputs and site groups. Best epoch is selected on the original
6VALIDATION people using patient-balanced meanDeltaE00. Record all80epochs,
best/final checkpoints and exact replay, per-device/type errors, true-color
error and pairwise predicted-color disagreement. Improved repeatability alone
does not establish accuracy. No synthetic targets, arbitrary RGB augmentation,
or conversion of angular error to DeltaE00.

## Exploratory camera-held-out check

After all mixed-camera source fits, compare raw paired baseline and the lowest
three-seed-average source patient-mean arm among the other four (tie by arm name).
Even if none beats the baseline, retain the negative and test the best challenger.
Refit from scratch for each seed/direction. Use ONLY TRAIN SLR8people or TRAIN
iPod16people to fit all parameters/preprocessing. Choose epoch using the three
VALIDATION people with the same training camera. Report the other camera's
three VALIDATION people only after fitting. Counts and shared/source exposure
must be explicit: this is camera-held-out fitting on previously studied source
people, NOT a new untouched final test. No held-out camera pixels, labels,
normalization statistics or pair covariance enter fitting/epoch selection.
Use ceil(training_images/32) paired steps/epoch. All choice rules above are
fixed before running new models; retain all12camera fits and both directions.

## Prior-art boundary (fresh check2026-09-11)

Pair consistency and removing nuisance directions are not new by themselves.
[VICReg](https://arxiv.org/abs/2105.04906) combines representation agreement,
variance and covariance regularization. [Slow feature analysis](https://pubmed.ncbi.nlm.nih.gov/11936959/)
learns invariant features from changing observations. [Orthogonal signal
correction](https://www.sciencedirect.com/science/article/pii/S0169743998001099)
removes unwanted variation in spectroscopy; our empirical within-site PCA is
not a reproduction of that supervised algorithm. Contemporary
[LeJEPA](https://arxiv.org/abs/2511.08544) studies joint-embedding learning with
Gaussian regularization; this experiment does not reproduce its theory or
SIGReg. The transferable idea is controlled shared representations, not an
unsupported claim that modern large-model techniques guarantee skin accuracy.

The narrow question is whether real paired skin measurements permit a compact
single-image estimator with a better direct color/capture tradeoff. A positive
source result still requires separate future independent facial-phone evidence.
