# Basis-free color-subspace probe: a bounded TRAIN-only falsifier

Hypothesis: replace a selected color frame with the invariant subspace projector
P=X(X^T X)^-1 X^T for an N-by-3 matrix of spatial patch colors. Use P applied to
fixed spatial probes as compact features. For X->XM, invertible M, P is unchanged
in exact arithmetic. Predict spatial weights from those features and return
X^T w; this output transforms with M rather than discarding sensor response.
QR can apply P without forming the dense N-by-N matrix. This is standard linear
algebra and related to color homography/invariant representation prior art;
no novelty claim or universal physical-camera model is made.

First cheap test, before fitting another network: only the existing1126 TRAIN
images/GT from SimpleCube++, original CC BY4.0. No VAL/RISK/CAL/TEST decoding.
At4x4,8x8,16x16 patch grids, record full-rank support, color-matrix condition,
and invariance of QR-projected fixed16 probes under three fixed invertible
positive sensor transforms. A numerical support threshold condition(X)<=1e4
is diagnostic; it is not camera invariant and cannot be silently presented as
such. Rank failure is recorded, not dropped from population counts.

Positive common spatial weights constrain the illuminant to the cone generated
by observed patch colors. Compute its GT-oracle using nonnegative least squares
in original RGB and independently check KKT optimality. Because the cone allows
arbitrary positive scaling, its Euclidean projection gives the minimum angular
RECOVERY direction. Report that recovery lower bound. Reproduction angular error
at the projected direction is only the error of a feasible oracle solution;
it is NOT a certified lower bound on reproduction angular error. Never conflate
those quantities or call GT-assisted oracle outputs an inference method.

For context, measure TRAIN errors of the already frozen seed17 V7gt_native best
checkpoint on these same rows. That comparison diagnoses representational
capacity on training data; it does not estimate test accuracy or justify model
selection on held-out labels. If the positive cone is restrictive, a signed
weight head requires a separate positivity/rejection analysis before training.

Fixed probe seed17; FP64 geometry and NNLS; source-only comparison uses the
existing FP32 neural weights. All1126 rows, failure counts, numerical residuals,
predictions, input/model/script hashes and elapsed CPU time are retained. No new
learned model or external benchmark result is produced by this diagnostic.

Failure modes: low-rank scenes, weak invariant semantic information, stable
algebra that erases useful chromatic priors, saturation/nonlinear ISP, and
sensor spectra outside the source linear span. V3 already showed that exact
color-frame equivariance can lose accuracy. This probe removes frame selection,
not the need to falsify the assumption that stronger invariance is helpful.
