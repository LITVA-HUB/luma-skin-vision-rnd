# Camera-subspace feasibility: measured TRAIN-only falsifier

The basis-free projector is numerically invariant under the three fixed linear
sensor transformations, but a nonnegative spatial-weight output is restrictive.
This is a representation diagnostic on 1,126 SimpleCube++ TRAIN images and real
illuminant GT, not a new trained model or a test-accuracy result. Original data
license: CC BY 4.0. No image is included in this evidence archive.

|Patch grid|Minimum recovery error, GT oracle °|Feasible oracle reproduction °|GT outside positive cone|
|---|---:|---:|---:|
|4 × 4|1.569883|2.062213|76.73%|
|8 × 8|1.027486|1.338250|53.82%|
|16 × 16|0.675077|0.880496|34.81%|

The positive-cone projection minimizes angular **recovery** error, with all
3,378 NNLS solutions checked against primal/dual feasibility, complementarity
and projection orthogonality. Its reproduction error is that of a feasible
GT-assisted solution; it is **not** a certified reproduction lower bound.
Maximum recorded KKT violation: 5.35e-16; no failed projections.

The frozen V7 seed17 GT-native checkpoint has TRAIN recovery mean 0.604205°
and reproduction mean 0.776971° on the same rows. Thus even a perfect selector
of nonnegative 16 × 16 patch weights cannot match that training recovery mean.
This does not rule out a positive-weight model generalizing better to new data.

For full-rank X with spatial rows and three color channels, P = X(XᵀX)⁻¹Xᵀ
is unchanged by X → XM for an invertible matrix M. The probe applies P to
16 fixed spatial vectors using QR without constructing dense P. All 1,126 rows
were supported at all three grids; maximum feature change across the fixed
transforms was 7.41e-13 in FP64. Condition-number median/p95/max at 16 × 16:
162.62 / 495.91 / 1,511.80. The support cutoff of 10,000 is itself not invariant.

Next hypothesis, **PLANNED**: derive signed spatial coefficients from invariant
subspace features, then predict Xᵀw with explicit positivity and conditioning
checks. Signed coefficients can leave the positive color cone. They also allow
unstable extrapolation, and exact invariance may discard useful reflectance
priors. V3 already demonstrated that mathematically exact color equivariance
can lose accuracy. A matched direct/positive/signed source screen is needed
before using more held-out camera data.

Neither a projector nor color invariance is novel by itself. Color homography
and illuminant-equivariant networks are established prior art. Linear mixing
does not cover arbitrary sensor spectral responses, clipping or nonlinear ISP.
This result does not establish camera independence or skin-color accuracy.

Elapsed probe computation: 3.85 seconds, excluding source loading and neural
comparator inference. This is not deployment latency.

[Frozen protocol](../research/projector_cone_probe_protocol.md),
[complete results](projector_probe/results.json),
[input/configuration bindings](projector_probe/config.json),
[artifact hashes](projector_probe/manifest.json).
