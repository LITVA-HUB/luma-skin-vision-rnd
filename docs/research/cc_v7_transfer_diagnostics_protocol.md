# V7 transfer diagnostics, fixed before completed model comparison

The objective is one-image accuracy on cameras absent from estimator/risk
training, without camera identity, a CCM, target-camera calibration, a teacher
at inference or test-time adaptation. Scene features may be invariant; the
RGB illuminant output must transform with the sensor. Invariance of output
itself is the wrong objective. No universal unknown-ISP guarantee is implied.

First complete the five source arms at seeds 17, 29 and 43. Do not drop arms or
choose a seed using observed phone results. All checkpoints are selected by
the same native 119-row source validation mean, with final states preserved.
The current 79-reference Samsung/Oppo screen is already observed and cannot be
presented as a new untouched test of V7. It remains historical evidence.

A diagnostic screen uses the same 119 source validation images under six fixed
positive invertible RGB matrices: identity; diag(2,1,.5); diag(.5,1,2);
[[.9,.05,.05],[.05,.9,.05],[.05,.05,.9]];
[[.65,.25,.10],[.10,.70,.20],[.20,.10,.70]]; and
[[.4,.4,.2],[.2,.5,.3],[.3,.2,.5]].
The last matrix exceeds the augmentation's mixing range. Apply each matrix to
pixels and GT, normalize only the GT; do not clip transformed pixels. Measure
real-source-GT reproduction error, p95, over10-degree fraction, and consistency
between f(Mx) and normalized M f(x). These are transformed-source diagnostics,
not measurements on real new camera hardware. The original input cache has
already undergone masking, thumbnailing, exposure scaling and clipping;
augmentation cannot undo that preprocessing or synthesize a new full ISP.
No stress result may select a checkpoint or rewrite this matrix list.

Compare V7 best/final states and the preserved V2 direct/SoG controls for the
same seed. A zero equivariance error is not proof of a correct illuminant.
The SoG control is analytically equivariant to diagonal gains within its
numerical validity domain, making it a useful check of the measurement code.

Next real transfer population: the already acquired, still pixel/GT-value
unused INTEL-TAU remainder. Use the 317 rows whose exact reference-file hash
does not occur in historical V2: 103 Canon, 112 Nikon, 102 Sony. Retain all 384
as a prespecified sensitivity population. Exact reference hashes are proxy
groups; unequal hashes do not establish physical-scene independence. The
dataset's original CC BY-SA 4.0 governs; here use it for evaluation only, with
the existing sparse-mirror provenance limitation disclosed. All model/risk
training stays on CC BY 4.0 SimpleCube++ and the separately licensed teacher.

Before decoding that population: freeze exact image/GT identity lists,
preprocessing, every estimator/selector/checkpoint hash and numerical source,
after source-only risk-head fitting. Compare all 15 V7 runs plus frozen V2
controls, classical algorithms and compact statistics/Fourier controls. Risk
heads must use the existing held-out RISK role and disjoint CAL role, with the
same standard feature/calibration budget. Report fixed coverages and realized
coverage under the source-calibrated threshold, not only a test-ranked curve.
Do not use the new external scores to choose or revise the reported method.

Any gain is an empirical property of these declared camera/data protocols.
Source-only improvement, virtual matrix consistency, teacher semantics and
lower parameter count cannot establish facial colorimetry, arbitrary JPEG/
HEIC support, generalization to all cameras, patent novelty or a risk guarantee.
