# Task 1 manual scientific/code review

Date: 2026-09-10. Scope: `scripts/cc_v3_model.py`, `tests/test_cc_v3_model.py`, `docs/research/cc_v3_spec.md`, and the v3 prior-art audit. This is an independent manual review. No CodeRabbit, installation, source upload, real-image/label access, GPU work, or model training was used. CPU constructed probes only.

Reviewed post-fix SHA-256: model `e4fdabf2ef60fe818133184b7a76a2fee1e4b99c1900f830661a14212f6dad88`; tests `b1395418e80a0d7a0af7dead784ba4b0324c2afca7aa5fcc9206a449492e7a46`.

Final verdict: **PASS after both P2 findings were fixed and independently rechecked; no unresolved blocking findings in this scope.** Initial tests passed (16), but manual probes exposed the two issues recorded below. After the core implementer's fixes, `python -m pytest tests/test_cc_v3_model.py -q` → **18 passed in 1.56s**. These are constructed numerical results, not color-constancy performance evidence.

Independent post-fix probes used a non-diagonal frame and GT scales `1e-300` and `1e300`, plus opposing GT/frame scales `1e-200/1e200` and `1e200/1e-200`. Canonical direction defects were at most `2.22e-16`; canonical and camera NLL defects were zero in these probes. The complete float32 aligned vMF component at kappa=1e20 now returns `-44.21382523` against analytic `-44.21382479`; a `1e-10` tangential offset adds exactly 0.5 in the probe. Against the elementary analytic density at kappa 0.999999, 1, 1.000001, 10 and 100, maximum discrepancy was `1.20e-14` in FP64. Fixes remove common magnitudes before solve/norm/Jacobian and combine the large-k density with a squared-chord angular term. The historical findings below explain why the added regressions matter.

## Findings

### Resolved P2 — canonical_target depended on arbitrary GT magnitude and could return a nonunit target

`canonical_target` solves against unscaled positive finite GT, then calls `F.normalize` with the default absolute epsilon. Positive direction labels have an irrelevant scalar, but this implementation changes the sphere target below that epsilon and can overflow its norm for large finite magnitudes. With identity frame, GT `(0.4,0.8,0.5)`, an aligned one-component posterior and kappa=3:

| GT scale | Returned target norm | Canonical NLL | Camera NLL |
|---|---:|---:|---:|
| 1 | 1 | 0.73678295 | 0.73678295 |
| 1e-15 | 0.001024695 | 3.73370886 | 24.38378945 |
| 1e-100 | 1.024695e-88 | 3.73678295 | 611.54606225 |
| 1e200 | 0 | 3.73678295 | infinity |

This violates the advertised positive finite GT interface and projective density semantics, even though typical normalized benchmark GT is unaffected. Normalize magnitude robustly before the solve and normalize the solved direction without an arbitrary absolute-scale floor. Add a regression asserting direction, canonical NLL and camera NLL invariance across finite GT scales. This is a mathematical interface issue, not an observed real-data failure.

### Resolved P2 — large-concentration stability did not extend from log normalizer to component likelihood

`vmf_log_normalizer` is stable in isolation, but `posterior_nll` computes `logC(k)+k*dot` in separate terms. For an aligned float32 component with k=1e20, cancellation gives NLL 0 rather than approximately `log(2*pi)-log(k) = -44.2138248`. The existing large-k test only differentiates the normalizer and misses the complete density. Use a stable combined log density, e.g. the regular branch `log(k)-log(2*pi)-log(1-exp(-2*k))+k*(dot-1)`, with a numerically justified dot-domain policy. Test the full aligned-component NLL, not only finite normalizer values. The network does not cap concentration; therefore the helper's claimed stability should extend to this regime or the accepted range should be explicit.

## Checks that passed and interpretation

- Frame selection enumerates all 560 ordered-by-combination triples of 16 spatial patch means, uses absolute determinants, and uses the first exact maximizer. The frame places patch vectors in columns. This is the intended max-volume construction.
- Image scaling, candidate determinants, singular values and linear solves use FP64. The common exposure scale cancels from canonical coordinates; the returned frame is in scaled-image coordinates. Projective camera transport is correct on jointly accepted cases.
- Additional manual reflection/permutation probe (negative determinant) gave zero context/prediction defect. An admissible signed mixing matrix gave maximum context defect `6.66e-16` and prediction defect `1.11e-16`. Existing tests also cover positive mixing and diagonal gains.
- A separate common-exposure `1e200` probe gave zero prediction defect and context defect `3.89e-16`. An initially attempted naive expected-vector normalization overflowed; that was a review-probe oracle issue, not a model failure, and was replaced by direct scalar-exposure comparison.
- Conditioning refusal is explicit. A diagonal transform with gain ratio 1e7 changed both rows from frame-valid to refused, as expected. Numerical acceptance is not GL-invariant; this must remain separate from the equivariance claim. No pseudoinverse or silently jittered frame solve is present.
- Invalid frame rows use an identity for safe downstream computation and report `frame_valid=False`; returned diagnostics retain original singular values/indices. Thus `frame` is the safe computational frame, not necessarily the rejected candidate. Consumers must use `frame_valid` when interpreting it.
- All-zero images in direct, diagonal and frame modes yielded positive finite diagnostic predictions, risk exactly 90, invalid posterior mass exactly 1. All-invalid canonical mean NLL raises, while per-row output marks exclusions as NaN. Point invalidity does not discard a valid-frame likelihood.
- A deliberately negative point was refused without clipping `raw_pred`. A positivity penalty on `raw_pred` retained a finite nonzero point-head bias gradient (norm `1.61e-4` in the constructed probe), despite diagnostic `pred` using fallback. The runner must actually use this raw-output penalty; this review did not audit runner integration.
- The default direct/diagonal/frame networks each have exactly **1,215,535 trainable parameters** with identical parameter shapes. Compactness still needs measured runtime/memory; equality of parameter counts does not equal equal numerical conditioning or equal frame-computation cost.
- The ordinary/near-zero vMF normalizer matches the S² density and numerical integration tests. The camera Jacobian sign is correct: `NLL_camera=NLL_canonical+log(abs(det B)/||Bq||^3)`. The current mathematical measure is the full sphere, not the positive-octant-conditioned density.
- Quadrature uses the correct vMF axial inverse CDF and tangent azimuths. Weights sum to one per component, concentrations affect the nodes, and invalid hypotheses retain their original mass with a stated 90-degree diagnostic cost. This is low-order quadrature, not exact integration or a physical error expectation on negative colors. The documentation correctly acknowledges orientation/boundary artifacts.
- Risk transports hypotheses before reproduction-angle evaluation. Tests demonstrate diagonal invariance and a changed risk under non-diagonal mixing. For invalid point predictions, risk describes spread about the diagnostic fallback; downstream acceptance must honor `valid=False`.
- The model is a graph after canonicalization with an unconstrained backbone, an established architecture pattern. There is no inspected claim that algebraic correctness proves novelty, camera invariance, real accuracy, calibrated risk or facial color accuracy.

## Residual limitations and useful next evidence

Numerical tests cover designed well-conditioned scenes; they do not establish stability close to the condition cutoff or real prevalence of low-rank patch frames. The 32-node quadrature can miss small positive/invalid regions, especially under strongly mixing/ill-conditioned frames; risk calibration must treat it as a feature. Future tests should compare quadrature to denser constructed sphere integration before presenting precise invalid-mass estimates. Source calibration and acceptance masking require an independent runner audit.

An additional exact-reference probe quantifies this quadrature limitation: for a uniform sphere (`kappa=0`) with identity transport, the true invalid-positive-octant mass is `7/8=0.875`, independent of the nominal vMF mean. The 32-node rule returned 0.9375 about `(1,0,0)` and 0.90625 about normalized `(1,1,1)` or `(1,2,3)`. Thus observed invalid mass can have several percentage points of orientation error in diffuse regimes. This agrees with the documented approximation and is not an additional blocking defect, but acceptance/calibration documentation should call the quantity a quadrature estimate rather than an exact posterior probability.
