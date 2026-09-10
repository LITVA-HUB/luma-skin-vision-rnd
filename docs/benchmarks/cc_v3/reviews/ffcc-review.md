# Task 3 manual scientific/code review

Date: 2026-09-10. **PASS — no blocking findings in the inspected numerical core.** Scope: `scripts/cc_v3_ffcc.py`, `tests/test_cc_v3_ffcc.py`, and the pinned Google source/formulation in `docs/research/cc_v3_prior_art.md`. Manual source reading and CPU constructed probes only. No external source uploads, CodeRabbit, package installation, real-image/label access, GPU work, or training. This does not establish benchmark reproduction or MATLAB numerical parity.

Reviewed SHA-256:

- model `f451744a67ff4c2d58cb05c34d4af0689332c6acd830904b696c7dac27e16391`
- tests `cff65664008f6d3ab3648bfaa45fe468ad148c3151ea57eb201bf09e66d565ec`

`python -m pytest tests/test_cc_v3_ffcc.py -q` → **13 passed in 1.63s** in this independent review. The tests cover constructed formulas and gradients; they are not trained accuracy evidence.

## Source correspondence

The implementation is consistent with the selected shallow, pad-covariance path from original `google/ffcc` commit `2fa9e1316954dbd3913630b7d597927941b4dd32`:

- Coordinates are `(log G-log R, log G-log B)`, with u as histogram rows and v as columns. Input counts are unweighted and use nearest periodic bins, with MATLAB half-away-from-zero rounding. Each of the two histograms normalizes independently; empty channels remain zero.
- Feature channels match masked original RGB and eight-neighbor masked mean absolute differences with replicated image/mask borders. The all-channel minimum-intensity test excludes invalid logs and weak features. Default mask follows all-positive input pixels. Explicit masks alter histogram construction, while the source-compatible average RGB includes every original spatial pixel.
- The chosen float `[0,1]` full-scale interpretation is explicit, resolving the archived code's ambiguous `isa(im,'float')` branch without pretending integer-path parity. Resizing, black-level correction, saturation/chart masking and quantization remain caller responsibilities.
- FFT filtering is circular convolution with no forward conjugation, no fftshift, and one bias map. MATLAB-compatible forward/inverse normalization is used. No paper gain map is silently claimed: the pinned source's shallow `EvaluateModel.m` has bias but no gain.
- Decode uses both circular first moments, the source's rounded one-based covariance chart, all cross-covariance terms, diagonal padding in bin units and the correct `h²` unit conversion. Optional gray-world unwrapping uses log ratios of average linear RGB and MATLAB rounding. The RGB decode has the correct negative UV signs.
- The two target branches reproduce the source's actual behavior, including its misleading switch name: nearest corresponds to source `SMOOTH_CROSS_ENTROPY=true`, bilinear to false. Input histograms remain nearest in both cases.
- Gaussian UV NLL uses the unwrapped decoded residual, padded covariance and the documented constant subtraction. It is explicitly not an exact BVM likelihood or RGB angular loss. Cholesky implements the same positive-definite Gaussian objective more stably than explicit inversion.
- FFT regularization has the original quadratic smoothness spectrum from `[-1,1]/sqrt(8)`, L2 shifts, full Fourier-bin sum and data-mass factor. It is not L1 TV or AdamW. The default source coefficients are retained as initial defaults rather than asserted tuned values.
- The core has **12,288 real trainable parameters**: two 64×64 spatial filters and one 64×64 bias, initially zero. The source's packed/preconditioned Fourier optimization is not reproduced by these real spatial parameters; this difference is clearly documented.

## Independent probes beyond the supplied tests

These use independently expressed arithmetic and literal loops, with NumPy random generator seed 461 and CPU FP64. They do not execute the MATLAB reference.

| Probe | Evidence |
|---|---|
| Random 5×7 RGB and boolean mask; independently enumerate replicated eight-neighbor valid differences, minimum-intensity filtering and periodic votes | Both histograms matched exactly; each had 26 included pixels. |
| Arbitrary asymmetric two-channel 8×8 histogram/filter pair; literal four-index circular sum plus bias | Maximum FFT versus direct-convolution defect `6.77e-15`. |
| Nonseparable/correlated posterior; independently calculated trigonometric mean and one-based chart moments | Mean defect `1.11e-16`, covariance defect `1.06e-15`; nonzero off-diagonal covariance `0.007058674535191451` exercised. |
| Random maps with unequal nondefault filter coefficients; independent spatial Parseval expression | Objective defect `4.55e-13`; maximum parameter-gradient defect `1.42e-14`. |

The independent regularizer identity used was

```text
0.5*n² * sum_Q [lambda_Q/8 * (||Q-roll_u(Q)||² + ||Q-roll_v(Q)||²)
                         + shift_Q * ||Q||²].
```

This tests the otherwise easy-to-miss FFT/Parseval factor and both channels' separate coefficients, rather than repeating the frequency-domain implementation. The supplied gradcheck additionally covers the coupled FFT → softmax → mean/covariance → CE/Gaussian-NLL/regularizer objective away from chart switches.

## Explicit adaptations and residual limits

The following are disclosed limitations, not blockers for the labeled FFCC-inspired control:

- Uniform/vanishing-resultant posteriors are explicitly undefined below `1e-10`; finite diagnostic means and zero confidence are returned. The original code did not have this refusal policy. Therefore a zero-initialized model requires CE warmup before Gaussian NLL; an undefined circular mean cannot be treated as a valid likelihood target or accepted prediction.
- The model marks empty-feature rows invalid even if the learned bias alone can yield a defined posterior. `gaussian_uv_nll` checks defined means, not feature count; training/evaluation callers should preserve the intended sample policy rather than silently dropping troublesome rows. It is acceptable for the learned prior to exist, but explicit model validity must be honored for selective reporting.
- The core implements covariance padding and non-isotropic moments, not every optional original clamp/isotropy/deep/temporal path. Those omitted configurations must not be implied in result labels.
- Default filtering is float32 after FP64 histogram formation; `.double()` enables FP64 filtering. Gradchecks and independent formula probes establish the double path. No bitwise MATLAB parity is established.
- Decoding confidence measures concentration. It is not source-calibrated reproduction risk and does not grant a selective-error bound.
- Default regularizer coefficients are documented source initial values. Without the original preconditioning, tuned hyperparameters and optimizer, a poorly converged control cannot support a claim to beat published FFCC. Training should record optimizer, objective coefficients, convergence curve and source validation selection.
- License attribution and adaptation notice are present, referencing the retained Apache-2.0 source/license. No weights, datasets or minFunc adoption is inferred from that license.

## Related source-screen lock check

Also inspected `docs/research/cc_v3_source_screen_lock.md` at the root agent's request. No blocking scientific issue was identified: source-only fitting/validation scope and prior validation reuse are disclosed, loss-only point clipping is separated from evaluation validity, raw positivity gradients remain available, all-population validation selection requires at least 99% valid rows, and epoch zero is retained. Equal canonical-NLL coefficients still induce different geometry across direct/diagonal/full-frame modes; the camera Jacobian corrects reported density but does not eliminate this conditioning difference. The text describes it as a hypothesis-driven experiment rather than optimal tuning. Any post-screen revision needs the stated new run ID and preserved negative run, which is consistent with the protocol.
