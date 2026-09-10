# CC v3: color-frame posterior architecture — experimental specification

The user requests a fundamentally different neural architecture, continued experiments and maximum measured improvement. This is an unverified research design, not an assertion of novelty or superior accuracy. Preserve milestones2685bf0,7637d6d,ea490d3 and all old data/outputs. New branch `codex/cc-color-frame-research`; all new executable code under `scripts/cc_v3_*` to preserve the old source fingerprints.

## Question and mechanism

V2 learned a residual in diagonal-anchor-normalized RGB. V3 instead constructs an image-dependent **three-vector color frame**, processes invariant spatial/color relations with a small graph network, predicts a directional illuminant posterior in that frame, and transports posterior hypotheses into current camera RGB before estimating correction risk. No CNN backbone, pretrained weights, foundation model, camera identity, CCM or cross-image test-time adaptation is required.

Let x_p be column RGB. Form16 fixed4×4 spatial patch means. Choose the lexicographically first triple maximizing absolute determinant, and place it as columns of B(x). For an invertible linear color transform T and unchanged selected triple, B(Tx)=T B(x), because every candidate determinant scales by det(T). Then u_p=B(x)^−1 x_p is invariant. A network predicting canonical direction q(u) produces an unnormalized illuminant B(x)q(u), which transforms by T. Final normalization gives projective equivariance. This is an algebraic specialization of known canonicalization/frame ideas, not a new theorem. Real camera spectral responses need not be related by one linear matrix; clipping, offsets, noise, rank degeneracy and floating-point tie changes limit the claim.

Use FP64 frame selection/solve for the initial correctness study, then assess whether FP32 is numerically adequate before deployment work. Scale each input by a common positive per-image scalar before determinant tests, record singular values/condition diagnostics, and refuse rank-deficient frames. Do not silently call pseudoinverse/fallback a full-rank equivariant result. Finite Gray World fallback predictions may support all-population diagnostic errors, but fallback rows are invalid for Proposed selective acceptance.

## Neural architecture

Convert canonical pixels into64 fixed8×8 spatial tokens. Features: canonical channel means, population standard deviations, mean asinh(canonical channel), and2 normalized spatial coordinates (11 channels). Process with4 learned edge-gated diffusion blocks on an8-neighbor spatial graph, width192 initially, global pooling and64-dimensional context. Gate/value/update modules operate on node embeddings and relative canonical/spatial differences. No convolutional backbone is used. Compactness is measured, not assumed.

Provide strictly capacity-matched modes: `direct` uses common-scale input with identity frame; `diagonal` uses the GW diagonal frame; `frame` uses the max-volume full frame. The graph and output heads must be identical in all modes. Extra data/compute cannot be mistaken for a mechanism effect.

The point head is zero-initialized around the canonical GW direction. Predict8 mixture directions on S² with learned logits and positive concentrations. Use a stable von-Mises–Fisher mixture negative log likelihood on normalize(B^−1 GT) plus camera-space reproduction loss. A separate positivity penalty prevents negative mapped point predictions; any still nonpositive point is explicitly invalid, rather than hidden by clipping. Invalid input/frames produce finite positive GW diagnostics and valid=False.

Posterior risk transport maps canonical hypotheses through B before applying reproduction-angle geometry. Canonical uncertainty itself is not a camera-independent reproduction risk: reproduction error is invariant to jointly applied diagonal gains but not general channel mixing. Include concentration-dependent deterministic tangent quadrature, its validity mass and the transport-risk proxy. A source-held-out error head can calibrate this proxy later; do not label raw posterior spread as a guaranteed expected-error bound. Invalid posterior mass must not silently disappear through renormalization.

Required model API: `ColorFramePosteriorNet(mode='frame', width=192, layers=4, hypotheses=8)`; forward(NCHW nonnegative floating RGB) returns dictionary with `pred` Nx3 positive finite diagnostic prediction, `context` Nx64, `valid` N, `frame` Nx3x3, `canonical_mean` Nx3, `directions` NxKx3, `logits` NxK, `concentration` NxK, `transport_risk` N, `invalid_posterior_mass` N and frame-conditioning diagnostics. `canonical_target(frame,gt)` and `posterior_nll(output,gt)` expose independently testable likelihood calculations. An unmasked all-population reproduction loss and explicit valid-only likelihood/positivity handling belong in the experiment runner.

## Data and validation

First test numerical identities, failures and source-only real-data feasibility using the original1126 training and119 validation images. Stream only selected NPZ rows with the frozen helper; do not inspect official462/Sony30/INTEL384 errors during architectural selection. Those old tests remain regression evidence only.

For a later multicamera experiment, original INTEL-TAU CC BY-SA4.0 now has direct official confirmation. V2's evaluation-only use remains immutable. A separate V3 training track may adopt **previously unused** images under original BY-SA terms, with separate model/data-rights records and no assertion that those weights have an unrestricted proprietary distribution license. This revises a project-level V2 usage choice, not the user's ban on research-only/noncommercial production dependencies. No images or derived trained weights are published. All V1/V2 test images are excluded from new fitting by exact identity. Before any new image/GT decoding, freeze the remaining128 entries per camera from the existing256 metadata-only manifests, after removing the used128. New LOCO folds must keep the outer camera absent from all fitting and calibration. True physical scene grouping requires auditing; same-reference hashes are only proxies.

Do not delay the architecture source feasibility tests for this acquisition. If the frame construction loses, preserve the failure and revise the design instead of forcing a novelty narrative. A mathematically elegant restriction may remove essential information.

## Strong baseline and prior art

Reproduce the compact FFCC formulation from the original Apache-2.0 Google implementation/paper, recording each deviation, code provenance and input/protocol differences. No supplied pretrained weights/data are adopted. Keep classical and matched graph direct/diagonal controls and V2 references. FFCC posterior uncertainty, known frame/canonicalization architectures and prior color invariants narrow novelty. Search absence is not proof of an invention.

## Acceptance evidence

Numerical finite/gradient/validity tests; stable full-GL transform identity on well-conditioned constructed inputs; deliberate rank/negative/near-tie failure cases; posterior likelihood and mapped-risk tests including non-diagonal risk changes. Real source learning curves and validation predictions for all modes at the same budget; source-only locked selection before new outer-camera evaluation. Then recovery/reproduction summaries, all six fixed coverages, full curves/tails, held-out residual calibration,3 seeds for confirmed comparisons, and paired uncertainty. Only positive real evidence justifies export/optimization. Facial Lab/ΔE00 and Skolkovo/patent novelty remain unvalidated.
