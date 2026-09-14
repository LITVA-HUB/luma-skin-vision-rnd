# ChromaSeed-G: compact acquisition-conditioned residual readout

Registered 2026-09-13 before fitting/selection. User-authorized autonomous compact-model research, same isolated worktree, no agents, publication or data acquisition. Previous D source/input/artifact and process state verified; previous goal turn classified progress. The entire high-quality compact/fast goal remains active.

## Scope and controls

Only original TRAIN `train.npz`, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Load color/target/patient/site/device; no images/tokens or legacy validation/calibration/test. Use frozen mixed, SLR-to-iPod and reverse roles and three inner person-disjoint folds. Roles are historically explored and overlap. No claim of fresh phone-face evidence or camera causality.

Reuse frozen P/KE exact128-center Nyström basis, width factor1, base alpha0.1, seeds17/29/43. Two bases: normalized-MSE and P shared constant perceptual metric. Every base payload must reproduce P's corresponding width1/alpha0.1 bank exactly; no old source is edited. These are fixed previously defined controls, not newly selected from G outer results.

Eight families: `norm_base`, `norm_uniform`, `norm_soft`, `norm_hard`, and the corresponding four `perceptual_*` families. Every correction family also has zero correction as a candidate, aliasing its exact base. Compare all families; do not drop a stronger previous reference when G loses.

## Conditional model and local training

Let k(x) be the single shared vector of128 kernel evaluations, base coefficients B, correction coefficients C and normalized Lab target y. The conditional answer is `k(x)B + rho*s(x)*k(x)C`, then the original Lab denormalization. This changes effective readout coefficients with the current input; it does not retrain at query time, grow topology, or repeatedly iterate until convergence.

Train one fit-side linear acquisition gate per two-camera bank. Input is the base normalized color36 coordinates. Standardize them using camera/person/site/image balanced weights summing1; solve weighted ridge alpha0.1 with unpenalized intercept for labels +1 SLR/-1 iPod. Collapse normalization into37 coefficients in the base coordinates, store FP32, and use those rounded coefficients during correction fitting and inference. `soft` uses score clipped to[-1,1]; `hard` uses +1 for score>=0 and -1 otherwise. No temperature, confidence threshold, probability calibration or gate hyperparameter search. The gate never consumes query acquisition labels, measured Lab or query-distribution statistics.

`uniform` uses s(x)=1 and needs no acquisition gate. It fits the same residual capacity and can collapse into the static base coefficient array at export. If a fit split has one acquisition group, `soft` and `hard` return the exact base payload for every candidate; uniform still fits its ordinary residual control. This fallback is trained-domain availability, not runtime detection of an unknown camera.

Keep the base fixed. Compute residuals against its actual stored FP32 coefficients. In the whitened shared basis Z, solve `sum_i w_i (s_i Z_i theta - residual_i)^T G (s_i Z_i theta - residual_i) + lambda*||theta||^2`. Skin-color fit weights are the original equal-person/site/image weights with mean1. G is identity for norm, or the fit-only shared P perceptual metric for perceptual. Use eigendecompositions of the weighted feature Gram and3x3 G to solve all residual penalties analytically; no global backpropagation or weight brute force. Collapse the residual through the existing Nyström whitening and store FP32 coefficients.

Residual lambda candidates0.1,1,10; strength rho0.25,0.5,1, plus the shared zero alias. The same coefficient solution is reused across strengths; these are not independent trainings. Base alpha/width and gate settings stay fixed. Uniform export stores FP32 `B+rho*C` in the original seven-array predictor. Active conditional export adds correction128x3, gate37, rho float32 and mode uint8 to those seven arrays (expected21,973 numeric bytes at128 centers, versus20,284 base). Measure actual size and arithmetic; do not infer latency from parameter count.

## Budget, selection and evaluation order

Each bank stores6 base and162 positive-lambda/strength configurations=168. Nine inner banks (1512 readouts) finish before24 family/role choices freeze. Each correction family has10 candidates including zero, each base1;186 family-candidate choices across roles before seed averaging. Select lowest inner native DeltaE00 person mean averaged across model seeds, with exact tie preference lower rho then higher residual lambda. Seeds are repeated fits, not independent people or an inference ensemble.

Then finish all three final banks (504 readouts) before evaluating72 selected single models. Total2016 stored configurations include864 exact one-camera routed fallbacks and72 preserved base controls. With fixed roles, expect360 nonzero residual coefficient solutions in120 shared-algebra solves,36 perceptual base solves, and4 two-camera gate fits. Count any divergence explicitly; never call stored fallback copies independent training. All24 choices freeze before any new outer score. Primary criterion remains person mean DeltaE00; retain p90/site/image summaries and paired exploratory comparisons to the matching base and uniform control.

## Verification and runtime

Tests first: exact P base preservation, zero aliases and single-camera routed fallbacks, ordinary residual still active without gate, eigensolve versus independent augmented SVD, gate score rounding/clipping/zero boundary, static collapse, prediction dependence on gate, grid/counts and invalid settings. Freeze core/runner/tests/protocol plus inherited sources, P bank/selection/results bindings and D source/verification before real fitting.

Audit every stored OOF prediction/selection, all72 P base payloads and864 fallback payloads. Independently reconstruct selected positive corrections using direct kernels, analytic P metric and augmented-SVD solves; independently fit the gate with augmented least squares. Also check fixed positive soft/hard probes in mixed inner fold0, both bases/all3 seeds/lambda0.1/rho1, so zero selections cannot evade dynamic-path testing. Require selected and probe native-Lab prediction drift <=0.001 against that independent analytic reference; report observed maxima. Source/hash/direct prediction checks cover all2016 stored models, while independent SVD refits are scoped explicitly.

After audit and without competing heavy jobs, profile all72 actual batch-one predictors (20 warmups,3 passes). For seed17 of all24 selected family/role choices, do one warmup and3 complete fits, including preparation/weights/gate/basis/readout, with exact payload checks. Also profile four fixed active mixed soft/hard candidates (both bases,lambda0.1,rho1) so inactive choices cannot hide dynamic cost. No input-image preprocessing is included. No extra quality search during timing.

Reports must separate selected performance from fixed diagnostic paths, conditional speed from fallback speed, numeric bytes from archive size, train-only loss reduction from held-person quality, and known-camera adaptation from unseen-camera generalization. No direct participant IDs/images in repository artifacts. Preserve failed candidates and all prior source locks.

Conditional experts and feature modulation are established ideas: [Jacobs et al.,1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf), [Perez et al.,FiLM](https://arxiv.org/abs/1709.07871). G is a small local residual-readout experiment, not a claim to invent mixture-of-experts or FiLM. No imported model/code/weights.
