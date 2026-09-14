# ChromaSeed-P: perceptual readout experiment

2026-09-13. Prospective local experiment, written before any P-series real-data fit. The previous KF/KE turn is verified **progress**: source/artifact integrity receipt passes, all study processes are terminal. Broader user goal remains active. The user authorized autonomous experiments; no delegation, publication or dataset acquisition is part of this study.

## Question and boundary

Can perceptual weighting or iterative readout correction improve the compact exact KE128 predictor without enlarging its stored numeric payload or inference graph? Preserve all prior code/protocol locks, especially KE-bound pyproject.toml/uv.lock. Use new files only. This tests training-time correction; it does not claim new recurrent inference or identity recognition.

Only original TRAIN `train.npz`, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Read color36, target native Lab, patient/site/device only. No image/token arrays or legacy validation/calibration/test. Same mixed, SLR→iPod, reverse roles and three person-disjoint inner folds as frozen KF; equal-person/site/image training weights. Roles overlap and have been repeatedly explored. Camera transfer also changes people and color distributions. No result is fresh external validation or ordinary-phone facial accuracy.

## Shared predictor and fixed search

Exact condensed median width, weighted randomized Cholesky128 landmarks, Nyström whitening eigenvalue floor1e-8, FP32 storage and FP64 arithmetic from KE/KF. Seeds17/29/43, width multipliers0.5/1/2 and ridge alpha0.1/1/10. Fit normalization and all geometric quantities use fit rows only. No bias term is added to the whitened readout. Payload retains the same seven arrays and approximately20KB. No deployment environment changes.

Five families:

1. `norm_mse`: frozen normalized-Lab ridge, no correction; must reproduce all corresponding KF controls.
2. `constant_de2`: same readout, one shared3×3 color tensor equal to the person-weighted mean of per-fit-target tensors.
3. `local_de2`: separate3×3 tensor at each fit target; quadratic local approximation.
4. `local_irls`: start from normalized ridge, iteratively reweight the fixed local geometry with a smooth robust loss, tau=2 native color-difference units.
5. `midpoint_irls`: recompute geometry at the midpoint of the current fit prediction and fit target; robust weights tau=2. Accept a step only if the actual smoothed CIEDE2000 objective plus ridge does not increase.

For both iterative families retain checkpoints1/4/16. Checkpoint0 aliases unchanged normalized ridge and is an eligible choice for each iterative family, without claiming an extra trained model. Width/alpha/steps selected jointly per family and role by mean person-balanced inner ΔE00 over three seeds; ties prefer fewer steps, then smaller alpha, then width. All15 family choices freeze before final fits or outer evaluation. Seeds are separately evaluated, never selected by outer errors or combined as inference ensembles.

Each shared bank stores243 numeric readouts: 3seeds×3widths×3alphas×(3static +2iterative×3positive checkpoints). Nine inner banks2187 and three final banks729, total2916 stored configurations. Each bank has54 iterative trajectories, up to864 correction solves, with algebra shared within the bank. Policy checkpoint0 aliases are not extra fits. Final selected/evaluated models45. Do not count configurations as independent complete trainings or people.

## Geometry and optimization definition

Let q(v,t)=ΔE00(t+v,t)^2. At anchor t, estimate its local symmetric quadratic tensor M by central differences with h=0.001Lab: diagonal `(q(h ei)+q(-h ei))/(2h²)`; off-diagonal `(q(h ei+h ej)+q(-h ei-h ej)-q(h ei-h ej)-q(-h ei+h ej))/(8h²)`. Symmetrize and project eigenvalues to at least1e-8; record any clipping. This is a local Taylor approximation derived for this experiment, not an exact global CIEDE2000 gradient or established novel color metric. Test against independent analytic infinitesimal CIEDE2000 geometry at chromatic anchors and published numerical pairs.

With D=diag(fit target std), the normalized residual tensor is D M D / s. The common positive scalar s is the weighted mean trace(D M_target D)/3, computed once from fit targets and held fixed throughout every trajectory. Constant geometry uses the weighted mean raw M_target, so both quadratic perceptual controls have the same overall scaling. Normalized MSE remains unchanged.

For whitened design Z and normalized targets T, solve `sum_i w_i (Z_i B-T_i)' G_i (Z_i B-T_i) + alpha ||B||²` using a coupled positive-definite3r×3r linear system. Collapse B back into the original landmark coefficient array for storage/inference. Reuse the frozen exact baseline coefficient computation, preserving zero-correction payloads.

For local IRLS, d² is the raw target-tensor Mahalanobis error; the objective is `sum_i w_i 2tau(sqrt(d²+tau²)-tau)/s + alpha ||B||²`. Weight multiplier is tau/sqrt(d²+tau²). For midpoint IRLS, use midpoint tensors to propose the same weighted solve, but the acceptance objective substitutes actual ΔE00². Fixed candidate step lengths1,1/2,...,1/64; if none improves, retain the old B. Stop further solves after a rejected direction and copy identical later checkpoints. Record requested versus executed solves, steps, objectives and normal-equation residuals. Training-objective descent is not a validation guarantee.

## Verification and measurements

Before real fits: published34-pair metric check, analytic-versus-finite tensor checks (including neutral anchors), local quadratic small-direction agreement, coupled solve versus independent explicitly augmented weighted least squares, baseline payload reproduction, IRLS objective nonincrease and zero-checkpoint equality on synthetic data. No claim of real accuracy from these tests.

During study: hash source and parent KF banks/selection/results, verify person-disjoint rows and complete bank hashes, compare all27 normalized baseline readouts per bank with original KF128 controls, freeze inner choices, then fit all final banks before accessing outer scores. Save trajectories and predictions. Audit chosen readouts independently using analytic color tensors and a separate augmented weighted design/SVD solver; check all45 final predictions/metrics and replay inner choices. Numerical discrepancies above0.002Lab block a success claim and must be investigated, not silently relaxed.

Measure actual CPU batch-one inference using frozen predictor and20 warmups/3passes, prepared color36 only. Separately rebuild each selected seed17 family/role from scratch, one warmup and three timed fits, including normalization, exact width, landmarks and readout; exclude file I/O and hyperparameter search. Record bank and end-to-end workflow time, exact solve counts, numeric/archive sizes. No competing heavy timing jobs. CPU one thread, existing environment; RTX4060 remains available but no GPU speed claim is inferred.

Report all five primary families in all three roles, checkpoint selections, train-versus-outer loss behavior and older strong references. Compare paired person differences descriptively; repeated exploratory selection prevents confirmatory significance claims. Preserve negative results and goal-active status unless the broad objective has actually been established.

## Sources and provenance

[Sharma, Wu, Dalal2005](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf) documents CIEDE2000 computation and discontinuities. Its [author page and numerical fixtures](https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/) support verification; the34 pairs already exist locally. No third-party MATLAB code is copied. Our local tensor is a derivation, not a published guarantee of optimization quality.

[O'Leary1990, Robust regression computation using iteratively reweighted least squares](https://www.cs.umd.edu/users/oleary/reprints/j30.pdf) discusses reweighted solves, conditioning and line searches. [Dong et al., Kernel based regression with robust loss function via IRLS](https://arxiv.org/abs/1903.11202) is prior art for robust kernel training. These sources motivate the experiment; their convergence statements do not automatically cover our midpoint/CIEDE2000 procedure. Existing dataset/code/dependency provenance and restrictions remain unchanged.
