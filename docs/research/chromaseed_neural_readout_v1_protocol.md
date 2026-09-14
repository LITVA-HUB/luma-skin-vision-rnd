# ChromaSeed NR: analytical output fitting on the fixed TG network

Registered before real fitting,2026-09-13. Previous goal turn: **progress**; TG completed numerical training, independent scalar reconstruction, timing, reports and unchanged read-only receipt19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098. Fresh TG verification session59007 returned exit0 at the start of this continuation. Broad goal remains active; this experiment alone cannot validate ordinary-phone face quality. Existing autonomous user authorization covers local implementation/fitting; no additional permission or agents. Preserve the existing isolated worktree and all frozen sources.

## Prior work and question

`scripts/chromaseed_frozen.py` already fits analytic outputs on random/clean/rendered/shuffled palette SiLU features plus a linear skip, with global palette normalizers. Its mixed random result5.889 beat the frozen clean5.978; reverse palette13.880/13.983 was worse than random10.722. Do not call frozen/random readout new, reopen old tests, or rerun that study. P/A/FG already contain normalized/perceptual kernel readouts; TG has independently verified Adam and full3 Gaussian updates on d→64 ReLU→3.

NR isolates whether replacing only the output layer of TG's random or short-trained network gives a better compact error/cost tradeoff. Alternatives were more online epochs and optimizer-grid expansion, or a new recurrent architecture. TG already shows nonmonotonic camera transfer with long training; an exact small linear solve is a bounded next question. Reuse TG architecture and training code, and evaluate both normalized and local-perceptual objectives. This is a matched combination of existing methods, not a novelty claim.

## Data, representations and fixed budgets

Only original TRAIN `train.npz`, SHA256d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0.966 rows/24 people, color36 and native instrument D65/10° Lab3. Person/site/device metadata only for splitting, weighting and reporting. Same three reused roles: mixed fit734/18people, query232/6; SLR→iPod323/8 versus643/16; reverse643/16 versus323/8. Same three person-disjoint camera-stratified inner folds, seeds17/29/43, fit weights equal person→site→image with mean1. Roles overlap and acquisition/person composition is confounded. No legacy validation/calibration/test, new images/data/weights/packages/publication.

Fixed raw36 and mean3[27,28,29] for every method and role. Same fit-only FP32 normalizers and FP32-normalize→FP64 computation as TG. Same d→64 ReLU→3 architecture and exported means. No skip/gate/recurrence/augmentation/new hidden units.451/2563 network parameters;1855/10564 numeric B including normalizers and mean3 indices. Existing actual TG NumPy Predictor is the deployment consumer.

Seven representation bases: random, Adam after1/4/16epochs, full3 Gaussian after1/4/16epochs. Fixed Adam rate.001 and Gaussian sigma1 across every role/group, chosen as fixed TG-grid values before NR fitting; no NR hyperparameter search for representation training. Both share TG Xavier means, Gaussian prior variances, person/site weights and per-seed permutations from RNGseed+600017. New primary trajectories run through16epochs in each inner/final fit. TG exports at1/4/16 are used as exact controls where the frozen TG run contains the same setting; do not substitute differently selected TG final hyperparameters. Random uses TG initialization plus fit-only normalizers, without supervised updates.

This protocol uses learned representation weights exported FP32 before the readout fit. All input normalizers, first-layer weights/biases, output target normalizers and input indices remain bitwise identical during readout fitting. Original unchanged trained outputs are separate controls; random output itself is not promoted as a color baseline. Existing TG bank FG norm_static/raw36+mean3 alpha.1 seed models and balanced constant are imported exactly.

## Output fitting

Compute hidden activations `h=ReLU(normalize32(x) @ w1_32.T + b1_32)` using FP64 matrix arithmetic. Fit-only unweighted population hidden mean μ and std s with floor1e-6 produce `Z=[1,(h−μ)/s]`, N×65. Target `T=(native_y−y_mean32)/y_std32` uses FP64 native y; this avoids further target rounding in the analytic solve. TG representation training retains its original FP32-normalized target contract.

Two losses: `norm` with M_i=I3; `perceptual` with local CIEDE2000 squared-error tensor G_i at true fit Lab. Primary uses P's finite-difference tensor(step.001); reference uses P's independent analytic infinitesimal tensor. Let D=diag(y_std32), c=weighted_mean(trace(D G_i D))/3, M_i=D G_i D/c. This is a local quadratic surrogate, not exact minimization of ΔE00 or a calibrated probability.

For alpha in(.1,1,10), solve

`sum_i weight_i (Z_i B−T_i)^T M_i (Z_i B−T_i) + alpha * ||B[1:]||_F²`.

The intercept is unpenalized. Norm solves65×65 with three RHS; perceptual solves195×195 in feature-major/output-minor order. Primary uses Cholesky normal equations; independent reference uses augmented least squares with pivoted QR (`scipy.linalg.lstsq`,gelsy), including separate reconstruction of hidden activations and the perceptual tensor. Positive alpha and observed intercept make the system positive definite. No adaptive clipping/rescue or silent alpha change. Nonfinite output or failed factorization stops the run.

Fold feature standardization into existing output arrays: `w2=(B[1:]/s[:,None]).T`, `b2=B[0]−μ@(B[1:]/s[:,None])`, cast FP32. Extra feature moments/tensors/linear systems are training-only and not deployed. Save max FP32-folding drift, normal-equation residual, hidden degeneracy, metric scale/eigen diagnostics and solver counts. Variance diagnostics belong only to the representation trajectory.

## Selection, controls and complete accounting

Per bank:42 representation payloads (7bases×2groups×3seeds),252 new readouts (42×2losses×3alphas),36 unchanged learned-network controls,6 FG controls and1 constant =295 model records. Nine inner and three final banks:504 basis records,144 new representation trajectories through16epochs with432 learned checkpoints plus72 random payloads,3024 head solves and3540 model records. Representation updates =12 trajectories per bank ×16epochs ×5100 summed fit rows =979200. Final reference exports match independently retrained means; no teacher state cache is excluded from full cost.

Inner OOF count295×1700=501500 row outputs. For each role/basis/group/loss, score all3alphas by equal-person mean ΔE00, averaged across separate seed errors; tie by mean p90 then alpha order.84 choices from252 candidate scores. Additionally freeze one quality policy per role/group/loss: choose among7 already-selected bases by inner clean,p90,representation epoch,basis order.12 policies. Policies add no fits or evaluated cases and must not be replaced after outer results.

Write all selections atomically before final banks. Final banks retain the entire registered alpha grid, but evaluate only selected heads84/role, unchanged trained controls36/role, FG6 and constant1 =127/role,381 total. Same33 affine-color transforms(identity +8RGB corners at doses1/4/16/64 divided by255), with targets fixed.127×1198×33=5020818 final row outputs; identity is included, not an independent extra test.12573 transform cases,1524 dose cases. Synthetic stress is an intervention diagnostic, not natural camera calibration.

Report all seven bases, both losses/groups and role outcomes, selected policies and adverse cases. Primary paired comparisons:84 NR method/group/basis/role means versus matching FG group;72 nonrandom NR comparisons versus the corresponding unchanged network;12 policies versus FG. Bootstrap descriptive fixed-prediction person differences with20000 draws, RNG771031, with reused-person/selection uncertainty limits. Seeds are individual deployable models, not an ensemble.

## Verification and timing

Before fitting, test scalar unpenalized-intercept ridge, correlated three-output objective against dense augmented least squares, analytic/finite-difference geometry, dead/constant hidden channels, exact first-layer preservation, folded consumer output, shape/finite/positive constraints, seed/prefix consistency and selection ties. Separate oracle code from primary solver assembly. Initial missing-function/module failure is required before implementation, then all appropriate numerical tests pass before source locking.

Audit all12 banks,504 basis payloads,3024 analytic head fits using independent QR, exact TG controls wherever available,84 FG imports, every OOF and final output, all252scores/84choices/12policies. Independently train all144 representation trajectories through16epochs and compare432 exported checkpoints plus72 random bases. Check row/hash/order/normalizer/floor metadata. Independent head prediction tolerance.001 nativeLab; same-consumer/direct prediction2e-8; train export tolerance2e-6atol/rtol, record exact-match counts. No claim that numerical reproduction establishes product accuracy.

Profile all381 actual consumers with20 warmups and3 passes, reporting median/p95 and initialization, numeric/archive/cache bytes. Complete selected NR fits84settings plus36 unchanged settings plus6 FG settings =126 settings ×3 full repetitions =378 fits; one warmup and two measured, seed17. Construct hidden representation from scratch every repetition; random includes normalizers/initialization, learned includes all chosen epochs. Include feature transforms, weights, tensor, linear solve and export in NR time; FG recomputes kernel geometry. Training-only diagnostic arrays/RNG/process storage are reported separately from deployed weights. CPU one thread, no concurrent heavy jobs; no image preprocessing/I/O/imports/phone/CUDA inference claim.

Seal sources, data/input/parent receipts, report/runtime/audit/consumer/model-card bindings and a read-only re-verifier. Mutable plan/current-status files stay mutable. All original process handles must be polled to terminal; no restart from observation timeout. Keep full objective active until ordinary-phone face/end-to-end quality is actually established.

[Parent TG report](../benchmarks/chromaseed_gaussian_v1/report.md) · [Prior palette readout](chromaseed_next_decision.md) · [Next-question provenance](chromaseed_gaussian_next_decision.md).
