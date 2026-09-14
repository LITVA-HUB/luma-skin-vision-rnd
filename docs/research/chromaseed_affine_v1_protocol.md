# ChromaSeed-A: analytical readouts with controlled mild color augmentation

Registered 2026-09-13 before fitting. Previous GS continuation is progress: its source/input/artifact bindings and terminal process state were reverified. The full compact/fast/high-quality user goal remains active. Local autonomous experiment, no agents, publication or data acquisition.

## Scope and architecture

Original TRAIN only: train.npz SHA256 d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0. Load color/target/patient/site/device. Preserve the three historical roles and three person-disjoint inner folds per role; all are exploratory, repeatedly used and overlapping. No legacy validation/calibration/test, raw images/tokens or new measured target pairs.

Use the exact original-row P/KE128-center basis, width factor1, seeds17/29/43. Fit input/target normalization, exact bandwidth, landmark selection, whitening and acquisition gate only on original fit-side rows. The gate is the frozen G procedure: ridge0.1, equal camera/person/site/image weights, FP32 collapsed coefficients, soft clip to[-1,1]. There is no query camera label, target or batch adaptation.

Four trained families: norm_static, perceptual_static, norm_joint_soft, perceptual_joint_soft. Static design is the whitened kernel vector Z. Joint soft design is [Z,s(x)Z]; both coefficient blocks are learned simultaneously under one ridge penalty. It is not G's frozen-base residual. Joint and static with identical zero-augmentation settings isolate this architecture change from augmentation. When a fit contains one camera group, joint models are exact aliases of their corresponding newly fitted static models at the same alpha/augmentation fraction. This is fit-domain availability, not unknown-camera detection at deployment.

Export static models using the same seven-array predictor (20,284 numeric bytes). Joint export adds128x3 correction coefficients,37 gate coefficients, rho fixed1 and soft-mode flag (21,973B). Exact runtime and cache memory will be measured, with image preprocessing excluded. No prediction loop, query-time fitting, growing topology or new GPU kernel is claimed.

## Training data and objective

Each original fit feature vector creates16 digital-corruption variants: the eight RGB cube corner anchors at doses1/255 and4/255, using GS's bounded contraction v'=(1−t)v+t*a. Quantiles/means transform consistently, std scales by1−t and correlations are unchanged. Variants keep the source's instrument target as a synthetic assumption. They are not16 new measured people/images.

Original fit weights w equalize people, sites and their images, with mean1. For augmentation fraction eta in{0,0.25,0.5,0.75}, original rows receive(1−eta)w and each of16 copies eta*w/16. Total weight per original row stays w, so augmentation does not silently rescale ridge or overweight participants. At eta0, copies have no influence. All original/copy rows stay in the same person's fit fold; fit normalizers/landmarks/gates/perceptual geometry never use held people or augmented distribution statistics.

Use normalized squared target error (G=I) or the original-row shared perceptual3x3 metric from P, with ridge alpha in{0.1,1,10}. Analytically combine original and mean-augmentation feature Gram/cross matrices for each eta. One feature Gram eigendecomposition is shared by two losses and three alphas; solve output coupling through the positive3x3 metric eigenbasis. No backpropagation, weight brute force or iterative convergence loop. Any negative numerical Gram eigenvalues are clipped to zero and recorded. Record normal-equation residuals and regularized objective against the zero-coefficient predictor; objective descent is a fit check, not a generalization claim.

## Controls, counts and ordering

Each bank stores144 new configurations (4families x3seeds x4etas x3alphas),12 exact imported G controls (norm/perceptual base and soft x3seeds) and one weighted native-Lab constant predictor:157 readouts. G soft references use G's previously frozen per-role selected settings, with the matching inner/final G bank rows verified. They are historically selected fixed controls, not newly nested estimates. Check all imported G payloads exactly; do not call copying them independent refitting. Constant control is fitted once per bank,12 numeric bytes, and is not triplicated as independent seeds.

Nine inner banks:1,413 readouts; three final banks:471; total1,884. Of1,728 new configurations,576 are exact single-camera joint/static aliases. Expect1,152 new coefficient solutions in192 shared feature-Gram eigendecompositions,36 basis preparations and4 acquisition-gate fits. Each frozen basis preparation additionally computes one normalized baseline and one theta solve (36+36 auxiliary solves), even when those helper outputs are unused. Twelve constant fits and144 imported G reference payloads are counted separately. Any count divergence must be explained.

Evaluate inner queries with identity plus eight anchors at4/255 (nine transforms), retaining all predictions. Score original mean native ΔE00 by person, and mean by person of each row's worst target error across eight anchors. Seeds average errors, not predictions or independent people. For each of four families and each role, register two policies:

- Clean: minimize original inner person mean over12 alpha/eta candidates; exact ties prefer smaller eta, then larger alpha.
- Guarded: among candidates whose original inner person mean is at most0.05 ΔE00 above that family's best eta0 candidate, minimize the inner worst-of-eight score. Exact ties use original error, then smaller eta, then larger alpha. Best eta0 is always feasible. This is an inner clean-error allowance, not an outer-quality guarantee.

All24 family/policy/role choices freeze after all inner banks, before any final-role evaluation. Both policies reuse the same144 family/candidate/role score pairs. Complete all three final banks before evaluating72 selected trained model instances,36 fixed G references and3 constant controls=111 records. Policy choices may alias the same stored model; do not count them as independent fits. Evaluate every final record under all33 fixed GS transformations, retaining original error, dose-wise worst target error/drift, person/camera summaries and clean comparison to old G/P controls. No further threshold/eta/alpha choice from final outcomes.

## Tests, audit and actual cost

Tests before data: normalized copy-weight mass, mixture Gram/cross equivalence to explicitly stacked weighted designs, coupled eigensolve versus independent augmented least squares, eta0 ignoring augmentation, static/joint single-camera alias, continuous export arithmetic, constant predictor, selection ties and clean-error guard. Freeze core/runner/tests/protocol plus GS/G dependencies and exact G-bank/GS-verification bindings before fitting.

Audit every bank, imported G control, alias, OOF prediction and both selection policies. Independently reconstruct all72 selected trained models using direct kernels, analytic perceptual tensors, separately fitted gate/normalizers and weighted augmented-design QR plus small SVD ridge solves. QR is an orthogonal compression of [design,target], not the primary Gram solve. Require native-Lab drift<=0.001 against that independent reference; report maxima. Audit fixed positive eta0.75/alpha0.1 static/joint probes for both losses and all3seeds on mixed inner fold0, so zero-augmentation selections cannot evade testing augmentation. Explicitly verify feature/copy weights and that all source people stay fit-side. Direct prediction checks and normal-equation records cover all1,884 stored configurations; independent QR/SVD refits are scoped to selected models/probes.

After audit, profile actual standalone NumPy batch-one inference for all111 final records,20 warmups and3 passes, verifying original outputs. For all24 selected family/policy/role choices at seed17, run one warmup plus3 complete fits and require exact exported array parity. Four fixed active mixed probes (both losses xstatic/joint,eta0.75/alpha0.1) add16 fits including warmups. Report full preparation/feature-augmentation/basis/gate/readout time separately from shared-grid workflow time; excludes imports/I/O/search and image preprocessing. Numeric payload is not process RAM. No concurrent heavy jobs during profiling.

Report adverse clean/transfer results and constant/stronger-ridge controls. A flatter predictor is not success merely because its output changes less. Synthetic robustness is not proof of ordinary-phone face/shade-match accuracy, and the full user goal cannot be marked complete on this evidence alone.
