# ChromaSeed-W: expanded ridge range

2026-09-13. Registered before W-series real-data fits. The previous P turn is verified progress: source lock, verification receipt and terminal process state checked. Its15 family/role choices all selected alpha0.1, the lower boundary. This motivates an expanded range; improvement is not assumed. The broader user goal remains active. Autonomous local experiments are authorized, without delegation/publication/new data acquisition.

## Single controlled change

Keep P's128 landmarks, color36/native-Lab interface, fit-only normalization, exact condensed median, weighted randomized Cholesky, eigenvalue floor1e-8, FP64 computation/FP32 storage, seeds17/29/43, widths0.5/1/2, five loss families, tau2 and corrections0/1/4/16. All iterative geometry, line search, stopping, solver and coefficient-collapse code is reused from the frozen P core. No inference graph or numeric payload expansion.

Expand alpha from0.1/1/10 to **0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,1,10**. The three previous values stay as controls. Equal numeric alpha does not imply equal effective regularization across loss geometries. No alpha0 or unregularized pseudoinverse. No adaptive grid expansion within this series.

Families remain `norm_mse`, `constant_de2`, `local_de2`, `local_irls`, `midpoint_irls`; mathematical definitions and fitting boundaries are those in [P protocol](chromaseed_perceptual_v1_protocol.md). Checkpoint0 aliases normalized ridge at the same width/alpha/seed. Joint per-family selection over width/alpha/steps uses inner person ΔE00 averaged over3 seeds; ties prefer fewer steps, smaller alpha, then width. No seed selection or inference ensemble.

## Data and sequencing

Use only original TRAIN SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`,966 rows/24 people; load color,target,patient,site,device only. Same frozen three roles and3 person-disjoint inner folds. Equal-person/site/image weights with mean1. Do not load images/tokens or legacy validation/calibration/test. Reused roles overlap and camera transfer also changes people/color distributions; all results remain exploratory, not fresh phone-face validation.

Each bank saves729 readouts:3 seeds×3 widths×9 alphas×(3 static+2 iterative×3 positive checkpoints). Nine inner banks6561; three final banks2187; total8748. This includes2916 exact-P control configurations and5832 new-alpha configurations, with shared algebra rather than8748 independent full trainings. Up to1944 iterative trajectories×16 correction solves, plus1944 static coupled solves. Actual executed solves and early stops are recorded.15 family selections freeze after all9 inner banks and before all final fits; evaluate45 selected models only after all3 final banks complete.

All243 saved P configurations per matching bank must reproduce exactly after alpha-index remapping, including prior positive corrections. Store control counts and hash all sources/parent banks/selections/results; inherit existing source locks instead of editing old files. Save all trajectories and bank predictions. Any implementation discrepancy is investigated before a success claim; no silent threshold relaxation or overwrite of old results.

## Tests, audit, timing

Before primary fits: tests for alpha0.1 preservation across all five families, weak-alpha fit predictions against independent analytic-tensor/augmented-SVD reference including near-collinear features, eligible checkpoint0 alias and explicit grid size. Existing15 P/KE numerical tests remain valid; do not alter their frozen sources.

Audit all bank/normalizer hashes, every stored OOF prediction, all logged trajectory monotonicity, all15 inner choices and45 selected final predictions/metrics. Independently refit selected models with analytic tensors and augmented SVD. Additionally replay inner fold0/seed17/width1/alpha0.0001 for both iterative families in all three roles, checking1/4/16 snapshots:18 positive-step probes at the new smallest penalty. Max permitted query-component drift0.002Lab, unchanged from P. This is local numerical replication, not external predictive confirmation.

Time the actual frozen CPU batch-one predictor for all45 selected models,20 warmups and3 passes. Rebuild each selected seed17 family/role from raw prepared fit arrays, one warmup and3 measurements, including weights/preprocessing/exact width/landmarks/readout, excluding I/O and search. Additional fixed16-step cost probes use seed17/width1/alpha0.0001 for both iterative families in all3 final fit sets, without new outer quality scores. No competing heavy jobs. Profile existing one-thread CPU environment; no inferred RTX4060 speed claim.

Report all five methods/roles, chosen penalty/width/steps, error versus inner penalty/checkpoint curves, conditioning/residuals, model size and full fit cost. Compare both expanded-grid normalized control and each matching old P family. Separate the effect of a broader penalty search from the benefit of perceptual loss or correction. Retain prior stronger per-role references and all negative results. Lower training loss, more search, or a smaller inner error alone does not complete the goal.

## Method context

[Rudi, Camoriano, Rosasco, Nyström computational regularization](https://arxiv.org/abs/1507.04717) treats the relationship between landmark approximation and regularization. [FALKON](https://arxiv.org/abs/1705.10958) is prior art on scalable regularized kernel training. Neither validates these skin-color data or guarantees this loss/grid will improve. P's CIEDE2000/IRLS sources and dataset/code/derived-model rights remain unchanged. No new implementation or model is imported from the web.
