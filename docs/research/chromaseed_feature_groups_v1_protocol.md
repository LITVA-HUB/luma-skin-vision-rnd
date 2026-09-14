# ChromaSeed FG: matched nonlinear feature-group ablation

Registered before fitting, 2026-09-13. Previous comparison turn is progress: primary-source model comparison and architecture capacity audit completed, and existing RGB controls were located. C read-only verification freshly passed at the start of this continuation, unchanged SHA256 80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd. Full compact/fast/high-quality goal remains active. No agents, publication or acquisition of new images, datasets or weights.

## Inventory and question

This is not the first RGB/statistics comparison. Existing skin_relational_probe uses median3 versus color36 in excluded-person linear ridge; skin_appearance_inverse uses mean RGB and18 statistics with linear/quadratic mappings on older roles; skin_support_curve tests mean replacement and pixel shuffling while retaining the original statistical branch. Those source protocols and next decisions are context only, never authorization to reopen validation or run old experiments. Current work isolates input groups in the newer128-center analytic kernel/readout, including exact current raw and projected controls.

## Data and representations

Read only the original train.npz, SHA256 d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0, keys color/target/patient/site/device. Preserve existing mixed, SLR-to-iPod and reverse roles, inner three person-disjoint camera-stratified folds and seeds17/29/43. Source validation/calibration/test, tokens and images are not loaded. These24 people and roles are reused and overlapping; this is exploratory mechanism evidence.

Input color36 is encoded-sRGB quantiles(.01,.05,.1,.25,.5,.75,.9,.95,.99), flattened quantile-major RGB (0:27), mean RGB(27:30), population std RGB(30:33), correlations RG/RB/GB(33:36). Seven groups, fixed order:

| Name | Zero-based indices | Dimension |
|---|---|---:|
| raw36 | 0:36 | 36 |
| mean3 | 27:30 | 3 |
| median3 | 12:15 | 3 |
| central9 | 9:18 | 9 |
| mean_std6 | 27:33 | 6 |
| quant27 | 0:27 | 27 |
| no_corr33 | 0:33 | 33 |

Eighth representation is the imported, unchanged X d16_t05 control, all alpha/seed/family candidates, matched by original bank fit/query rows. It still needs all36 input statistics. No claim that a subset represents calibrated reflectance or is camera-invariant.

## Model, matching and storage

Each group fits population input/target mean/std on its fit rows only, FP32 parameters, std floor1e-6, FP32 subtraction/division followed by FP64. Person/site/image balanced weights have mean1. Exact positive lower-median RMS pair distance, threshold1e-10 and width floor1e-6, then FP32 width. RPCholesky seed controls a maximum128 actual landmark rows; same on-demand squared-distance columns, weighted residual stopping and Kmm eigenvalue cutoff1e-8 as A/X. Reduced features use the existing dimension-generic X operator; raw36 preserves exact A preparation and all raw payloads are compared to A at eta0.

Four matched families: norm_static, norm_joint_soft, perceptual_static, perceptual_joint_soft. Shared perceptual tensor and coupled analytic ridge are unchanged A. Alpha candidates0.1/1/10. Joint design [Z,clip(s,-1,1)Z] learns both blocks together; the G ridge gate is fit using only the group's standardized inputs and fit people/cameras. In one-camera banks, joint is an exact static alias. No query camera, target, other query rows, correction loop or calibration target enters inference. No augmentation fitting.

Reduced payload adds uint8 feature_indices (d bytes) to a dimension-d version of the A payload; gate_beta has d+1 coefficients. Raw36 and X controls keep their exact old schemas. Consumer accepts one color36 vector, gathers the declared subset and evaluates it; discarded inputs cannot affect a reduced model. Numeric storage counts indices, normalizers, centers, coefficients and gate. Cache arrays and archive bytes are separate. Extracting features from images and ordinary-phone latency are not timed.

## Counts and selection before final evaluation

Each of12 banks stores252 newly computed configurations (7 groups x4 families x3 seeds x3 alphas),36 imported X controls and one newly fit weighted constant =289. Total3,468 records include1,008 exact single-camera static/joint aliases,432 imported X and12 constants. Expect2,016 actual coefficient solutions /336 shared Gram eigendecompositions /252 basis fits /84 widths /28 gates; raw helpers add36 baseline plus36 theta auxiliary solves. Among these are432 exact A eta0 control payloads; reduced feature solutions number1,728. Target metric is shared across groups in each bank and counts as12 preparations.

Evaluate all nine inner banks on identity only and retain every prediction (491,300 row predictions). For each role/family/group select alpha by seed-average person-mean native DeltaE00; exact ties prefer p90 then larger alpha. Freeze96 group choices before fitting any final bank. Over these eight selected groups, freeze quality and compact policies for each family/role (24 choices). Quality minimizes clean, numeric bytes, p90, group order. Compact first allows clean<=raw clean+0.05 and p90<=raw p90+0.10, then minimizes maximum inner numeric bytes, clean, p90, group order. This is an internal allowance, not an external quality guarantee; no alpha reconsideration after group selection.

Fit all three final banks before evaluating. Evaluate the96 selected group/family/seed cases and one constant per role:291 records. Policy choices reference these same cases and are not extra independent models. Apply all33 frozen GS encoded-RGB contractions after the entire color36 input is transformed, then gather subsets. Retain5,441,700 predictions with native target error and perturbation summaries. Report every group's selected alpha and final outcome, not only the best outer case. Six new reduced groups yield72 matched comparisons to raw (24 mixed,48 transfer); X remains a historical reference, not a new representation claim.

## Validation and complete cost

Tests before primary fitting: exact feature mapping; ignored-feature perturbation invariance; generic moments and width versus explicit differences; raw A fit equivalence; one-camera aliases; consumer FP32 contract and invalid payloads; selection guard/ties; batch/one-row agreement; independent reduced QR/SVD refits. Freeze new core/consumer/runner/tests/protocol with inherited C sources and exact A/X bank files before training.

Independent post-fit audit reconstructs all inner predictions, candidate summaries and frozen choices; checks all raw A and imported X arrays; verifies every final stored output with a separately coded direct-distance evaluator and actual NumPy consumer. Independently refit all288 selected group models with dense kernels, SVD whitening, analytic perceptual metric and augmented QR ridge. Require max native-Lab discrepancy<=0.001 and report effective-rank or landmark-path differences rather than hiding them. Check all source/input/file hashes and exact exclusion boundaries. Constants are independently reconstructed.

After audit, profile291 actual batch-one consumers,20 warmups and3 passes. For every96 selected role/family/group setting at seed17, perform one warmup and three complete fits;384 timing fits must reproduce every array exactly (X control uses its original fit_single). No cached teachers, landmarks or grids in individual fit timing. Record numeric/cache/archive bytes, full fit and query times and primary grid workflow cost, one CPU thread, no concurrent heavy jobs. Finish with reproducible report, model contract, next decision and read-only verification receipt. No universal or ordinary-phone quality promotion from this reused cohort.
