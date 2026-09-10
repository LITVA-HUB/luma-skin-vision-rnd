# Frozen bounded feasibility protocol v1

Written before the first real-data training/test run. This protocol is a first controlled feasibility experiment, not a faithful reproduction of every named paper.

## Data and preprocessing

SimpleCube++ publisher train/test unchanged. Read linear uint16 RGB; subtract the documented approximate scalar black level 2048. Normalize by per-image NormalWhiteLevel minus black; mask pixels above 98% of that metadata white level and dark values within 8 DN of black. Explicitly zero the bottom-right rectangle width175 × height250. No inverse-sRGB transfer, camera CCM, JPEG visualization, GT-derived pixels or illuminant metadata enters prediction. Sensor-level decoding metadata is used; this does not validate uncalibrated consumer JPEG operation.

Classical estimators run on supplied original thumbnail resolution: Gray World p1, Max RGB, Shades of Gray p6, Gray Edge first derivatives with Gaussian sigma1/Sobel3/p6, excluding an 11×11 neighborhood of masked pixels. Their full-resolution input/CPU cost is explicitly reported. Neural models receive area-resized 128×128 linear thumbnails, normalized by each image's 95th percentile valid maximum channel. All models share these settings; they are not tuned on test.

Within official training, stable SHA256 of capture date assigns 55% hash-space to estimator training, 10% validation, 20% risk-head training, 15% threshold calibration. Actual image fractions can differ. These sets are disjoint by date; risk targets are exact residuals of a frozen estimator never trained on those dates. No refit after risk labels. Official test remains publisher random-image split: dates/scenes may overlap development, so this is official-split feasibility, not scene-disjoint generalization. Report overlap/duplicate audit.

Camera protocol: estimator/risk/calibration only from official-train Canon550D; test only Canon600D dates not present in any fitted/selected source subset. Camera models share sensor type: call this a limited held-out-camera experiment, never unseen-sensor proof. Additional Sony IMX135 test consists of the C5 authors' 30 redistributed INTEL-TAU examples, frozen evaluation-only, no target-set fitting or adaptation. It is author-selected and too small/unrepresentative for full INTEL-TAU claims.

## Models and fitting budget

Standard learned baseline C: torchvision MobileNetV3-small feature encoder (random initialization), global pooling, 64-dimensional SiLU context, positive 3-channel illuminant. Standard C+ adds a regularized error regressor from frozen context plus predicted log-illuminant and post-hoc source calibration.

Proposed feasibility variant: identical backbone/context/direct illuminant; add softmax mixture over direct output and four classical experts. Weight MLP receives context plus candidate vectors. Architecture includes the same small branch in the baseline parameter accounting, unused there. Record active/trainable distinction. This is a bounded B-mechanism ablation within direction C; no novelty claim. Both receive identical data/resolution/optimizer/epochs. Standard source-only brightness augmentation and horizontal flips apply to both; no test-camera augmentation.

AdamW lr0.001, weight decay0.0001, cosine decay to0.00002, 60 epochs, batch32, gradient clip5, float32. Optimize established reproduction angular loss. Select lowest validation reproduction mean. Seed17 feasibility first; seeds29/43 only for the same fixed comparison, with no test-based hyperparameter change. Hardware-budget changes, failures or deviations are logged, not hidden.

Risk fitting: ridge penalty10 (intercept exempt), standardized/clipped features, log1p reproduction residual target on disjoint risk set. Post-hoc additive mean residual calibration on separate cal set. C+ context-only; ablations disagreement-only and combined context+candidate vectors+pairwise recovery angles. Same risk supervision/budget for all. Strong classical comparison uses the same learned context and combined feature risk head, fitted to each classical estimator's residuals. Therefore its SELECTOR is learned, with the CNN's compute cost; pure classical color estimate remains deterministic. No distribution-free/conditional calibration claim.

## Endpoints and decisions

Report recovery/reproduction mean, median, trimean, best/worst25, p90/p95, max, fractions above10/20 degrees. Fixed diagnostic accepted fractions100/95/90/80/70/60 use floor(n×coverage), stable ID tie breaking. Full AURC; frozen calibration quantile thresholds and achieved test coverage separately. Ranking curves use scores only, never errors; oracle shown only as an unattainable reference. Paired capture-date bootstrap for source comparison; Sony image bootstrap caveated for unknown scene dependence.

No physical ΔE00 targets. No published author scores mixed into local rows. Positive evidence requires Proposed to beat strongest locally measured comparator AND C+; a point estimate alone is exploratory, not a confirmed contribution. Record negative comparisons and defer optimization/export if no positive evidence. Existing synthetic result is unchanged at its frozen tag.
