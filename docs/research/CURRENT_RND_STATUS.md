# Latest research: paired-capture invariance falsified in source screens

27new fits completed: five matched arms x3seeds, then two methods x3seeds
x2camera directions. This uses ONLY original TRAIN/VALIDATION roles and is
exploratory. Prior independent400-image test and208-image calibration endpoints
were not loaded. Source raw baseline mean over seeds3.4855DeltaE00; output
consistency3.5123; VICReg3.5450; hard3-direction nuisance quotient4.1101.
Same-site prediction consistency improves without better instrument accuracy.
Camera-held-out fitting also loses: SLR->iPod5.6063raw versus6.1526consistency;
iPod->SLR6.1399 versus6.5656. All six paired seed/direction comparisons worsen.
Source architecture selection had previously seen both cameras: not a new
independent unseen-camera test. 27fits/81exact prediction-array replays and
5,544independent scalarDeltaE00 comparisons pass;265tests pass.

[Paired-capture report](../benchmarks/skin_pair_v1/report.md),
[fixed mechanism/protocol/prior art](skin_pair_invariance_protocol_v1.md).
Next invert the failed assumption: retain and infer capture-process evidence
instead of suppressing it; test matched auxiliary-task/conditional estimation
and perceptual-loss controls on source data. No background training is running.
The independent benchmark below remains unchanged and authoritative.

# Independent instrument skin-color test completed

Primary endpoint is actual native-Lab skin color, measured with CIEDE2000, not
illumination angle. Frozen test: 400 images / 105 sites / 10 previously held-out
people. Primary 2,774,796-parameter patch ensemble mean4.45698, median4.00120,
p959.26312 DeltaE00. At80% accepted coverage Proposed mean4.15912 versus matched
C+4.33276; paired difference -0.17364, patient-bootstrap95%CI[-0.56848,0.16472].
The interval crosses zero. Ordinary six-model fusion full mean4.30047 and
density-ranked80% mean4.14472 is stronger observed full-system evidence.
No convincing proposed-method victory; no product accuracy victory.

The primary method's80% median3.88337/p958.09657 fails the working <=2/<=5
product aspiration. The calibration-fixed80% threshold actually accepts74.25%
of test images. Both development camera families are known; ordinary unseen
iPhone/Android facial accuracy remains unvalidated. No angular proxy claim.

[Independent report](../benchmarks/skin_mskcc_selective_v1/report.md),
[protocol](skin_mskcc_selective_protocol_v1.md),
[declared pre-test reference amendment](skin_mskcc_reference_amendment_v1.md).
49prediction arrays/294independent coverage cases and8risk/calibrator replays
pass;18OOF fits replay;262tests pass. TEST is now an exposed evaluation archive,
not a future tuning or fresh confirmation set. No background training remains.
The earlier sections below are historical and do not describe current access.

# Historical: direct skin pixels measured, source validation only

The local JPEG-to-instrument-Lab pipeline is now implemented and measured.924932parameter patch models achieve meanDeltaE00 3.4888/3.5074/3.4964 across3seeds, versus MobileNet3.8139/4.0029/4.1147 and matched global MLP3.8113/3.7303/3.9020. Six-person validation was used for selection;10test and6calibration people remain unopened. Iterative special mechanisms did not reliably beat simple matched pooling.

[Current decision](skin_mskcc_pixel_decision.md), [pixel benchmark](../benchmarks/skin_mskcc_pixels_v1/report.md), [negative ablations](../benchmarks/skin_mskcc_pixel_ablation_v2/report.md).42checkpoint replays and256tests pass. No novel-method victory or ordinary phone facial accuracy claim. The text below is historical evidence, not the latest execution state.

# Latest: real skin DeltaE00 source pilot (2026-09-11)

The primary endpoint is instrument-referenced skin color, following the user correction. Original MSKCC CC-BY data now supports direct native-Lab DeltaE00. On six validation people, the source-selected ordinary MLP has mean4.3702, median3.8038, p959.7150; fixed-distance80% coverage mean4.3694, with worse p95. This is not an independent final test or novel architecture result. Ten test people remain numerically unopened.

[Source-only benchmark](../benchmarks/skin_mskcc_summary_v1/report.md), [next decision](skin_mskcc_next_decision.md). Ten model replays and60independent metric cases pass. All1838original paired JPEGs (2,128,062,766bytes) are now acquired and individually verified. No complete pixel pipeline has yet been evaluated. A separate-directory refit exactly reproduced all10models. Historical He white-reference limitation and all angular/synthetic negatives remain valid below.

# Latest priority: actual skin-color error, 2026-09-11

User clarification makes instrument-referenced skin color the product endpoint.
Angular normalization remains a component diagnostic. [Target and acceptance
criteria](skin_color_target_2026_09_11.md) supersede proxy-only prioritization.

NEW REAL SKIN PILOT: original-author CC BY4.0 paired regional RGB/XYZ data,
200 sites/40 training people and100 sites/20 held-out people. All14 ordinary
controls frozen before test extraction; all14 weight replays and168 independent
metric cases passed. Source-CV-selected RAW poly3 XYZ RMSE2.025092; JPG poly2
1.952336. Simple affine controls actually perform better on test,1.830634 and
1.860912 respectively; do not hide failed source selection. These are XYZ
coordinate errors, not perceptual DeltaE or a percentage accuracy. Exact white
reference remains unverified; the proposed companion-spectrum match failed.
No V7/full-image/phone skin-color accuracy is established. [Skin report](../benchmarks/skin_he_xyz_v1/report.md).

V7 COMPLETE NEGATIVE PRIMARY HYPOTHESIS: all15 trainings/30 checkpoint replays,
all three source-risk/stress audits, and the29-method external benchmark finished.
On317 reference-history-disjoint INTEL images, canonical teacher full/risk80
5.5013/5.3135 versus matched raw teacher4.9761/4.7481. Paired differences favor
the control, including descriptive95% intervals. Source-only sensor augmentation
helps versus native training but does not beat the strongest control. Fourier
ridge raw4.3552/3.6233 is the strongest observed pooled baseline. This is a custom
split, not author FFCC reproduction/SOTA. All261 method/population records and
1,566 coverage/threshold cases passed an independent atan2 audit; max4.61e-11.
[Full real-camera report](../benchmarks/cc_v7_external/report.md).

Projector/cone TRAIN probe completed: numerical invariance holds, but the
positive16x16 patch-cone oracle recovery floor0.6751 exceeds the current fitted
comparator's TRAIN recovery0.6042. Signed-weight geometry remains PLANNED and
secondary to obtaining a valid skin-color endpoint. [Probe](../benchmarks/projector_probe_report.md).

Historical states below are retained as chronology; they are not current process
status. No prior negative result is overwritten and the R&D goal remains active.

# Previous progress: V7 camera-robust training, 2026-09-11

[V7 seed17](../benchmarks/cc_v7/seed17_report.md) completed all five120-epoch arms;
all10 best/final checkpoint replays passed. Deployment estimator3.034M;
training-only semantic projection0.369M; teacher never enters inference.
Canonical-teacher+sensor does not beat raw-teacher C+ on reused source VAL:
full2.4591 versus2.4349°, risk80 1.8630 versus1.7608°. GT-only native is stronger
on this source screen:2.3383° full /1.7595° risk80. No new real-camera gain yet.

Sensor augmentation helps a fixed virtual mixing diagnostic: GT-native19.4090°
versus GT-sensor2.6372° at the extrapolation matrix. Preserved V2SoG is still
better there at1.5651°. These are transformed real-source images, NOT real new
camera measurements. All six matrices, tails, curves and best/final outcomes
are retained. Independent audit:162 fixed-coverage/threshold/stress cases,
max scalar error difference8.47e-11°. All source risk heads are fitted only on
259 held-out RISK rows, calibrated on268 date-disjoint CAL rows.

Seeds29/43 are running sequentially in session23891; never duplicate them.
The [317-primary/384-sensitivity external population](../data/provenance/cc_v7/external_population.json)
is metadata-frozen, still not decoded. Complete all15 models and source heads,
then freeze exact methods before new INTEL-TAU evaluation. Target-camera data
remain evaluation-only, preserving the permissive source-only training line.
[Transfer protocol](cc_v7_transfer_diagnostics_protocol.md).

[V6 final](../benchmarks/cc_v6_report.md): all12 arms /24checkpoint replays complete,
combination negative versusV5. [Fourier report](../benchmarks/fourier_representation_report.md):
all34cases completed and replayed. Full suite236passed before two new risk-role
tests, which passed separately. No V7 latency/export or facial color validation.

Historical progress below is preserved; newer status above takes precedence.

# Previous progress: phones, V6 and alternative representations

Latest real phone screen: [79-reference report](../benchmarks/phone_v1_alias_report.md).
Thirty source-only methods on Beyond RGB Samsung/Oppo; original dataset CC BY4.0.
V2 SoG mean4.367 versus C+4.838; risk80 3.951 versus4.498. Confidence intervals
include zero and performance is camera-dependent. V5 physical critic risk80
5.106: newer architecture loses. Original72-row format-strict result and both
loader repair receipts are preserved; this is a custom transfer screen, not
skin/JPEG/HEIC accuracy or a new test after the format repair.

[V6 status](cc_v6_execution_status.md): seed17 combination is negative so far;
seeds29/43 in progress. [Broader search](aggressive_search_2026_09_11.md) explicitly
challenges global illumination, point outputs and the need for a spatial CNN.
Fourier ridge12,288-coefficient control gives2.747 full/2.071 raw risk80 on reused
source validation; all16 initial outcomes are preserved, no phone scoring.

Previous milestones follow unchanged as historical evidence.

# Current R&D status — V5 completed, phone transfer next, 2026-09-11

All18 primary arms completed. All36 best/final records passed independent
scoring and CPU weight replay; the full suite passed206 tests. Mean best
reproduction: generic action2.4136°, ordinary physical transport2.4548°;
mean raw risk80:2.1333° versus2.0192°. Selected-action/derivative training
failed to improve consistently. Equal-query controls do not support a useful
feedback-specific gain. [Final V5 report](../benchmarks/cc_v5_report.md).
This is reused development validation, not independent calibrated superiority.

The3.23GB Samsung/Oppo selection is fully acquired and byte-verified. Only
three TRAIN scenes were inspected;44 paired-phone test scenes remain reserved.
The next action is a frozen method/weight lock and separate phone evaluation.
No training/download process remains live. [Authoritative execution state](cc_v5_execution_status.md).

Historical V5 interim updates follow; they are superseded by the completion
report above and retained as development context.

Execution update: seeds17/29 completed all six arms each. All24 best/final
prediction records passed independent scoring and CPU checkpoint replay. New
selected-action/gradient training has no consistent advantage so far; seed43
continues in the same serial session. Phone
coverage is now an explicit priority: a licensed Samsung S21 Plus/Oppo Find X5
Pro subset is downloading, and a small CC0 iPhone SE2/XS Max repeatability archive
is acquired. Phone loading/reference extraction and classical diagnostics passed
on six TRAIN captures; no held-out phone accuracy has been measured. [Current handles, interim
numbers and next steps](cc_v5_execution_status.md),
[phone-data rights and protocol](../data/smartphone_benchmark_plan.md).

New diagnostic: on both completed seeds, all five trained critics retain the
original point after the coarse25-query search on every validation image.
Thus51-query adaptive and fixed multiscale searches select exactly the same
actions. At103 queries the difference is small and mixed. This specifically
weakens the repeated-refinement hypothesis for the present design;
[equal-query control](../benchmarks/cc_v5/two_seed_search_control/report.md).

V5 is now testing selected-action error supervision and physical cost derivatives
against point-only, posterior and generic action controls. Six arms per seed,
seeds17/29/43, 120 epochs. All arms clone complete20-epoch warmup state;
strict CUDA warmup replay passed bitwise. Primary execution was launched in
session74620 from committed training code2b23f90. This is an execution receipt,
not evidence that all runs completed. Read run result.json/session state before
assuming completion or restarting. [Protocol](cc_v5_protocol.md),
[preflight](../benchmarks/cc_v5_preflight.md),
[source binding](../benchmarks/cc_v5/launch_receipt.json).

The full suite passed190 tests before primary launch. Two additional independent
scorer tests subsequently passed. V5 primary accuracy, calibration and camera
generalization remain pending. The separate scorer verifies run hashes and
recomputes errors from real validation GT, retaining best and final checkpoints.
The strongest completed positive milestone remains V2; the latest completed
architecture screen remains the negative V4 result below.

## Preserved V4 screen

**Latest V4 architecture does not beat its controls.** Three 3.097M-parameter, 120-epoch runs give step-2 mean reproduction error 2.389° for physical evidence transport, 2.317° for generic action feedback, and 2.181° for the action-independent posterior. Raw risk80 is 2.000° / 2.009° / 1.895°, respectively. These 119 validation images were reused for development, with one seed; nondeterministic CUDA histories diverged even during the common warmup. This is a negative screen for the proposed mechanism, not a stable causal comparison or an independent benchmark. [V4 report](../benchmarks/cc_v4_report.md), [architecture/training synthesis](cc_v4_universal_method.md).

The user's repeated-refinement idea was implemented and measured at 1/2/4 stages. Posterior mean 2.186°→2.181°→2.174° is only a small change; transport 2.357°→2.389°→2.371° worsens relative to its starting estimate. Predicted risk decreases while some true errors increase. Proposed step-2 batch-1 median latency was 6.104–6.203 ms across three sessions on RTX 4060, from device-resident 128×128 RGB through the cached encoder and all queries. The model has 3,097,189 parameters and a 12,599,089-byte checkpoint. Training peak allocation was 765.46 MiB including 233.45 MiB of source cache; inference peak was 29.12 MiB. Other mode/stage timings vary more on the desktop; all receipts are retained. No V4 export optimization was performed.

Modern leading-lab training and close photometric sources were reviewed; no external pretrained weights were adopted. Next priority shifts from adding refinement stages to improving selected-action error supervision with paired initial checkpoints and stronger scene context. New-camera, source-test and calibration roles remain untouched by V4. [Decision](cc_v4_next_decision.md). The repository passed 185 tests; independent metric recomputation and CPU checkpoint replay passed. The overall R&D goal remains active.

## Preserved V3 architecture screen

**Latest experiment: the new full color-frame graph loses on real development validation.** Three matched1.216M/120epoch runs give mean reproduction4.281° full-frame Proposed versus2.547° direct RGB and2.942° diagonal. At80% raw-posterior acceptance, full-frame gives4.155° versus2.304° direct. These119 validation images were reused for development; this is not a fresh test result. The negative design is preserved and a relaxed graph with explicit global color state is the next unverified hypothesis. [V3 report](../benchmarks/cc_v3_report.md), [revision decision](cc_v3_revision_decision.md).

The384 newly acquired INTEL-TAU images have not yet entered training or error evaluation. Identity auditing found67 rows sharing old reference-file hashes, despite zero image duplicates; fold/group policy remains to be locked. A source-grounded FFCC numerical control is implemented but not yet trained. The broader R&D goal remains active.

## Preserved V2 measured milestone

**Measured improvement against the strongest matched direct CNN: reproduction error at 80% acceptance on 384 fresh unseen-camera images falls from 5.558° to 3.805° (31.5%). No universal best-method, patent-novelty or facial skin-color claim is established.**

The proprietary instrument-paired facial dataset remains unavailable under the user's hard constraint. Public-data R&D continues. Historical synthetic state `2685bf0` / `milestone/synthetic-only-2026-09-10` and public V1 state `7637d6d` / `milestone/public-cc-feasibility-2026-09-10` remain frozen. Their negative results are preserved in [synthetic status](SYNTHETIC_MILESTONE_STATUS.md), [V1 status](PUBLIC_V1_MILESTONE_STATUS.md) and [V1 report](../benchmarks/public_benchmark_report.md).

## Implemented and measured

- Train only on SimpleCube++ real illuminant ground truth, CC BY 4.0. Separate capture-date groups for estimator fitting, model validation, error fitting and calibration. No imported pretrained weights.
- Fresh evaluation-only INTEL-TAU subset: 128 images each from Canon 5DSR, Nikon D810 and Sony IMX135_BLCCSC. Original CC BY-SA 4.0 verified directly. Deterministic IDs, 4.434 GB transferred, all 768 image/GT files hash-checked. Pinned mirror consistency is verified; byte identity to original full archives is not. This is a custom subset, not the full published benchmark protocol.
- A known anchor-normalized residual construction with a source-fitted relative-feature error head. Seventeen CNN trainings, matched 0.964M/3.034M variants, three final seeds, source-only selector and statistics-regression screens, preserved failed variants, all final choices locked before new-camera errors.
- Strong matched 3.034M direct C+ and Proposed receive the same 128×128 input, data, scratch initialization, 120 epochs, optimizer/augmentation budget and selector search. Classical Gray World, Max RGB, Shades of Gray and Gray Edge, compact learned statistics controls and preserved V1 methods are locally evaluated.
- Recovery/reproduction errors, all fixed coverage levels, complete curves, tails, paired bootstrap, per-camera errors and realized coverage at frozen source thresholds. Independent recomputation of 594 records agrees to numerical precision.

Fresh full mean: Proposed 4.719° versus C+ 5.662°. Fresh risk80: 3.805° versus 5.558°, paired difference −1.753° with proxy-cluster 95% interval [−2.132,−1.336]. Same-estimator context-only versus combined-risk ablation improves 4.266°→3.805°. Cheap GW+ridge achieves 4.558° full mean and 3.977° risk80; its risk80 difference from Proposed is inconclusive. A one-seed GW residual diagnostic has lower full mean, 4.229°.

**Negative findings:** source risk80 worsens 1.626°→2.276°; Canon risk80 worsens 4.042°→4.683°. Nikon and Sony improve versus C+, but there is no camera-uniform dominance. A source-calibrated nominal80% threshold accepts only47.66% of new images. Accepted catastrophic errors remain. Source and target differ in scenes/dataset as well as camera; physical scene independence is not established.

## Compute and export

Representative seed17: 3,033,651 parameters, 12.324 MB checkpoint, 12.142 MB FP32 ONNX. RTX4060 batch1 median4.181 ms model-only;24.691 ms PNG bytes→score at648×432 source image size, excluding disk and correction rendering. Peak PyTorch allocated training762.34 MiB includes233.45 MiB cached source data; inference28.12 MiB, excluding driver/context memory. ONNX CPU parity passed, maximum score difference4.30e-6°, all tested invalid inputs refused. TensorRT and quantized accuracy/latency remain unmeasured.

## What remains unvalidated and next decision

Prioritize compact single-image unseen-camera transfer (candidate C), retaining relative-context risk estimation (B) and the limited reproduction-risk proxy (A). The equivariant wrapper is known prior art, not a new invention. Keep cheap GW+ridge and GW-residual controls. Next: independent multi-camera source validation, stronger published compact-method reproduction and a new locked test set; investigate Canon failure without tuning this now-observed subset.

Physical facial Lab/ΔE00, skin-specific reliability, arbitrary smartphone ISP robustness, cosmetics outcomes, production calibration guarantees and patentability remain **NOT VALIDATED**. Skin-specific colorimetric validation remains future work requiring a facial dataset with appropriate reference measurements.

Full evidence: [V2 report](../benchmarks/cc_v2_report.md), [reproduction](../benchmarks/cc_v2/REPRODUCE.md), [prior art](cc_v2_prior_art.md), [licenses](../data/public_dataset_inventory.md), [Skolkovo addendum](../skolkovo/cc_v2_evidence_addendum.md).
