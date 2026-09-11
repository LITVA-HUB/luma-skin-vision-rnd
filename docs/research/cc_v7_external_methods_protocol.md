# V7 real transfer method roster, fixed before new target decoding

Evaluate all15 V7 runs (five arms by seeds17/29/43), six preserved V2 runs
(direct/SoG by the same seeds), two compact learned statistics controls
(gw_ridge1/direct_hgb7), four classical algorithms, and two risk variants of one
Fourier ridge estimator:29 methods total. No seed or family is dropped using
external results. V7 combined standard risk heads are primary; their context/
cheap ablations remain source-development diagnostics.

Primary candidate contrast: canonical_teacher_sensor versus raw_teacher_sensor
(same pretrained teacher, capacity, image/augmentation/training budget). Other
essential contrasts: gt_sensor versus gt_native, and every proposed family
against the strongest locally reproduced V2/statistics/classical/Fourier control.
Average per-seed metrics are not inference ensembles. Negative outcomes remain.

Fourier control: the previously frozen phase1 source-validation winner,
sigma2/ridge.001/gray_world,12,288 real filter/bias degrees of freedom. It is a
local FFCC-inspired convex filter, not an author FFCC/IFFCC reproduction. Use
its original200x score softmax and decoder without changes. Two selectors:
(1) raw1-confidence, positively scaled on source CAL; (2) the same15 standard
RISK-fold candidates as V7, with primary combined64 posterior-grid mass features
plus21 ordinary relative-color features. Posterior mass features are8x8 blocks
of the64x64 PMF (each block sums8x8 cells), not extra learned estimator weights.
Both selectors are frozen before target decoding. Context/cheap are source-only
head ablations, not extra target-selected alternatives.

Classical confidence is mean recovery-angle disagreement with the four classical
estimates, positively scaled on source CAL. Use the same full-image saturated-
pixel masking/algorithms as earlier local benchmark scripts. V2 and statistics
weights, risk heads and CAL scores are inherited byte-for-byte from their
historical locks. All positive scales preserve ranking. No target calibration.

Primary external population:317 already acquired INTEL-TAU rows with reference
hashes absent from historical V2;103 Canon5DSR,112 NikonD810,102 SonyIMX135.
Sensitivity: all384 acquired remainder rows, including67 history-linked reference
hashes. These are unseen camera models relative to the source training cameras
Canon550D/600D. No camera identity, CCM or reference metadata enters prediction.
Reference hashes are bootstrap proxy groups, not confirmed physical scene IDs.

Include the official SimpleCube++462-image TEST as a separate known-camera
comparison after the same method lock. That official image split shares capture
dates with source fitting, and earlier methods already observed it; disclose
both limitations. It is neither a new group-independent test nor evidence of
unseen-camera transfer. Do not use its new V7 scores to change the method roster.

Lock exact population/preprocessing/weight/head/source hashes before decoding
either evaluation cache. INTEL preprocessing matches historical V2: publisher
linear uint16 TIFF RGB, divide65535, mask max>=.98, no chart rectangle because
publisher omits chart acquisition frames, area thumbnail128 with the existing
sample() exposure/clip convention. GT is the original positive RGB .wp vector;
no surface DeltaE, gamma or CCM manufacture. Mirror byte-provenance and original
CC BY-SA4.0 terms remain explicit; target data are evaluation-only in this track.

Report full recovery/reproduction summaries, fixed coverage100/95/90/80/70/60,
full curves, catastrophic tails, and realized target coverage/error under frozen
source thresholds. Always include unsupported inputs in population counts.
Report pooled and per-camera results. Use paired2000-draw reference-hash cluster
bootstrap, stratified by camera, seed20260911, for main family differences at
full/80% coverage. Such intervals do not correct unknown scene dependence,
multiple comparisons, prior source search or teacher pretraining overlap.
No published-author number is substituted for a locally measured result.
