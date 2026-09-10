# V5: paired correction-critic training and equal-query refinement

**Completed development screen; no new independent benchmark claim.** All18
120-epoch arms (six per seed17/29/43) finished. All36 best/final prediction
records passed independent GT scoring and checkpoint replay on CPU. The full
repository suite passed206 tests with14 historical ONNX warnings in32.66s.

The proposed selected-action and derivative-training schemes do not improve
the ordinary transport critic consistently. A modest selective-ranking signal
remains for physical transport, while the generic action critic has the lowest
mean error. This is not evidence of a large new accuracy advance or superiority
over a fully calibrated matched C+ on an independent domain.

## Real data, method and scope

SimpleCube++, original CC BY4.0:1126 training and119 reused development-validation
images with real illumination GT. No synthetic training, pretrained teacher,
new camera labels or phone test data entered these runs. Each seed has one
20-epoch warmup; all six arms clone its model, BN, optimizer and scheduler state.
Strict deterministic CUDA warmup replay passed. Training scripts are identical
across seeds. Seed43's repository snapshot differs only by added h5py dependency
metadata for separate phone preparation; exact per-seed provenance is retained.

The network caches16 local illuminant hypotheses and scene context. Its critic
weights hypothesis costs either independently of correction (posterior), from
generic action context, or using nonlinearly corrected color statistics
(physical transport). New training arms change action sampling to current selected
corrections and optionally supervise physical cost derivatives. Candidate scoring,
ensembles, derivative supervision and recurrence are known mechanisms; no broad
novelty claim is made. [Frozen training protocol](../research/cc_v5_protocol.md).

## Measurements

Numbers are arithmetic means of three per-seed metrics, **not a three-model
inference ensemble**. Lower reproduction angular error is better. Best checkpoints
were selected on these same119 images. Final-epoch outcomes expose selection
effects. Raw risk80 accepts95/119 images per model, without post-hoc calibration.

| Arm | Best full mean ° | Final full mean ° | Best raw risk80 ° |
|---|---:|---:|---:|
| Point-only compact estimator |2.4814|2.5911|Not trained|
| Fixed posterior critic |2.5612|2.6657|2.1307|
| Generic action critic |**2.4136**|**2.5666**|2.1333|
| Physical transport, random actions |2.4548|2.5853|**2.0192**|
| Transport, selected-action sampling |2.5346|2.6415|2.1895|
| Transport, selected actions + derivatives |2.4812|2.6874|2.1448|

Physical transport is about1.7% worse in full mean than the generic action critic;
its mean risk80 is about5.3% lower. Its risk80 advantage over that control appears
in seeds17/29 and reverses slightly in43. The posterior has lower risk80 in seeds29
and43, so transport is not a consistent per-seed winner. Three training seeds do
not supply three independent datasets or a confidence interval for deployment.

| Coverage | Posterior | Generic action | Transport random | Transport selected | Transport derivatives |
|---|---:|---:|---:|---:|---:|
|100%|2.5612|2.4136|2.4548|2.5346|2.4812|
|95%|2.4010|2.3159|2.3412|2.3730|2.3381|
|90%|2.3377|2.2358|2.2665|2.3326|2.1873|
|80%|2.1307|2.1333|2.0192|2.1895|2.1448|
|70%|1.9680|1.9863|1.8259|2.0628|1.9557|
|60%|1.8237|1.8325|1.6172|1.7712|1.7540|

![Raw development risk-coverage](cc_v5/final_summary/risk_coverage.png)

[Full per-seed table and predictions](cc_v5/three_seed_screen/report.md),
[recovery/trimean/quartiles/tails and all119 coverage points](cc_v5/three_seed_screen/summary.json),
[aggregate and provenance](cc_v5/final_summary/aggregate.json).

## The repeated-refinement test

Compare25/51/103 adaptive queries with exactly as many fixed multiscale queries
around the original point, using the same encoder, weights and predicted cost.
For seeds17/29 all1190 model/image cases retain the original point at stage1,
so51-query answers are bitwise identical. In seed43 only one of595 cases moves
at stage1;51-query fixed/adaptive action differences remain at most1.193e-7.
No useful feedback-specific advantage is established. At103 queries effects are
small and mixed, not a large iterative-refinement gain. Predicted risk reduction
does not guarantee true error reduction. The fixed grid is a specific matched
control, not an exhaustive search of all nonadaptive designs.

[Two-seed control](cc_v5/two_seed_search_control/report.md),
[third-seed measurements](cc_v5/seed43_search_control/summary.json),
[control protocol](../research/cc_v5_search_control_protocol.md).

## Compute and verification

Each model has3,097,189 parameters; each best checkpoint is12,599,089 bytes.
Recorded peak training tensor allocation is approximately844–845MiB for ordinary
arms and901MiB for derivative supervision, including233MiB of image cache plus
cloned common model/optimizer state. Driver-reserved VRAM is not this metric.
Training ran on RTX4060 8GB. V5 batch-1 latency, inference VRAM, ONNX/TensorRT
deployment and calibrated selective thresholds are **NOT MEASURED**. V4 timing
must not be silently relabeled V5. No export optimization was done on this screen.

Independent FP64 recomputation used real validation GT. All36 CPU checkpoint
replays pass1e-3 tolerance for actions/risks/degree errors; exact receipts:
[seeds17/29](cc_v5/two_seed_cpu_replay.json), [seed43](cc_v5/seed43_cpu_replay.json).
Historical synthetic, V1/V2 and V3/V4 negative artifacts remain intact.

## Next decision and Luma relevance

1. Reject selected-action/derivative training as the next default; retain the
   ordinary action and physical transport controls and the modest selective signal.
2. Do not spend another iteration merely adding passes. Test stronger scene
   context and training-time teacher supervision against ordinary distillation,
   no-teacher and calibrated C+ controls, keeping a compact inference model.
   Distillation and semantic weighting themselves are not novel.
3. First freeze reference/method/weight identities and run the newly acquired
   paired Samsung/Oppo transfer screen. The3.23GB selection is fully acquired;
   only three TRAIN scenes have been numerically used for loader development.
   No iPhone or Android final accuracy claim exists yet.

This extends reproducible R&D of Luma's photometric normalization/reliability
component. It does not validate facial skin color, physical DeltaE, universal
camera support, recommendation quality or patent novelty. Skin-specific reference
measurements remain future work. The overall R&D goal remains active.
