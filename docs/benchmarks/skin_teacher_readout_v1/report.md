# Frozen teacher information for actual skin-color prediction

108 locally reproduced Ridge fits: six arms, six alphas, three source protocols.
Alpha chosen on same-camera source validation; all candidates retained. These
source camera families and validation people have already been explored.
Native instrument Lab / CIEDE2000, single-image predictions. No TEST/CAL use.

| Protocol | Features | Selected alpha | Mean DeltaE00 | Median | p95 | Mean at 80% |
|---|---|---:|---:|---:|---:|---:|
| mixed | color | 0.01 | 4.5139 | 4.0781 | 8.7596 | 4.3072 |
| mixed | teacher | 1 | 4.2651 | 3.7437 | 8.4045 | 4.2112 |
| mixed | combined | 1 | 3.9462 | 3.5933 | 7.5969 | 3.8573 |
| mixed | shuffle17 | 10 | 4.7328 | 4.4084 | 8.7900 | 4.6604 |
| mixed | shuffle29 | 10 | 4.7756 | 4.3495 | 8.6209 | 4.6722 |
| mixed | shuffle43 | 1 | 4.6024 | 4.0553 | 8.6708 | 4.6281 |
| from_SLR | color | 0.1 | 7.8220 | 7.0267 | 15.7958 | 8.0710 |
| from_SLR | teacher | 1 | 6.3118 | 4.9849 | 13.8034 | 6.4218 |
| from_SLR | combined | 1 | 5.6611 | 4.5200 | 13.5414 | 6.0811 |
| from_SLR | shuffle17 | 1 | 5.8550 | 5.3907 | 12.2588 | 5.8438 |
| from_SLR | shuffle29 | 1 | 6.0047 | 5.5173 | 12.0678 | 5.8210 |
| from_SLR | shuffle43 | 1 | 6.0191 | 5.5158 | 11.7630 | 5.6697 |
| from_ipod | color | 0.01 | 8.2597 | 8.3407 | 13.2100 | 8.6911 |
| from_ipod | teacher | 1 | 7.5572 | 7.2079 | 12.5224 | 7.5507 |
| from_ipod | combined | 1 | 8.2684 | 8.0297 | 14.6784 | 8.3628 |
| from_ipod | shuffle17 | 10 | 6.1725 | 5.8452 | 11.6615 | 6.6398 |
| from_ipod | shuffle29 | 10 | 5.8540 | 5.6859 | 10.4742 | 6.4064 |
| from_ipod | shuffle43 | 10 | 6.0944 | 5.8630 | 11.6828 | 6.6934 |

Shuffled controls independently permute teacher features within fitting and
source evaluation subsets, retaining the corresponding absolute-color features.
Permutation seeds are null controls, not extra independent people or training
replications. Readouts use a fixed SVD solution; there is no stochastic fit seed.

| Protocol | Strong historical compact model | Mean over three seed scores |
|---|---|---:|
| mixed | plain_mse | 3.4771 |
| mixed | mixture_mse | 3.4406 |
| mixed | graph_always | 3.6394 |
| from_SLR | plain_mse | 5.8301 |
| from_SLR | mixture_mse | 5.0193 |
| from_SLR | graph_always | 5.8339 |
| from_ipod | plain_mse | 5.5089 |
| from_ipod | mixture_mse | 5.9635 |
| from_ipod | graph_always | 4.9736 |

Historical models are locally reproduced with compatible source splits; they
are stronger nonlinear comparators, not capacity-matched Ridge readouts.

## Cost and integrity

The frozen teacher has **22,056,576 parameters**, with original
weights **88,283,115 bytes**. Combined readout adds only 2,415
coefficients/intercepts, but the teacher is required at inference for this screen.
This exceeds the target model budget and must not be called a compact model.
The 36-feature color-only control does not require the teacher.

1,230 vectors of 768 values were extracted from existing 128px source RGB,
upsampled to 224px. This adds no image detail. CLS and mean patch tokens are
concatenated; no teacher fine-tuning. All vectors replay exactly at the same
device and batch size. Original teacher code/weights hashes and notices checked.
Pretraining overlap with the benchmark remains unknown.

Feature extraction took 3.659s in this run, peak allocated CUDA
memory 272.75 MiB. This is batch preprocessing,
not batch-1 or end-to-end product latency. No export or deployment optimization.

Audit: 216 exact prediction arrays;
38016 independent scalar color cases;
648 independent fixed-coverage rows; all 108 weighted
normal equations checked and independently solved. Fit-only scalers, weights,
shuffles and same-camera alpha selection verified.

Five-neighbor input distance is uncalibrated density, not expected DeltaE00.
[All candidate risk curves](all_candidate_risk_coverage.csv) retain every alpha.
No ordinary facial-phone or new independent skin-accuracy conclusion follows.

[Protocol](../../research/skin_teacher_readout_protocol_v1.md), [audit](audit.json),
[next decision](../../research/skin_teacher_next_decision.md).
