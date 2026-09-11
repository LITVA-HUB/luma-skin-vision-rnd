# Stable-color correction improves this internal screen; OOF is not the winner

Original MSKCC CC-BY real photographs and instrument-native skin Lab. Only original TRAIN was loaded.18 people/734 photographs train the final core/head;6 people/232 photographs are internal evaluation. These same people were used in earlier exploratory screens. Source VALIDATION and independent CAL/TEST remain unopened. Both cameras occur in support and evaluation: this is not unseen-camera evidence.

## Actual skin color results

Means and quantiles below average three individual seed results, not ensemble predictions or pooled quantiles.

| Correction training | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |
|---|---:|---:|---:|---:|---:|
| base | 6.1082 | 5.1353 | 14.9960 | 15.37% | 5.6609 |
| in_full | 5.8579 | 4.8002 | 13.6359 | 13.79% | 5.4313 |
| in_matched | 5.6560 | 4.6887 | 12.9644 | 12.50% | 5.2608 |
| out_person | 5.8945 | 4.8164 | 13.4503 | 14.94% | 5.5260 |

The strongest local arm in_matched lowers mean error by7.40% versus its unchanged core, from6.1082 to5.6560. Its p95 decreases14.9960 ->12.9644 and at80%5.6609 ->5.2608. All three correction arms improve the core in all three seeds. This is a useful source component result, not an independent universal-model victory.

The tested explanatory hypothesis fails: out_person5.8945 is worse than in_matched5.6560 and in_full5.8579 on the seed-averaged endpoint. Correct OOF construction is necessary when claiming excluded-person supervision, but did not produce the best color correction here. More realistic base error targets alone are not sufficient.

| Arm | People improved vs core, after seed averaging | Mean person error change |
|---|---:|---:|
| in_full | 4/6 | -0.2548 |
| in_matched | 5/6 | -0.4607 |
| out_person | 5/6 | -0.2287 |

These are six reused people and three correlated seed runs; no significance or population-generalization claim follows. The later choice of a leading arm must be followed by a separately frozen camera-transfer experiment, not selection of different winners per camera.

## Mechanism and matched controls

The reference bank and512-dimensional neural context were removed. A stable39-vector combines36 fixed color descriptors with3 native base-color estimates. A188,035-parameter39->384->384->64->3 residual head corrects the frozen929,297-parameter core. Total1,117,332 parameters, under the authorized1,129,297 cap. One image, one core and one head at inference; no camera ID, query reference or reference-memory payload.

Nine new cores were fitted from scratch:three inner folds x three seeds. Each inner core saw12 support people; six inner-query people were absent from its training and its target scales. Each fit used930 updates, the same fixed recipe as the existing18-person final cores. No new epoch selection occurred.

in_full trains its correction on predictions of the18-person core that saw each support person. in_matched uses a12-person core that saw that person; out_person uses the corresponding12-person core that excluded the person. The latter two reuse exactly the same inner cores, with one encoder per training image and no averaging. Their roles are routed cyclically to keep core data size/budget equal. At evaluation every head uses the same18-person core. Thus the12-to18 support-size deployment shift remains shared by the two inner-core arms.

Heads share initialization, zero initial correction,300 steps, image draws, target scales, capacity and standardized Lab MSE loss. Original full-core predictions replay exactly. Learned hidden coordinates from separately fitted encoders are never mixed. A changed representation and training recipe distinguish this screen from the previous551-input head screen; it is not an isolated ablation proving that context removal alone caused the gain.

## Supervision diagnostic

| Prediction table on support people | Mean base error DeltaE00 |
|---|---:|
| in_full | 3.5823 |
| in_matched | 3.3675 |
| out_person | 4.3241 |

The out_person table has larger base error, as expected for genuinely excluded-person queries. Its head still does not beat the matched inclusion control. These table errors describe supervision, not final head accuracy on new people. The head itself is trained on all18 support people; only its out_person base predictions are OOF.

## Risk, compute and previous evidence

Fixed100/95/90/80/70/60% coverage uses identical nearest-support color36 distance for all arms.72 individual fixed-coverage checks and24 aggregate rows are retained; full per-image curves remain in private reproducible arrays. This is not calibrated expected error or a refusal guarantee. Older reports used a different mean-patch novelty ranking; do not mix their80% figures with this ranking.

![Risk curves and individual-seed results](correction_results.png)

The same internal cohort previously had ordinary person/site balancing6.0084 and color allocation6.0012 ([allocation report](../skin_color_sampling_v1/report.md)); current in_matched5.6560 is lower, with extra head capacity/optimization. These are historical contextual comparators, not equal-total-budget refits. The independent4.3005 fusion result concerns a different endpoint and is not directly comparable to this internal6.1082 baseline.

Inner-core allocated CUDA training peak107.97–107.97MiB; cached-head peak75.54–75.57MiB. Complete core+head checkpoints4477653–4477653bytes, including78 scale scalars and no bank. Training peaks are process allocations with caches; no new isolated inference VRAM, latency or export measurement. Previous phase latency is not transferred to this model.

## Verification, prior art and Luma boundary

9 inner and3 original core replays;9 exclusion/scale checks;4404 prediction routing rows;one complete core andthree complete head refits with exact final state hashes.9 NumPy head replays;12 exact evaluation arrays;9390 scalar CIEDE2000 values and72 coverage checks. Largest NumPy native Lab gap4.02e-6; scalar color gap3.56e-15.

[Wolpert,Stacked Generalization,1992](https://www.sciencedirect.com/science/article/pii/S0893608005800231) already describes learning corrections from base predictions on withheld data. The publisher abstract was verified in fresh search; no paper code/weights or author result numbers were adopted. This is a known-method falsifier and measured component improvement, not an invented stacking architecture.

For Luma, the result concerns actual instrument-referenced skin color, not illuminant angle. It does not prove ordinary smartphone facial accuracy, unseen-camera reliability, cosmetic decisions or novelty. Independent evidence remains primary4.4570/80%4.1591 versus ordinary fusion4.3005/4.1447. The practical goal of median<=2,p95<=5 at>=80% on unseen ordinary phones is unmet.

[Frozen protocol](../../research/skin_crossfit_correction_protocol_v1.md) · [Next decision](../../research/skin_crossfit_correction_next_decision.md)
