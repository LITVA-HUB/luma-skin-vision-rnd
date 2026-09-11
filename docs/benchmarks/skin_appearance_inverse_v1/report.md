# Conditional appearance inversion: actual skin color

108 locally reproduced closed-form fits, original MSKCC real photographs and
instrument-native Lab. Source cohorts are repeatedly inspected; no new independent
result. No camera or capture-mode ID enters inference. All errors are DeltaE00.

Each family selects alpha using only same-camera validation patient-mean error.
Mean/direct outputs are shown below; all 60 selected endpoints including medoids,
all candidate fits and both risk rankings are retained in JSON/CSV.

| Protocol | Input / family | Alpha | Mean | Median | p95 | Common 80% | Posterior 80% |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed | rgb/direct_d1/direct | 1 | 5.0058 | 4.7501 | 9.7810 | 5.0286 | n/a |
| mixed | rgb/direct_d2/direct | 1 | 4.4408 | 3.9259 | 8.9029 | 4.4306 | n/a |
| mixed | rgb/global_d1/mean | 100 | 5.0380 | 4.3810 | 10.1929 | 5.0459 | 5.2669 |
| mixed | rgb/global_d2/mean | 0.01 | 4.7677 | 4.1009 | 9.9805 | 4.7484 | 4.9292 |
| mixed | rgb/mode_d1/mean | 100 | 4.5468 | 4.0983 | 9.0981 | 4.5563 | 4.4852 |
| mixed | rgb/mode_d2/mean | 100 | 4.5090 | 3.8403 | 9.2495 | 4.5421 | 4.5143 |
| mixed | stats/direct_d1/direct | 0.01 | 4.2919 | 3.8932 | 8.4808 | 4.2019 | n/a |
| mixed | stats/direct_d2/direct | 0.01 | 3.9046 | 3.4626 | 7.9859 | 3.7645 | n/a |
| mixed | stats/global_d1/mean | 0.01 | 4.1015 | 3.7350 | 8.1941 | 4.0571 | 4.0463 |
| mixed | stats/global_d2/mean | 100 | 3.9774 | 3.6494 | 7.6994 | 3.8968 | 3.9954 |
| mixed | stats/mode_d1/mean | 100 | 3.9724 | 3.6176 | 7.8135 | 3.8876 | 4.0296 |
| mixed | stats/mode_d2/mean | 100 | 3.8663 | 3.4440 | 7.9727 | 3.7307 | 3.8248 |
| from_SLR | rgb/direct_d1/direct | 1 | 6.3818 | 5.9836 | 11.7928 | 6.2776 | n/a |
| from_SLR | rgb/direct_d2/direct | 0.01 | 6.4935 | 6.0080 | 11.6943 | 5.9652 | n/a |
| from_SLR | rgb/global_d1/mean | 0.01 | 6.1427 | 5.6928 | 11.1749 | 5.7588 | 5.9846 |
| from_SLR | rgb/global_d2/mean | 100 | 6.1722 | 5.8002 | 11.3536 | 5.7248 | 6.1665 |
| from_SLR | rgb/mode_d1/mean | 0.01 | 5.8663 | 5.4247 | 10.7383 | 5.5765 | 5.8888 |
| from_SLR | rgb/mode_d2/mean | 1 | 6.3639 | 6.0055 | 11.5228 | 5.7968 | 6.4659 |
| from_SLR | stats/direct_d1/direct | 1 | 8.1069 | 7.9256 | 13.0242 | 7.7339 | n/a |
| from_SLR | stats/direct_d2/direct | 1 | 11.0046 | 9.1459 | 26.4756 | 8.1270 | n/a |
| from_SLR | stats/global_d1/mean | 1 | 5.1159 | 4.9410 | 9.4643 | 4.9010 | 5.0740 |
| from_SLR | stats/global_d2/mean | 100 | 5.1976 | 5.2883 | 9.3087 | 4.8991 | 5.2259 |
| from_SLR | stats/mode_d1/mean | 1 | 5.4028 | 5.0788 | 9.9324 | 5.0502 | 5.2365 |
| from_SLR | stats/mode_d2/mean | 1 | 5.5627 | 5.3713 | 10.1131 | 5.1291 | 5.4531 |
| from_ipod | rgb/direct_d1/direct | 1 | 7.4882 | 7.1118 | 14.3663 | 6.6873 | n/a |
| from_ipod | rgb/direct_d2/direct | 1 | 11.3555 | 7.9512 | 27.2912 | 7.9093 | n/a |
| from_ipod | rgb/global_d1/mean | 100 | 9.8882 | 9.0956 | 16.7137 | 9.3636 | 10.5824 |
| from_ipod | rgb/global_d2/mean | 0.01 | 9.4782 | 8.5065 | 16.8186 | 9.0456 | 10.2150 |
| from_ipod | rgb/mode_d1/mean | 100 | 8.3234 | 8.2635 | 14.9823 | 7.3980 | 8.8891 |
| from_ipod | rgb/mode_d2/mean | 1 | 6.7572 | 5.5590 | 13.9760 | 6.6990 | 6.9175 |
| from_ipod | stats/direct_d1/direct | 0.01 | 10.0085 | 9.8042 | 15.2418 | 10.0034 | n/a |
| from_ipod | stats/direct_d2/direct | 0.01 | 15.9210 | 14.7532 | 28.1996 | 14.5719 | n/a |
| from_ipod | stats/global_d1/mean | 0.01 | 7.2790 | 7.1704 | 12.1735 | 7.3425 | 7.4616 |
| from_ipod | stats/global_d2/mean | 100 | 6.9891 | 6.8382 | 11.7150 | 6.9398 | 7.2359 |
| from_ipod | stats/mode_d1/mean | 100 | 6.3231 | 5.9874 | 10.1102 | 6.4344 | 6.4013 |
| from_ipod | stats/mode_d2/mean | 1 | 7.4273 | 7.3829 | 11.2160 | 7.6410 | 7.8675 |

![Risk and coverage](risk_coverage.png)

Common input ranking has identical accept sets within each input kind. Posterior
expected color error is a separate uncalibrated hypothesis; it is not a per-image
bound or a matched calibrated C+ comparison. Simple direct controls use fewer
parameters/fit operations and no capture labels. Neural controls use all 64 patch
tokens rather than only their mean: these are not equal-capacity architectures.

## Representation and prior controls

| Protocol | Actual TRAIN color atoms | Constant prior mean error | Known-target nearest atom mean | Oracle p95 |
|---|---:|---:|---:|---:|
| mixed | 248 | 9.9678 | 1.1043 | 1.9727 |
| from_SLR | 85 | 9.2982 | 1.8851 | 4.5999 |
| from_ipod | 163 | 11.8522 | 1.6807 | 3.8978 |

The nearest-atom oracle uses true reference color and is not deployable.
Its gap to the actual model assesses a representation limitation, not causal
camera identifiability. A discrete empirical prior is not an independent physical
skin model; posterior mean remains in the TRAIN convex hull. Gaussian forward
noise includes out-of-person residual bias but is not calibrated on new devices.

## Verification and boundaries

360 exact selection/evaluation arrays; 31680 scalar evaluation cases; 1944 coverage rows; 57024 curve points.
2880 person-excluded forward folds and 3096 normal-equation checks; 357120 independent Gaussian densities.
238146 independent palette/expected-cost cases plus 98736 scalar representation/prior diagnostics.
All 108 fitted models and OOF arrays replay exactly. Direct and forward alphas
are selected separately with frozen candidate order. All results remain exploratory.
Full model NPZ archives include OOF diagnostic arrays; they are not deployment
file sizes. This is a CPU statistical falsifier; no GPU latency/VRAM/export claim.
Original MSKCC CC-BY; no new external code, weights, data or participant publication.
Independent MSKCC primary mean remains 4.4570/80%4.1591, ordinary fusion
4.3005/4.1447. Ordinary phone facial accuracy and novelty remain unproved.
[Decision](../../research/skin_appearance_inverse_next_decision.md).

## Opponent check: restrict direct predictions to the same empirical support

Post-hoc controls were frozen separately after the original source screen.
They use predicted color and TRAIN atoms only. Hull projection uses standardized
Lab Euclidean distance; nearest-atom uses DeltaE00. Neither uses image truth.
Alpha is reselected on the same-camera source validation for each control.
Their full common-ranking curves are included in the CSV.

| Protocol | Input / degree | Control | Alpha | Mean | Common 80% |
|---|---|---|---:|---:|---:|
| mixed | rgb / 1 | nearest_atom | 1 | 4.9556 | 4.9765 |
| mixed | rgb / 1 | hull | 1 | 5.0179 | 5.0243 |
| mixed | rgb / 2 | nearest_atom | 1 | 4.4803 | 4.4981 |
| mixed | rgb / 2 | hull | 1 | 4.4126 | 4.4016 |
| mixed | stats / 1 | nearest_atom | 0.01 | 4.3301 | 4.2748 |
| mixed | stats / 1 | hull | 0.01 | 4.2219 | 4.1763 |
| mixed | stats / 2 | nearest_atom | 0.01 | 3.9550 | 3.8514 |
| mixed | stats / 2 | hull | 0.01 | 3.8947 | 3.7535 |
| from_SLR | rgb / 1 | nearest_atom | 1 | 6.1301 | 6.2784 |
| from_SLR | rgb / 1 | hull | 1 | 6.3680 | 6.2811 |
| from_SLR | rgb / 2 | nearest_atom | 0.01 | 6.1184 | 5.9868 |
| from_SLR | rgb / 2 | hull | 0.01 | 6.4014 | 5.9138 |
| from_SLR | stats / 1 | nearest_atom | 1 | 7.5751 | 7.6407 |
| from_SLR | stats / 1 | hull | 1 | 6.4663 | 7.2777 |
| from_SLR | stats / 2 | nearest_atom | 1 | 8.0457 | 7.4017 |
| from_SLR | stats / 2 | hull | 1 | 6.8433 | 6.6357 |
| from_ipod | rgb / 1 | nearest_atom | 0.01 | 7.0825 | 6.3822 |
| from_ipod | rgb / 1 | hull | 1 | 6.8119 | 6.5634 |
| from_ipod | rgb / 2 | nearest_atom | 1 | 9.1717 | 7.7193 |
| from_ipod | rgb / 2 | hull | 1 | 8.7614 | 7.3345 |
| from_ipod | stats / 1 | nearest_atom | 0.01 | 9.9289 | 9.7768 |
| from_ipod | stats / 1 | hull | 0.01 | 9.4229 | 9.2399 |
| from_ipod | stats / 2 | nearest_atom | 0.01 | 14.3709 | 13.2198 |
| from_ipod | stats / 2 | hull | 0.01 | 13.7872 | 12.3453 |

Support audit: 2356992 scalar atom distances, 25344 scalar color cases, 432 coverage rows and 12672 curve points.
1575 projected selection/evaluation instances satisfy hull feasibility and KKT stationarity; unchanged interior predictions remain exact.
These instance counts are computational checks, not additional people or photos.
The empirical support control reduces some direct-transfer failures but does
not generally match inverse regression. Neither family beats the strongest
archived compact image models universally. No calibrated selective win.
