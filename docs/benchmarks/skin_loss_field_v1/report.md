# Native skin-color loss-field experiment

36locally reproduced real-image fits; native MSKCC instrument Lab ground truth.
All color errors are DeltaE00. No fresh TEST/CAL or ordinary-phone result.
Source validation has been inspected repeatedly. Prefit checkpoint:
`checkpoint/skin-loss-field-pretrain-2026-09-11`.

Each result averages three separate seed scores, not an ensemble. For atom
models the primary answer minimizes a learned field on15,625fixed candidate
colors; secondary is the atom-weighted Lab at the same selected checkpoint.
Direct is the ordinary continuous regressor. Both field losses share a full
Gram factor; no spectral truncation or invented instrument reference is used.

| Protocol | Arm | Primary mean | Median | p95 | At80% | Secondary mean | Negative risk |
|---|---|---:|---:|---:|---:|---:|---:|
| mixed | direct | 3.4406 | 2.9776 | 6.9316 | 3.4407 | 3.4406 | 0.00% |
| mixed | soft_ce | 3.7810 | 3.1755 | 7.9785 | 3.5943 | 3.6319 | 0.00% |
| mixed | risk_simplex | 4.0426 | 3.6861 | 8.2604 | 4.0284 | 3.6971 | 0.00% |
| mixed | risk_affine | 3.7598 | 3.3510 | 7.6275 | 3.7475 | 3.6247 | 0.00% |
| from_SLR | direct | 5.0193 | 4.8340 | 9.7825 | 5.1087 | 5.0193 | 0.00% |
| from_SLR | soft_ce | 5.5461 | 5.3680 | 10.3963 | 5.4338 | 5.6316 | 0.00% |
| from_SLR | risk_simplex | 5.6813 | 5.1236 | 11.0181 | 5.5767 | 5.4695 | 0.00% |
| from_SLR | risk_affine | 4.9641 | 4.2494 | 10.1217 | 4.9285 | 5.0315 | 0.00% |
| from_ipod | direct | 5.9635 | 5.5527 | 11.0896 | 5.7914 | 5.9635 | 0.00% |
| from_ipod | soft_ce | 7.4636 | 7.4083 | 12.2443 | 7.5342 | 6.8011 | 0.00% |
| from_ipod | risk_simplex | 9.5897 | 8.6170 | 18.7014 | 9.8045 | 8.8714 | 0.00% |
| from_ipod | risk_affine | 8.9582 | 8.8379 | 15.4983 | 9.1635 | 8.2969 | 0.00% |

soft_ce is a standard smoothed categorical objective (fixed width2DeltaE00).
risk_simplex learns candidate loss with nonnegative unit-sum weights.
risk_affine removes nonnegativity; it is not a probability estimator.
Negative predicted risks, where present, are unsupported expected errors.
No score is post-hoc calibrated or a per-image error bound.

## Matched primary comparisons

Patient intervals are descriptive on reused source cohorts, without
multiplicity adjustment. Camera protocols also change people and capture mode.

| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |
|---|---|---:|---|---|
| mixed | risk_simplex vs soft_ce | 0.2616 | [-0.0324,0.4731] | False |
| mixed | risk_affine vs soft_ce | -0.0212 | [-0.4137,0.3948] | False |
| mixed | risk_affine vs direct | 0.3192 | [-0.0067,0.6266] | False |
| from_SLR | risk_simplex vs soft_ce | 0.1352 | [-0.2456,0.3710] | False |
| from_SLR | risk_affine vs soft_ce | -0.5820 | [-0.9001,-0.1265] | True |
| from_SLR | risk_affine vs direct | -0.0552 | [-1.1498,0.5679] | False |
| from_ipod | risk_simplex vs soft_ce | 2.1261 | [0.7878,3.1179] | False |
| from_ipod | risk_affine vs soft_ce | 1.4945 | [0.4661,2.1933] | False |
| from_ipod | risk_affine vs direct | 2.9947 | [0.2548,4.8980] | False |

## Post-hoc grid limitation checks

Projecting the frozen ordinary answer onto the same grid uses no true color
during projection. The separately labeled oracle uses the true reference and
is only a grid approximation diagnostic, never an image-accuracy result.

| Protocol | Direct projected mean | Known-target oracle mean | Oracle p95 |
|---|---:|---:|---:|
| mixed | 3.5335 | 0.7264 | 1.1592 |
| from_SLR | 5.0652 | 0.7121 | 1.0729 |
| from_ipod | 5.9965 | 0.6926 | 1.1732 |

![Risk and coverage](risk_coverage.png)

All six fixed coverages and full curves are retained. Expected losses over a
finite TRAIN color dictionary need not describe a new-camera distribution.
Known-target grid feasibility does not imply inverse image identifiability.

## Compute, verification and remaining scope

Stored parameters range951,142-993,033.
Maximum learned checkpoint3,977,773bytes; shared fixed palette up to16,507,670bytes.
Maximum fit-process GPU allocation129.60MiB.
Fixed tables are not trainable parameters; they are required by the field
decoder. Direct does not mathematically need the table despite shared harness.
No latency, ONNX, or deployment promotion. No foundation inference/camera input.

108exact color arrays; 12672scalar color cases;
432coverage rows and12672curve points; scalar palette audit and full Gram checks pass.
FP64 field products check the selected action with explicit FP32 roundoff bounds.
Original MSKCC CC-BY; no external data, weights or CIE tables adopted here.
Independent benchmark remains unchanged4.4570/80%4.1591; ordinary fusion
4.3005/4.1447 is stronger. Novel superiority and facial-phone accuracy unproven.
