# V5 paired correction-critic source screen

**REUSED DEVELOPMENT VALIDATION ONLY.** 1126 real SimpleCube++ training images;119 reused validation images. CC BY4.0. No external weights.

All arms share exact warmup model/BN/optimizer/scheduler state per seed. Strict CUDA determinism passed epoch1 replay. Same data and noise streams; gradient loss adds backward compute. No new camera/test labels were consumed.

| Seed | Arm | Best epoch | Best mean repro ° | Final mean repro ° | Best raw risk80 ° |
|---|---|---:|---:|---:|---:|
| 17 | point | 73 | 2.5192 | 2.6400 | not trained |
| 17 | posterior_random | 73 | 2.7126 | 2.7999 | 2.3670 |
| 17 | action_random | 63 | 2.4461 | 2.5938 | 2.1980 |
| 17 | transport_random | 108 | 2.3649 | 2.5222 | 1.9157 |
| 17 | transport_policy | 79 | 2.4607 | 2.5799 | 2.0738 |
| 17 | transport_gradient | 81 | 2.4676 | 2.6788 | 2.1928 |

Best checkpoints were selected on these same119 images. Both best and final values are retained to expose selection effects. Fixed coverage100/95/90/80/70/60, complete119-point curves, recovery/tail metrics and refinement diagnostics are in summary.json. Nominal80% accepts95/119 images.

Point-only risk is omitted because its critic was not trained. No post-hoc risk calibration has been performed. More refinement has no guaranteed improvement; equal-query nonadaptive control remains pending.

The model retains3,097,189 parameters. Recorded peak training allocation includes source tensors and the copied common optimizer/model state; elapsed run time includes validation/checkpoint writing. Deployment latency is NOT MEASURED for V5.

This screen cannot establish universal camera handling, physical surface-color DeltaE, facial accuracy, published-method superiority or novelty. Independent camera evaluation and conventional calibrated C+ comparison remain required after a stable development result.
