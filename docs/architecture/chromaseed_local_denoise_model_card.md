# ChromaSeed-ND model card

Research-only five-family comparison on prepared 36-dimensional skin-region color features. Output is native instrument D65/10-degree Lab. Actual consumer scripts/chromaseed_local_denoise_numpy.py accepts exactly one finite (36,) vector; trace returns each block's clean prediction. No target, hidden reference photo or inference seed is passed. Fixed zero initialization and two/four unshared steps for local variants. Backpropagation remains inside each local block.

Plain 2,763 parameters / 11,364 numeric B; local2 2,758 / 11,344; local4, blind4 and e2e4 2,764 / 11,368. Blind4 has192 inactive state-input coefficients, giving2,572 active parameters. FP32 payload, FP32 input normalization, FP64 cached NumPy arithmetic. Local4 cached arrays22,504B, excluding Python objects. Face localization, feature extraction, camera calibration, model code and runtime dependencies are outside payload size and latency.

No universal replacement: local4 mixed error5.700539 exceeds plain5.636588 and FG5.438652. Its transfer errors8.495492/8.324291 improve over FG8.597000/8.705018 on reused, camera/person-confounded roles; descriptive FG comparison intervals include zero. E2e4 reverse8.026476 is stronger and faster to train than local4. No ordinary-phone facial or cosmetic shade-match accuracy established.

Local4 full standalone RTX4060 fits about103/286/287ms for mixed/forward/reverse selected settings; CPU response25.5/25.7/25.9us on one Ryzen7900X thread. All45 complete selected-setting replay fits exactly reproduced payloads. No global backprop-free training or superiority to the original NoProp paper claimed. New data, weights and packages were not used.

Training and selection use original TRAIN only, with person-disjoint inner folds and fixed seed/rate/step grid. Exposed old validation/calibration/test are excluded. Upstream data/code rights remain those in the existing provenance archive; no external NoProp implementation was copied.

[Full evidence](../benchmarks/chromaseed_local_denoise_v1/report.md) · [Protocol](../research/chromaseed_local_denoise_v1_protocol.md).
