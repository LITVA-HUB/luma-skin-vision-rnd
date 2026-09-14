# Luma ChromaSeed-WA model card

Same-trajectory equal weight means of frozen LT checkpoints, deployed as the unchanged NP643-parameter36->16ReLU->3 head. Input one finite prepared color36 vector, output native instrument D65/10-degree Lab3. Predictor in scripts/chromaseed_neural_prefix_numpy.py takes(36,); predict batches Nx36. All normalizers and original head metadata preserved; no new gradient training in primary exports. No cross-seed mixing or query-dependent inference loop.

2,886numericB,643parameters,5,456cached-arrayB,about6.4us prepared-feature single-thread CPU. FP32 payload/normalization, FP64 remaining consumer arithmetic. Image/face/skin/feature extraction and archive/runtime/code overhead are additional. Equal weight mean differs from a mean of predictions; no ensemble deployed.

91recipes/role,273 inner candidates,15 frozen policies. Selected overall errors5.716113/8.381401/8.640083 versus LT5.716113/8.294346/8.683471 and NP5.770540/8.294346/8.589532. Both selected averaged transfer outcomes remain worse than initial NP. No universal quality promotion. Original TRAIN966rows/24people only; repeated/confounded exploratory roles, no ordinary-phone facial accuracy claim.

63 final payloads include all selected methods, matching endpoint controls and baseline, not63 independently chosen deployment winners. Follow policies.csv and full source/selection binding; preserve adverse outcomes. Ten tests,2457inner scalar mean checks/464100OOF vectors,63final models/789525actual consumer calls audited.810 parent checkpoint payloads and189 final exports reconstructed bitwise after27 original singleton fits and9 complete LT banks. Pure averaging~0.15ms; all-method/control three-seed construction~6.028/5.856/5.980s with original upstream and fixed18-slot costs, not one-head training time. No new data, assets, packages or external publication. Upstream terms continue to apply.

[Evidence](../benchmarks/chromaseed_weight_average_v1/report.md) · [Protocol](../research/chromaseed_weight_average_v1_protocol.md).
