# ChromaSeed AS model card

Seven patch/pooling/four-pass soft/dynamic residual regressors at17,374/15,246 and approximately4.85–4.96million deployed parameters. Includes a frozen643-parameter NP anchor; all residual weights are FP32. Training and CUDA inference FP32, NumPy consumer FP64 after FP32 input normalization. Inputs color36 and64x18 descriptors; native D65/10-degree Lab output. No identity or ethnicity inference.

See ../benchmarks/chromaseed_architecture_scale_v1/report.md for all selected results, negative outcomes, latency and full construction cost. Original TRAIN966observations/24people only, repeatedly reused roles and camera/person confounding; no independent phone-face validation. This architecture scale screen is progress, not completion of the broader quality goal.
