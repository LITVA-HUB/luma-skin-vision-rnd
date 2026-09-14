# Luma ChromaSeed-LT model card

Long continuation of the accepted NP643-parameter blind head:36->16ReLU->3. One prepared color36 vector produces instrument-native D65/10-degree Lab. Existing scripts/chromaseed_neural_prefix_numpy.py Predictor accepts one(36,) input; predict batches Nx36. Keep NP metadata and fit-only FP32 normalizers. The historical family='blind4' and original_k/prefix fields retain lineage; LT uses ordinary within-head backprop/AdamW, not a new four-block denoising training pass.

FP32 numerical payload2,886B,643 parameters,5,456B cached consumer arrays, about6.4us single-thread prepared-feature response. NPZ archive, code, runtime, image decoding and face/skin/feature extraction are additional. CPU arithmetic FP64 after FP32 normalization. No new quantization claim.

Only original TRAIN966rows/24people; exact per-fold ND warm starts for inner training, exact NP exports for final fitting. Modes0/16/256 bounded digital variants, unchanged target by synthetic assumption, half clean probability in augmented modes. No new measured people or external images. Six checkpoints through131072;216 trajectories,1296 records including216 baseline aliases. Reused exploratory roles and camera/person confounding remain.

Inner choices prefer no augmentation in all three roles:2048/0/2048 additional steps. Overall errors5.71611/8.29435/8.68347 against initial5.77054/8.29435/8.58953; mixed modestly improves, forward unchanged, reverse worsens. At131072/256variants/lr.0003:5.46195/9.12002/9.13320, an exploratory long-run comparison, not selected deployment settings. There is no universal model promotion or independent facial-phone quality result.

All324 final models use the same NP consumer. Their directory is historically named selected, but only selected_per_mode/selected_overall flags identify the frozen12 policies. Baseline0 duplicates are exact aliases. Consult policies.csv and source/selection/results receipts before choosing a payload; do not choose from known outer accuracy.

Independent audit validates all4,269,672 final vectors and actual consumers. Rebuilt27 original models/NP exports and81 selected payloads are bitwise identical. Full selected construction costs1.425/0.874/1.422seconds per three-seed role recipe; all upstream fits and complete18-slot banks charged, not individual-model latency. Full primary workflow266.44seconds reuses saved warm starts. Eight behavioral tests pass. No photos, pretrained weights, packages or external publication added. Existing upstream licenses apply.

[Evidence](../benchmarks/chromaseed_long_training_v1/report.md) · [Protocol](../research/chromaseed_long_training_v1_protocol.md).
