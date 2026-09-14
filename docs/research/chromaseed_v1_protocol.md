# Luma ChromaSeed v1: palette-to-skin transfer

Status: PLANNED, frozen before real fits on 2026-09-13. User explicitly authorized autonomous continuation and proposed palette-first learning. Working research name: **Luma ChromaSeed**, not a trademark-clearance statement.

## Hypothesis and interpretation

Pretraining a small regressor on generic color surfaces may learn a useful color representation before fine-tuning on instrument-labelled skin crops. A clean-palette control separates learning RGB/Lab geometry from attempting invariance to illumination/camera rendering. Randomized rendering is a simplified engineering simulator, NOT a spectral model of skin, a measured camera profile, or a guarantee of color constancy. One uniform RGB patch cannot uniquely identify both unknown surface color and unknown illumination. Real transfer must be measured independently of synthetic loss.

This is an extension of the previous exploratory original-TRAIN series. Its mixed and cross-camera held groups were already exposed historically and in the preceding run. No fresh independent confirmation is claimed. Original validation/calibration/test must not be opened. No images, data or model files are uploaded.

## Data and model

Real source is exclusively original `skin_mskcc_pixels_v1/train.npz`, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Load color36, instrument-native Lab, person/site/camera metadata only. Reuse the previous exact mixed18/6 roles and SLR→iPod / iPod→SLR roles; nested 3 person folds remain identical. Fit sample weights equalize people, then sites, then repeat images. Camera is split metadata, never a model input.

Every arm uses the same 36→64 SiLU→3 MLP plus 36→3 linear skip, 2671 learned scalars plus 78 fixed preprocessing scalars = 2749 floats / 10,996 bytes FP32. Unlike the previous experiment's fitted normalizers, use common fixed physical input normalization: 27 quantiles +3 mean channels centered at0.5/scaled0.25,3 standard deviations centered0/scaled0.2,3 correlations centered0/scaled1. Output native Lab uses mean[50,0,0], scale[25,30,30]. No real-data scaler can contaminate pretraining or other roles.

## SYNTHETIC palette construction

32768 fit colors and4096 validation colors, independent fixed seeds731013/731014. Uniform sampling in Lab box L[10,95],a[-80,80],b[-80,90], rejecting colors outside encoded sRGB[0.02,0.98]. Targets are exact canonical D65 Lab for those colors. There is no fitting to real skin colors, test statistics or proprietary palette.

Each color is represented by128 synthetic samples of one surface: zero-mean per-sample shading (std uniform[0,0.04]) and independent linear-RGB texture noise (std[0,0.003]), clipped to physical range. Clean observations are these samples encoded as sRGB and rounded to8-bit. Randomized observations additionally apply per-surface exposure exp(U[-0.45,0.45]), zero-log-mean RGB gains exp(N(0,0.16)), a near-identity color-mixing matrix I+N(0,0.025) normalized by row sums, black offset U[-0.005,0.005], clipping, sRGB encoding, gamma exp(U[-0.12,0.12]), saturation U[0.85,1.15], and sensor/output noise stdU[0,0.003]. This is an explicitly approximate rendering family. Clean and randomized observations share canonical colors/textures; labels are the original surfaces' Lab, never the rendered RGB converted to Lab.

Compute exactly the color36 definitions: quantiles[.01,.05,.1,.25,.5,.75,.9,.95,.99]×RGB, mean3, population std3,3 upper-triangle correlations. Synthetic sampling is lower-resolution than the real128×128 crop statistics; this domain gap is retained as a limitation, not hidden.

## Arms and training budgets

Seeds17,29,43; identical architecture and initial weights for each seed. AdamW wd0.01, batch256, MSE in fixed-normalized Lab, gradient norm clip5. Pretraining/warmup:2048 updates, lr0.001, no early stopping. Fine-tuning resets optimizer for ALL arms, including scratch controls, with LR{0.0003,0.001,0.003};1024 updates. Save fixed64/256/1024-step checkpoints. No early stopping by outer metrics.

Enable deterministic PyTorch algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, one CPU thread and disable TF32. No mixed precision, compilation or external training service is used in this version.

1. **scratch**: random initialization → real1024 updates.
2. **clean_palette**: clean synthetic2048 → real1024.
3. **rendered_palette**: randomized synthetic2048 → real1024.
4. **shuffled_palette**: same rendered observations with a fixed shuffled canonical-label pairing2048 → real1024 (negative information control).
5. **skin_long**: same current-fit real data2048 lr0.001 → real1024 (matches extra optimizer updates/batch size, not wall time or data diversity).

Pretraining is shared across real protocols/folds only because it uses synthetic data exclusively; its cost is separately accounted once and not described as free. Real warmups in skin_long are fit separately for each training fold/final role and never see that fold's held people. Fine-tuning minibatch index sequences are identical across arms/LRs for the same seed/role. All checkpoints contain full inference parameters, no teacher at inference.

Primary endpoint uses1024-step checkpoint; select LR separately for each arm/protocol by mean person-balanced native DeltaE00 of concatenated inner OOF predictions, averaged across all3 seeds. Learning curves64/256 use that same primary-selected LR; do not retune LR at each budget. All primary choices, final checkpoints and code/data hashes freeze before outer evaluation. Synthetic validation metrics are diagnostic only and cannot select pretraining duration or renderer strength in this version. Zero-shot synthetic models are scored on real data only at the final diagnostic evaluation, never used to select a winner.

## Measurements and decision

Primary: person-balanced DeltaE00, plus image mean, site-person mean, p90, per-camera errors and learning curves. Compare pretrained arms to both scratch and skin_long; a win over shuffled labels alone is insufficient. Report individual-model seed means, not an ensemble. Report synthetic fitting/validation loss separately from instrument skin metrics, preparation/pretraining/adaptation times, serialized bytes and numeric payload.

Retain all outcomes, including negative transfer. A candidate is worth further study if palette pretraining improves person generalization versus both real-only controls without paying for a larger inference model. The three reused protocols cannot establish ordinary-phone face accuracy or patent novelty. If transfer fails, preserve the simulator and result as a falsified version, not evidence that all palette pretraining is impossible.

## Prior art and provenance

[Tobin et al.,2017](https://arxiv.org/abs/1703.06907) motivates randomized synthetic appearances but studied a different task; its results are not Luma results. [Afifi et al.,CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Afifi_When_Color_Constancy_Goes_Wrong_Correcting_Improperly_White-Balanced_Images_CVPR_2019_paper.html) describes the difficulty introduced by nonlinear camera rendering after white balance. Only papers were consulted; no third-party data/code/weights downloaded. Existing original MSKCC attribution applies separately to real-data fine-tuning. Synthetic colors and simulator are locally generated, and no face photos are generated.
