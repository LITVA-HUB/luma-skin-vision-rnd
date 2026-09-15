# C OOF recipe forensic audit — 2026-09-15

## Scope and result

Read-only search of available workspace, tracked source, reports, archived source snapshots and all 109 reachable Git commits. No model fitting, no parameter search, no endpoint selection. The requested C architecture/50+50 duration are known from the owner, but the available repository does not identify the complete original optimization trajectory. Older experiments cannot safely fill its missing parameters.

## Parameter evidence

| Item | Recovered evidence | Status for exact C replay |
|---|---|---|
| Real sample scope | Owner: 966 TRAIN rows / 24 persons; original MSKCC role code and metadata restored by prior audit | Known; source data retained |
| Architecture | Owner: Linear(36,64), ReLU, Linear(64,3), 2563 parameters | Known |
| Epoch count | Owner: 50+50 real-training epochs | Known count; second-stage trajectory unknown |
| Seed | Prior conversational context identifies seed17; owner latest list omits exact RNG implementation | Seed integer known from context; torch/NumPy/order RNG usage and offsets unknown |
| color36 order | Historical scripts/skin_mskcc_pixels.py: 9 RGB quantiles (.01,.05,.1,.25,.5,.75,.9,.95,.99), mean3, population SD3, correlation RG/RB/GB | Historical definition recovered; C cache identity not verified |
| JPEG preprocessing | Original pixel protocol: EXIF transpose, RGB bytes, central int(.8*min(W,H)) square, Lanczos128, explicitly NO ICC conversion | Original protocol recovered; C reportedly used ICC-to-sRGB in conversational history. Current full capture audit finds all966images have no embedded ICC and orientation1, so that particular switch cannot alter these recovered pixels; exact C feature cache still unavailable |
| Input normalization | Various older scripts use incompatible strategies: no x normalization, fixed analytic moments, or fit-only population moments | C values/calculation precision/floors absent |
| Target normalization | Older scripts use native target mean/std or fixed Lab moments; these are different experiments | C exact moments, precision, ddof/floor absent |
| Target convention | Original MSKCC instrument native D65/10-degree triplicate-mean Lab; raw repetitions restored | Known reference contract |
| Optimizer | Multiple older experiments use AdamW, NumPy Adam, L-BFGS, etc. | C optimizer not found |
| LR / weight decay / betas / epsilon | Older experiment values exist but are not C provenance | Not recovered |
| Batch size / row ordering / replacement weights | Older protocols differ, and C trainer absent | Not recovered |
| 50→50 transition | Could preserve or reset optimizer, scheduler, RNG, LR and epoch index; owner duration does not decide these | Not recovered; 100 uninterrupted epochs cannot be silently substituted |
| Loss / clipping / early stopping | Exact C loss/scales/gradient clipping not present in available tracked files | Not recovered |
| Saved folds | Separate folds forensic task; no private_folds/run1 source ever appeared in tracked filenames searched | Do not substitute old source-validation-selected or later ChromaSeed folds |
| Runtime versions | Current audit versions do not establish old torch/NumPy/Pillow versions | Not recovered |
| Checkpoints | No C trained checkpoint found in current workspace/repo | Missing |
| Aggregate reproduction target | mean5.050436 median4.423216 p95 11.007970; tail rates from prior accepted result | Historical comparison only, cannot reverse-engineer weights or train parameters from aggregate values |

## False matches excluded

1. `scripts/skin_mskcc_train_ablation_v2.py` + `skin_mskcc_vote_v2.GlobalColorMLP`: input36 but architecture 36→512→768→512→256→3 with SiLU, 80 epochs AdamW .001 / weight_decay .01 / cosine to .00001 / batch32. Checkpoint chosen by source-validation patient-balanced mean. This is not C and violates latest TRAIN-only reproduction if executed as-is.
2. `scripts/skin_mskcc_oof_risk.py`: six-fold PatchVotes, three seeds, 80 epochs with source-validation-selected epochs. This is not the 2563-parameter C OOF study.
3. `scripts/chromaseed.py`: hidden36→64 SiLU, output3 plus linear skip; 2671 learned scalars and fixed synthetic normalization. Same hidden width does not establish the C model.
4. `docs/research/chromaseed_gaussian_v1_protocol.md`: raw36 has the same two-layer ReLU shape, but online NumPy float64 updates, Xavier Gaussian initialization, epochs1/4/16/64, own role protocol, per-person/site weights and optimizer study. This is not C's50+50 run.
5. Exact aggregate strings (5.050436 / 4.423216 / 11.007970), archive names and 50+50 appear only in newly written error-floor audit context. They do not point to an earlier saved C training source.

## Searches performed

- Exact-string workspace search for C aggregate values, `oof_anonymized`, `LUMA_TRAIN_OOF_C`, `50+50`.
- Source and protocol search for `bioskin`, `luma_face_v3`, `LUMA_USE_NOW`, old baseline value4.186740, exact36→64→3 architectural spellings.
- All-history filename search for run_luma, train_color, BioSkin, face_vN, AGENT_NEXT, train_oof, color_mlp, mlp_train, oof_c, run1: no matching tracked file.
- All-history pickaxe across relevant scripts/research/model cards/old MSKCC benchmark files for50+50, C aggregates, private_folds, luma_face_v3 and BioSkin: only current audit addition matched.
- Relevant histories: original pixel/OOF code arrived in18308a0/e8e5cab; archived ChromaSeed sources in43f8d9d; these predate missing C local runs.

Root agent separately checks remote refs, missing local commit availability, runtime and any authorized artifact sources. A full training contract is still required before any retraining can honestly pass the requested exact-C gate. Selecting optimizer settings or new folds to make the five historical aggregates match would be reverse tuning, not reproduction.

## Additional factual resolution

Existing complete `private_artifacts/error_floor/capture/capture.json` reports 966/966 JPEGs without embedded ICC profiles and all EXIF orientation1. Consequently the historical no-ICC versus conversational ICC-to-sRGB branch difference is inactive for these recovered files. This rules out one potential pixel difference; it does not supply missing normalization/optimizer/folds or establish byte identity with the unavailable C cache.
