# Compact error-guided search, version 1

Status: PLANNED and frozen before real fits, 2026-09-13. User explicitly resumed research.

## Question

Can GPU-batched selection of nonlinear atoms by training-error reduction produce a small native skin-Lab regressor with useful person generalization and lower training cost than backpropagation? This is an exploratory mechanism experiment, not a claim of algorithmic novelty or phone-face validation.

## Data boundary

Only original `skin_mskcc_pixels_v1/train.npz` is permitted, SHA256 `d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0`. Explicit absolute source path required. Use color36, native instrument Lab, person, site and camera metadata. Do not load RGB, validation, calibration, test or transfer caches. The original TRAIN has 966 images / 24 people: 8 SLR, 16 iPod. No identifiers or row-level predictions in tracked reports.

Three fixed outer roles, all exploratory and overlapping within original TRAIN:

1. Mixed: the historical support-curve 18/6 people (734/232 images), exact role digest `7fa14adc41525a868d737539571c9d8cb9e954a680e457f828d7aab35a2bc064`. These six people were used historically and are not a fresh confirmation set.
2. SLR to iPod: fit 8 SLR people, evaluate 16 iPod people.
3. iPod to SLR: reverse direction, fit 16, evaluate 8.

Three inner folds per outer fit, formed by seeded (917031) per-camera permutation of unique people followed by array_split into 3 groups. All models use identical folds. All scalers, widths, centers and coefficients fit on the current inner training role only. Final models refit on the complete outer-fit role after selecting hyperparameters exclusively by inner held-person errors. Finish and persist selections for all protocols before evaluating any outer predictions. Cross-protocol overlap forbids treating the three outcomes as independent replications.

## Fixed methods and selection budget

Input: 36 color statistics, standardized with fit-only image mean and std (floor 1e-6); targets likewise standardized. Fit weights give equal mass to people, then to sites within each person, then to images within site; rescale weights to mean one. Camera is never a model input.

- Linear ridge, bias unpenalized, alpha in {0.1, 1, 10}.
- Exact Gaussian kernel ridge, same alpha grid and weights. Target uses the common fit image-mean/std; no additional kernel intercept. Solve (K + alpha*diag(1/w))c=y, corresponding to weighted squared error plus alpha times RKHS norm squared. Width is the fit-set median nonzero RMS pairwise distance. All fit vectors count toward inference payload.
- Random RBF + linear skip, 64 atoms, same alpha grid.
- Error-guided RBF + linear skip, 64 atoms, same alpha grid. At each step choose the atom giving the largest exact decrease of weighted regularized standardized-Lab SSE conditional on previously chosen atoms. Compute all candidates' gains with a GPU Schur-complement update. This is training optimization, not an error-confidence head; its training-error decrease alone is not evidence of generalization.
- Adam MLP, color36 to 64 SiLU to 3, plus linear skip 36 to 3. Learning rates {0.0003, 0.001, 0.003}, weight decay 0.01, 512 updates of batch64 sampled with replacement by fit weights. Direct standardized-Lab objective. Same approximate final storage budget as the 64-atom RBF including centers, coefficients and scalers. No early stopping based on outer errors.

RBF candidate bank: at most 256 distinct fit observations. First take one feature medoid per (person,site), then round-robin across shuffled people/site lists, followed by additional observations if fewer than 256 sites. Selection uses geometry and metadata, never evaluation labels. Expand each center with widths median-distance times {0.5,1,2}. Random and guided arms have the identical bank per seed/fit role and final atom budget. Multiple widths of one center are allowed and counted. Seed 17,29,43 for stochastic methods, deterministic ridge/KRR run once per fold/config. Hyperparameter choice (alpha for RBF, learning rate for MLP) for each stochastic family minimizes average person-mean inner DeltaE00 across seeds; no seed cherry-picking. MLP and RBF objective/regularizer differ and this comparison tests end-to-end methods rather than isolating optimizer alone; random versus guided RBF isolates selection.

## Measurements and artifacts

Primary: person-balanced mean CIEDE2000 on native Lab. Also image mean/median/p90, site-then-person mean, per-camera results. Report seed-average individual models; any ensemble must be separate and include multiplied payload. Report weighted training SSE and inner errors, fit seconds with CUDA synchronization, total tuning time, peak allocated VRAM, exact serialized file bytes and numeric payload bytes including data-derived centers and preprocessing. CPU model-only batch1 latency separately from preprocessing; do not claim camera-to-result latency from this. No confidence interval on seeds as if they were independent people.

Before runs, save cache/protocol/source-code hashes and environment. Store row-level arrays and model weights only in gitignored experiments/runs/skin_local_search_v1. Tracked docs contain aggregates only. No changes to archived experiments. Code has independent NumPy/exhaustive objective tests. Report losses as well as wins, and never equate synthetic checks with actual data measurements.

## Prior art

The ideas are related to [OMP](https://ieeexplore.ieee.org/document/4385788), [Stochastic Configuration Networks](https://ieeexplore.ieee.org/document/8013920/), [ELM](https://doi.org/10.1016/j.neucom.2005.12.126) and [random-feature kernel approximation](https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). Implementation of these mechanisms is not itself a breakthrough. The potential contribution must be a measured accuracy/storage/training-cost result for Luma's task, subsequently checked on genuinely new facial acquisition.

