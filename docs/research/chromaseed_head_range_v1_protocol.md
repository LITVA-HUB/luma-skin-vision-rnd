# ChromaSeed HR v1: change residual range, preserve the matched learning experiment

The previous goal turn is verified progress: Seg1 completed training, held evaluation and latency measurement; additional16TRAIN spectral face cubes were acquired and numerically checked. HR tests a separate actionable finding from AS inner data:27.15/34.37/26.61% equal-person targets lie outside the necessary +/-one-target-standard-deviation correction box. This is not proof that the bound explains all error. New spectral data will be prepared separately; it does not enter this matched experiment.

## Fixed contrast

Preserve all seven AS architectures (patch_small,patch5m,soft_small,soft5m,dynamic_small,dynamic5m,pool5m), parameter counts, warm NP parents, token/color normalizers and four-pass recurrence. Three heads: unit=tanh(z), wide=4*tanh(z/4), linear=z. All have value0 and slope1 at zero. Four-pass increments still multiply the selected head by0.25. Unit calls the frozen AS function exactly. Dynamic connections still involve dense token scoring; they are not a sparse speed claim. No new architecture novelty is claimed.

Only original MSKCC TRAIN cache966observations/24people with native measured Lab is loaded. Preserve original mixed/slr_to_ipod/ipod_to_slr roles and three patient-disjoint inner folds. These roles are historically reused exploratory evidence, not fresh phone validation. Legacy validation/calibration/test and LaPa test are excluded. Spectral and rendered colors are not substituted for native labels.

## Training and selection

Six fixed slots: seeds17/29/43 × learning rates1e-5/1e-4, unchanged order. Batch64, original weighted replacement sampling and independent seed generators, AdamW decay.01, norm clip5, deterministic FP32 and TF32 disabled. Original cosine horizon8192; checkpoints128/512/2048, stop at2048. Original .6mean-pass MSE + .4final-pass MSE + .001connection penalty; frozen fit-only normalization. CUDA graph warmup resets weights, moments, learning rates and counters before step1.

Reuse verified unit-mode AS checkpoints and scores at matching rate/steps. Two new heads ×7architectures ×3roles ×3folds =126new inner banks,756trajectories,2268checkpoint model payloads. Every fit/query row array and warm parent must match the corresponding AS bank. Verify all inherited AS artifacts once before freeze; verify each consumed bank's stored hashes. No saved new receipt is treated as complete without matching code, parent, selection and artifact hashes.

For each role, score21architecture/head pairs ×6settings + frozen NP/WE controls =128candidates;384scores total. Rank by mean equal-person native DeltaE00 across three seeds, then p90, numeric bytes, steps, learning rate; original fixed candidate order breaks complete ties. These are separate seeded models, not an ensemble. Select21per-pair policies,7per-architecture policies and one overall policy per role:87choices. Unit per-pair settings must reproduce AS. Freeze all choices before any new final fit or held-role inference.

Fit every wide/linear per-pair selection from the original matching warm parents:42new six-slot final banks. Total168new banks/1008trajectories. Export three chosen-rate models per pair, all-pass outputs and native metrics. Include63original unit results and18NP/WE controls plus126new results =207records. Record every outcome, including losses. Any overall selected old control is a legitimate result.

## Gates, storage and observability

CPU adapter/fitter receipt46bbefcf1155c85a8a2097409910cc95e35680c228684111ccee5b5f03a96b2d remains immutable. Synthetic CUDA preflight checks every architecture: old/new unit16steps with8/16checkpoint payload equality, plus wide/linear64steps; original0.002nativeLab absolute/1e-6relative CPU/GPU prediction tolerance and6.5GB peak allocated limit. No tolerances may be changed because a case fails. Source/preflight/config are frozen before production fitting.

New raw arrays live in D:/Luma-RnD/chromaseed_head_range_v1 via the existing project junction. Require40GB free before freeze and15GB before each bank. Preserve partial files and failures; never restart based only on an observation timeout. Progress records actual PID, bank, completed steps, loss and elapsed time; no new scheduler is needed for this finite workflow. A process failure remains explicit. Train one GPU job at a time; record actual costs separately from preflight and later timing probes.

After primary terminal success: independently reconstruct splits, normalizers, candidate metrics and all selections; check exported predictions and each recurrent pass using the independent NumPy consumer within unchanged tolerances. Measure actual response/training costs separately after relevant GPU work stops. Final report must separate inherited unit controls, newly fitted outcomes and any post-screen diagnostics. Do not promote a selected model from held-role errors, silently narrow the grid, retrain on exposed tests, or declare full ordinary-phone cosmetic shade accuracy from this experiment.
