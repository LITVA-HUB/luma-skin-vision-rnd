# ChromaSeed Gaussian Learning Implementation Plan

> For agentic workers: use superpowers:executing-plans sequentially. Existing AGENTS.md disallows delegation; the user's autonomous instruction already authorizes local implementation and experiments without an additional permission checkpoint. Keep this existing isolated worktree.

**Goal:** Measure whether local Gaussian weight updates improve the error, training cost and deployment size of a fixed small color regressor.

**Architecture:** d→64 ReLU→3, matched Adam, stabilized diagonal TAGI-style inference and full3 output-covariance variant. Fixed mean3/raw36 inputs, fit-only moments, three seeds, 1/4/16/64 epochs. Inner choices precede final fit; independent scalar inference audits the vectorized primary engine.

**Tech Stack:** Existing Python3.12, NumPy2.5/SciPy1.18, PyTorch2.8 for independent gradient checks, one CPU thread. No installation.

**Spec:** [Registered protocol](../../research/chromaseed_gaussian_v1_protocol.md).

## Global constraints

Original TRAIN hash d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0 only. No agents, images, new data, weights, packages or publication. Preserve FG verification459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908 and every older binding. FG/C/H/X/A verifier mains are read-only after receipt; never G/GS mains. Explicit original-project .venv Python and BLAS thread environment1. No concurrent heavy jobs. Keep the full model goal active until actual phone-face/end-to-end quality is established.

## Task 1: mathematical updates and tests

Files: `scripts/chromaseed_gaussian.py`, `scripts/chromaseed_gaussian_reference.py`, `tests/test_chromaseed_gaussian.py`.

Interfaces: `initialize(d,seed,hidden=64)` returns four FP64 mean arrays and four positive variance arrays; `gaussian_step(mean,variance,x,y,weight,sigma,kind)` updates leading independent bank slots and returns floor diagnostics; `adam_step(mean,m,v,x,y,weight,lr,step)` uses the same bank layout. `train_block(x,y,weights,group,cases,checkpoints)` returns checkpoint model payloads and per-trajectory records. `fit_single` wraps one case with full preprocessing. Scalar reference owns its initializer/normalizer/update arithmetic.

- [x] Write tests before implementation and observe missing-module failure. Dense Gaussian fixture: scalar-output design a, diagonal prior v, noise r; `expected_mean=m+(v*a)*(y-a@m)/(r+(v*a*a).sum())`, `expected_var=v-(v*a)**2/(r+(v*a*a).sum())`. Compare local output-layer conditioning and frozen-hidden first-layer cases; include a three-output Cholesky fixture and ReLU derivative-zero cases.
- [x] Implement the protocol equations. Compute all reverse quantities from pre-update weights, then mutate means/variances. Count floor events before clamping. `np.linalg.inv` handles only batched3x3 innovations for full3; diag uses elementwise reciprocals. Non-finite state raises an error.
- [x] Independently implement scalar layerwise smoothing with Cholesky solves and explicit posterior hidden marginals, then update incoming parameters through their conditional gain. Implement ordinary scalar Adam and verify gradients/multiple steps against PyTorch float64 autograd/Adam on deterministic synthetic inputs.
- [x] Test trajectory independence and prefix consistency on a fixed synthetic matrix, including different seeds/noise and one-case versus multi-case fitting. Test exact group mapping, x/y normalization, finite/shape validation and tie preference.

## Task 2: deployable mean network

File: `scripts/chromaseed_gaussian_numpy.py`; extend the same test file before implementation.

Interface: `Predictor(payload)` validates FP32 network arrays and optional increasing uint8 indices, caches FP64 weight arrays; `call(color36)` returns native Lab3, `cached_array_bytes` counts stored arrays.

- [x] Test batch/one-row output equality, ignored-feature invariance, malformed payload rejection and expected numeric bytes1855/10564.
- [x] Implement input contract, FP32 normalization then FP64 ReLU network and output rescaling. Keep uncertainty/Adam state out of deployed payloads; save training-state byte counts in the training record instead.

## Task 3: registered training banks

File: `scripts/chromaseed_gaussian_train.py`; run `experiments/runs/chromaseed_gaussian_v1`.

- [x] Build lock from inherited FG sources plus new primary, reference, consumer, runner, tests and protocol. Bind FG receipt/verification source, FG bank/OOF arrays and inventories. Check original cache hash and all old bindings before fitting.
- [x] Run9 inner banks with54 trajectories each and four checkpoints, importing exact FG norm_static/raw36+mean3 alpha.1 seed controls and constant. Retain all223 records/bank and every379,100 OOF output. Write receipts atomically with row hashes; only reuse a bank if every binding matches.
- [x] Score216 hyperparameter/epoch candidates, select18 method/group/role pairs by person mean, p90, fewer epochs, hyperparameter order; atomically freeze selection before any final fits.
- [x] Run54 final trajectories through64 epochs and retain216 checkpoints/21 controls. Evaluate all237 clean curves, then75 selected/reference cases with33 transformations. Preserve all outcomes, source/settings/model/prediction hashes, timing/floor counters and terminal handle evidence.

## Task 4: independent validation and timing

Files: `scripts/chromaseed_gaussian_audit.py`, `scripts/chromaseed_gaussian_runtime.py`.

- [x] Check every source/input and parent control, independently merge OOF row indices, recompute metrics and choices. Independent consumer math checks94,642 clean and988,350 transformed outputs, plus300 dose summaries.
- [x] Retrain each of54 selected-hyperparameter trajectories independently to64 epochs, compare216 checkpoint arrays/predictions and variance counters. Keep all maximum discrepancies and fixed-prediction paired descriptive intervals.
- [x] Profile75 actual consumers,54 new-network and18 FG-reference full standalone timing fits after audit passes. Report initialization, numeric/cache/archive/training-state bytes, query median/p95 and selected-epoch complete fit samples; no hidden teacher cache or image-processing claim.

## Task 5: report, decision and receipt

Files: `scripts/chromaseed_gaussian_report.py`, `scripts/chromaseed_gaussian_verify.py`; report under `docs/benchmarks/chromaseed_gaussian_v1`, new model-card and next-decision docs.

- [x] Generate every method/group result and epoch curve, adverse outcomes, floor diagnostics, source attribution, cost and current limitations. Visually inspect any generated scientific plots.
- [x] Run appropriate numerical tests and lint, seal immutable code/artifact/parent bindings, then verify read-only without rewriting receipt. Check terminal process state and original-repository cleanliness.
- [x] Update mutable AGENTS/current-state/active-goal/plan only with measured outcomes. The Gaussian comparison cannot establish ordinary-phone face quality; keep full objective active unless stronger evidence actually exists.


## Execution and final verification

Completed2026-09-13. Primary PID26824/session6175, audit99472 and runtime47215 all terminal exit0; no worker remains. Numerical preflight18 tests passed2.41 s. New report preflight found a Windows path unicode escape in a document template; changed that new template to a raw string before report generation and receipt sealing. No frozen numerical source changed. Report lint and generation pass; both figures visually inspected.

Initial verifier64071 terminal exit0, final18 tests2.32 s,34 local Markdown links checked. Read-only rerun66734 also terminal exit0 and preserved verification19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098. Parent FG/C/H/X/A receipts verified read-only; original repository clean. Immutable report/audit/runtime/model-card/next-decision/source bindings are sealed. This plan remains mutable by explicit receipt contract.

All540 trajectories,2160 learned checkpoints and84 exact FG imports checked. All54 independent scalar refits reproduce216 exports exactly;72 complete timing fits exact. New networks are smaller and faster to query, but slower to train;16/18 lose to FG and7/12 Gaussian comparisons lose to Adam. Full3 avoids the observed diag variance-floor case. Full goal remains active, not blocked: ordinary-phone face/end-to-end high quality unvalidated. Next matched analytical-readout study is planned, not implemented or launched; earlier random/palette frozen-readout work is recorded as prior art.
