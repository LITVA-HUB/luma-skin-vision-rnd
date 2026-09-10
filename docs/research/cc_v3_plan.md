# Color-frame posterior R&D implementation plan

> Use superpowers:subagent-driven-development for bounded implementation and independent reviews. The user's autonomous-work authorization replaces routine design/implementation confirmation prompts.

**Goal:** Test a fundamentally different color-frame graph/posterior architecture against matched controls and compact published methods on real public illumination data.

**Architecture:** Max-volume per-image linear color frame, canonical relational tokens, graph diffusion, directional posterior and camera-space risk transport. Ordinary direct and diagonal modes use the same graph/heads. All mechanisms are hypotheses; preserve every negative result.

**Tech stack:** Existing Python3.12/PyTorch2.8/NumPy/scikit-learn; RTX4060; no new dependency or paid service.

**Spec:** [cc_v3_spec.md](cc_v3_spec.md).

## Global constraints

- Preserve old source code, data, tests, weights, locks and milestone tags; new modules are `scripts/cc_v3_*`.
- No proprietary facial collection, external publishing, pretrained/foundation weights or cloud inference.
- Separate estimator/validation/risk/calibration/outer-camera data; no old test fitting.
- Source-first numerical/real feasibility, then frozen new-camera evaluation; no fakeΔE00 or novelty claim.
- Budget-match direct/diagonal/frame; track invalid/fallback fractions, memory and all failed variants.

### Task1: numerical/model core

Create `scripts/cc_v3_model.py`, `tests/test_cc_v3_model.py`. API is fully specified in the spec. Tests first: random positive invertible color mixing maps output equivariantly when support/frame are unchanged; diagonal transforms do too; rank1 and zero/nonfinite/negative inputs refuse; finite forward/backward; same parameter count across modes; scalar VMF normalizer matches direct formula away from overflow; changing RGB mixing changes transported reproduction risk without changing canonical posterior. Implement and run only CPU constructed probes before independent review. Report parameter count, numerical failures and design deviations. No real target access.

### Task2: source-only runner and measured screen

Create `scripts/cc_v3_experiment.py`. Consume `ColorFramePosteriorNet` and frozen `cc_v2_statistics.read_npz_rows`; only source train/val rows are decoded during fitting. Train three modes at fixed120epochs/batch32/seeds17 initially, AdamW0.001/cosine and fixed loss coefficients recorded before results. Save exact scripts/source/data/IDs/checkpoints, validation curves/predictions, frame invalid fractions and GPU allocator peaks. Run source frame-conditioning audit first; shorten only a declared smoke command, never mislabel it a full comparison. Confirm promising comparisons across17/29/43; retain all failures. Compare validation with fixed V2 references, not old target errors.

### Task3: original-source prior art and FFCC control

Create `docs/research/cc_v3_prior_art.md` with primary-source citations and honest overlap. Read original FFCC MATLAB and paper, pin source/license metadata; distinguish faithful components from protocol/optimizer changes. Then implement a bounded compact baseline in separate `scripts/cc_v3_ffcc.py` with histogram/torus/convolution/decoding/likelihood tests and source-only training. No claimed author-score reproduction without matching protocol.

### Task4: independent unused multicamera population

Create new V3 provenance/manifests/data outputs, never edit V2 downloader or manifests. Derive exact256-minus128 remainder and freeze ID/hash/rights before image/GT inspection. At most~4.5GB initial new transfer, within remaining disk/bandwidth; original BY-SA and mirror-identity caveat retained. No old test IDs or pixels can fit. Document separate BY-SA training-track status; train-only-CCBY4 models remain separately identified. Decode/inspect grouping before declaring LOCO roles; lock outer-camera absence and source-only choices before target errors. Full experiment may span goal turns.

### Task5: evidence, hypothesis revision and next experiment

Independent algorithm/data/metric review. Record real measured gains/losses, matched controls, invalid conditions and uncertainty limits. Any architecture change creates new immutable run IDs. Update research/IP/Skolkovo documents only with actual evidence; publish nothing. Leave the user's broader innovation goal active while required research remains unverified.
