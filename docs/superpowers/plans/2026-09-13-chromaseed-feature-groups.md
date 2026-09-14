# ChromaSeed feature groups implementation plan

> For agentic workers: use superpowers:executing-plans sequentially. No delegation is authorized. The user's autonomous instruction authorizes local implementation and experiments without an additional approval checkpoint.

**Goal:** Test whether subsets of the existing color statistics improve the compact kernel's measured accuracy/cost tradeoff while preserving exact references.

**Architecture:** Seven fixed input groups, A analytic static/joint readouts and imported X16 controls. Inner group/alpha choices freeze before all final fits; independent refits and NumPy execution check the result.

**Tech stack:** Existing Python3.12/NumPy/SciPy/PyTorch environment, CPU one thread. No dependency or frozen source changes.

**Spec:** [registered protocol](../../research/chromaseed_feature_groups_v1_protocol.md).

## Global constraints

Only original TRAIN hash d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0. No agents, new data/weights, uploads or publication. Use original-repo .venv Python explicitly. Preserve C/H/X/A read-only receipts and all older bound artifacts; never run G/GS verifier mains. Full goal remains active unless ordinary-phone/end-to-end high quality is actually established.

## Task 1: numerical implementation and consumer

Files: scripts/chromaseed_feature_groups.py, scripts/chromaseed_feature_groups_numpy.py, tests/test_chromaseed_feature_groups.py.

- [x] Write numerical tests for feature contracts, held-input isolation, exact raw controls, aliases, independent QR refits and selection rules. All 14 passed before primary fitting.
- [x] Implement fit_bank, fit_single, predict, gate_score, model_id and selection helpers. Reuse frozen arithmetic without editing it.
- [x] Implement and validate the actual one-row NumPy consumer; run numerical tests and lint.

## Task 2: freeze and run complete registered banks

File: scripts/chromaseed_feature_groups_train.py. Run: experiments/runs/chromaseed_feature_groups_v1.

- [x] Bind inherited sources, C verification, A/X banks and inventory context; freeze before fitting.
- [x] Run nine inner banks, independently merge query indices and freeze all96 group/24 policy choices.
- [x] Fit all final banks, then evaluate all291 cases under33 transformations; capture terminal process evidence and operation counts.

## Task 3: independent audit and cost

Files: scripts/chromaseed_feature_groups_audit.py, scripts/chromaseed_feature_groups_runtime.py; use the existing independent X QR/SVD reference on explicitly gathered subsets.

- [x] Audit all bank membership, raw/imported payloads, OOF metrics/selection and final predictions. Refit all288 selected nonconstant models independently.
- [x] Profile291 actual NumPy consumers and384 standalone fits; require exact payload reproduction and retain timing samples.

## Task 4: scientific decision and receipt

Files: scripts/chromaseed_feature_groups_report.py, scripts/chromaseed_feature_groups_verify.py; docs/benchmarks/chromaseed_feature_groups_v1 and new model-card/next-decision docs.

- [x] Generate all-group results, adverse camera transfers, real storage/runtime and limitations; inspect plots if produced.
- [x] Verify frozen sources, artifacts and numeric counts; preserve the final receipt on read-only rerun.
- [x] Update mutable current-state docs, leave the full goal active and describe only measured progress.

Initial receipt: 459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908; 14 tests pass in 1.46 s. The plan is mutable and identified by path in the receipt; primary sources, numerical outputs and scientific report artifacts are frozen separately. [Report](../../benchmarks/chromaseed_feature_groups_v1/report.md), [arithmetic count erratum](../../research/chromaseed_feature_groups_count_erratum.md). The full goal remains active; no background FG job remains.

Fresh read-only verification after mutable status updates: session 80187, terminal exit 0, the same receipt SHA256. Original source/input chain and C/H/X/A receipts preserved. All implementation-plan tasks are complete; the broader model goal is not complete.
