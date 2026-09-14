# ChromaSeed Neural Readout Implementation Plan

> For agentic workers: use superpowers:executing-plans sequentially. AGENTS.md disallows delegation. The user's autonomous local-experiment instruction supplies authorization; keep the current isolated worktree. No commit/publication is part of this study.

**Goal:** Test whether analytical readout on the fixed compact TG network improves the accuracy/complete-training-cost tradeoff.

**Architecture:** Random, Adam1/4/16 and full3 Gaussian1/4/16 bases, fixed learning parameters. Two fit-only readout objectives and three positive alpha values. Fold every fitted readout back into the existing FP32 TG payload and use its unchanged NumPy consumer.

**Tech Stack:** Existing Python3.12/NumPy2.5/SciPy1.18, one CPU thread. Independent pivoted QR and scalar TG reference. No installation.

**Spec:** [Registered protocol](../../research/chromaseed_neural_readout_v1_protocol.md).

## Global constraints

Only original TRAIN hashd7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0. No agents/new data/images/weights/packages/publication/exposed validation/test. Preserve TG receipt19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098 and older chain. TG/FG/C/H/X/A read-only; never G/GS mains. Explicit original Python and all BLAS thread variables1. Full goal remains active.

## Task 1: exact output solution and independent reference

Create `scripts/chromaseed_neural_readout.py`, `scripts/chromaseed_neural_readout_reference.py`, `tests/test_chromaseed_neural_readout.py`.

Interfaces: `fit_head(source,x,y,weights,family,alpha)`→FP32payload,diagnostics; `solve(z,t,metric,weights,alpha)`→B,residual; `representation_bank(x,y,weights,group,seeds)`→bases,trajectories; `fit_single(x,y,person,site,group,basis,family,alpha,seed)`→model,metadata. Reference `refit(source,x,y,weights,family,alpha)`→model,diagnostics owns activation normalization, analytic tensor and augmented QR.

- [x] Write failure-first scalar and correlated-output checks. Example scalar oracle: `Z=column_stack([ones(n),x]); B=lstsq(vstack([sqrt(w)[:,None]*Z,[[0,sqrt(alpha)]]]),concatenate([sqrt(w)*y,[0]]))[0]`. Verify unpenalized bias and exact unchanged first-layer arrays.
- [x] Implement `A[a::3,b::3]=Z.T@((w*M[:,a,b])[:,None]*Z)`; add alpha only after first3 diagonal entries; `rhs=(Z.T@(w[:,None]*einsum('nij,nj->ni',M,T))).ravel()`. Cholesky solve and normalized residual. Norm specialization uses65×65 with3RHS. Record arrays and export drift.
- [x] Implement independent QR via whitened row blocks from Cholesky(M).T, `sqrt(w)`, and penalty rows with zero intercept; use `lstsq(...,lapack_driver='gelsy')`. Check perceptual tensor against primary finite differences on synthetic valid Lab and native-output drift.
- [x] Test invalid arrays/alpha/weights, constant/dead features, mean3 unused-feature invariance, same numeric size and actual TG consumer. Test representative seed/prefix export equality with TG's independent scalar reference. Use meaningful tolerances from protocol.

## Task 2: frozen banks and selections

Create `scripts/chromaseed_neural_readout_train.py`; run `experiments/runs/chromaseed_neural_readout_v1`.

- [x] Bind inherited TG sources plus new core/reference/runner/tests/protocol, parent receipt/verifier/bank/model/OOF/inventory inputs. Recheck hashes before fit. Save source lock before opening new real-data training phases.
- [x] Train12 representation trajectories per bank through16epochs, construct42 bases and252 heads, retain36 unchanged networks/7 FG controls. Atomic bases/models/receipt and inner OOF archives; bank reuse requires source/row/file/settings consistency.
- [x] Merge9 inner banks into501500 OOF outputs; compute252 candidate scores,84 alpha choices and12 quality policies, then atomically freeze selection before final banks.
- [x] Complete three final banks and evaluate381 selected/control consumers over33 transforms. Save5020818 row outputs and all12573transform/1524dose summaries. Preserve negative cases and exact parent controls.

## Task 3: independent verification and true construction cost

Create `scripts/chromaseed_neural_readout_audit.py`, `scripts/chromaseed_neural_readout_runtime.py`.

- [x] Independently reconstruct all144 representation trajectories and72 random bases; compare432 learned checkpoint arrays/order/floor counters and matching frozen TG controls.
- [x] Refit all3024 output solutions with independent QR/tensor, verify unchanged hidden payloads, all OOF/final/selection/policy fields, dose summaries and paired descriptive intervals. Write bound audit receipt with all counts and maximum discrepancies.
- [x] Profile381 actual consumers and378 complete construction repeats after audit. `fit_single` must reconstruct the learned base rather than consume a cache. Record initialization/query/fit times, bytes, numerical drift and exact repeats. One warmup/two measured seed17 fits; one CPU thread.

## Task 4: complete report and immutable receipt

Create report/verifier scripts, outputs under `docs/benchmarks/chromaseed_neural_readout_v1`, model card and next-decision.

- [x] Generate every selected base/loss/group/role, policies, adverse comparisons, training/inference/storage costs and all references in CSV/JSON/report. Plot comparisons and inspect images. Clarify prior readout work and reused-data/phone-validation limits.
- [x] Run numerical suite/lint, verify full hash chain, original-repository cleanliness and terminal workers. Seal report/audit/runtime/model-card sources and artifacts once, then fresh read-only verification must retain receipt hash. Never rewrite old verifiers.
- [x] Update mutable AGENTS/current-status/goal ledger/plan with measured outcomes and process handles. Full goal stays active unless actual scope is proven complete; no background work claim when processes are terminal.


## Execution and final verification

Completed2026-09-13. Fresh TG read-only59007 exit0, original receipt unchanged.24 preflight numerical tests1.47 s; meaningful missing-module failures observed before primary/reference implementations. Early tests passed a matrix into one-row TG Predictor: verified existing API and corrected test usage; separate TG model-card interface erratum retained and bound as NR input. No old source/receipt rewritten.

PrimaryPID18004/session82004 exit0,61.634 s. All144 representation trajectories and3024 head solutions completed, source/selection frozen before outer evaluation. Initial audit79371 completed numerical checks but exit1 at extra mistaken504-exact-count assumption. Protocol required original2e-6atol/rtol and measured exact counts, not universal bitwise equality; corrected only audit counting, added explicit non-bitwise record, and reran full audit11895 exit0,141.416 s.503/504 bases exact; one inner/slr_to_ipod/fold1/tagi_full3_e16_raw36_s29 w1 drift7.45e-9. QR maximum5.81e-6Lab, selected stress2.48e-6, below original.001. No numerical tolerance or frozen primary changed.

Runtime51100 exit0,97.794 s; all378 complete construction fits exact. Report generated129 rows/516dose rows/252candidates/12policies/168paired comparisons; both scientific figures visually inspected. Initial verifier46923 exit0,24 final tests1.44 s,36local links, source chain777entries/891diagnostic inputs. Read-only verifier77690 exit0 preserving receipt3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65. Parent TG/FG/C/H/X/A verified read-only; original repository clean; no active study worker. Plan remains mutable by receipt contract.

Same10564/1855B deployment; mixed random norm complete5.929/5.079ms but error6.048158/6.788076.54/72 unchanged-model comparisons improve;79/84 lose to FG, all12policies lose.77/84alpha choices and all12policies at10. Next inspect regularization geometry/inner curves before a bounded stronger-alpha study on identical bases. Planned, not implemented/launched. Full goal remains active, not blocked; ordinary-phone facial/end-to-end high quality unvalidated.
