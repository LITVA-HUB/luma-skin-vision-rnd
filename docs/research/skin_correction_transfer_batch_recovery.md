# Recorded evaluation-batch recovery after a strict replay failure

Initial frozen run stopped with exit1 at the original-core equality assertion
for from_SLR seed17, after16 inner cores and9 mixed-protocol heads completed.
The session was authoritatively terminal. No unseen head had been fitted or
scored. Mixed outputs had been observed; no architecture, hyperparameter, role,
candidate, loss or checkpoint-selection rule is changed in this recovery.

Cause: the old source script processed known and unseen domains separately in
batch32. The follow-up processed all264 validation images together before
splitting into132-image domains. Tail GEMM shapes differ;4/396 known-domain
Lab values differed, maximum3.8146973e-6. A direct diagnostic replay of the
same unchanged weights, separately per domain, matched BOTH historical arrays
bit for bit (max gap0). This was numeric batching, not an accuracy or data leak.

Fix: restore original per-domain batch32 processing for baseline/deployed
evaluation, then scatter predictions into the original validation row order.
No relaxation of the exact baseline assertion. The regression test uses a
batch-size-dependent callback and interleaved domains to verify this contract.

Original source_lock.json and pretest Git checkpoint remain unchanged. A new
recovery_lock.json binds only the revised runner/test and this recovery note;
all original datasets, model/training helpers and protocol bytes must match.
Existing completed banks return after SHA checks;16 completed inner cores and
9 heads are retained. Remaining fits use the same frozen fit_core/fit_head.
Final audit replays every array and original baseline with domain boundaries.
Record this recovery in the report rather than concealing the partial failure.
