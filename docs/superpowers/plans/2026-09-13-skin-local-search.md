# Compact Local Search Implementation Plan

> Execute autonomously under the user's explicit authorization to resume experiments.

**Goal:** Implement and measure an error-guided small regressor and adequate controls on original TRAIN only.

**Architecture:** Tested float64 torch numerical core; independent experiment runner for fit-only preprocessing, nested person roles, analytic fits and Adam control; frozen selections followed by exploratory outer evaluation.

**Tech stack:** Python, NumPy, PyTorch 2.8 CUDA 12.8, RTX 4060 8 GB, pytest.

**Spec:** docs/research/skin_local_search_v1_protocol.md.

**Constraints:** Preserve old archive, no external test use or uploads, no identifiers in tracked output, no claims beyond observed evidence.

1. Numerical core and meaningful independent tests (implementation_status agent owns scripts/skin_local_search_core.py and tests/test_skin_local_search_core.py).
2. Runner, protocol/source locks and resumable artifacts (root owns scripts/skin_local_search_train.py and research docs).
3. Independent split and numerical review (research_evidence agent, read-only).
4. Run synthetic plumbing check, relevant tests, frozen GPU experiment. Fix implementation faults transparently; no adaptive outer tuning.
5. Measure model storage/inference and independently audit saved predictions; publish local aggregate report and current research status.

## Execution ledger

- Completed: isolated branch codex/skin-local-search-2026-09-13 at f7e56d7; original archive preserved.
- Completed: baseline CPU unit suite, 41 passed; RTX 4060 CUDA availability verified.
- Completed: numerical core and runner, independent numerical/split reviews.
- Completed: 330 primary GPU fits, 216 authoritative prefix refits, 99 precision artifacts; all choices frozen before respective outer evaluations.
- Completed: CPU/GPU profiling, independent 33-array metric audit, 61 fresh relevant tests, Ruff and diff checks.
- Completed: aggregate reports, standalone PNG/SVG comparison, outcome/next-decision document and reproduction commands.
- Recorded exception: compact evaluation bookkeeping amended after fitting to preserve immutable selection manifest; fit-time lock and explicit amendment retained. Prefix cost-receipt predecessor is superseded, preserved in ignored history. No model or hyperparameter changed by the amendment.
