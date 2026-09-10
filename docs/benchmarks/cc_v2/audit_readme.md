# Reproducing the independent v2 audit

From the repository root, with Python 3.11+, NumPy and Git available:

```powershell
.venv/Scripts/python scripts/audit_cc_v2_metrics.py
```

The script writes a timestamped `independent_metric_recheck_*.json` beside this file. To choose its name:

```powershell
.venv/Scripts/python scripts/audit_cc_v2_metrics.py --receipt docs/benchmarks/cc_v2/my_recheck.json
```

Existing receipts are refused. Exit code 0 means all audit checks pass and the recorded comparison fields exactly reproduce the original review; mismatches return 1. The script runs independently of the working directory; `--repo-root PATH` can select a relocated complete checkout. Historical absolute Windows paths are mapped from their recorded workspace prefix into that checkout, while original evidence files remain unchanged.

Required local inputs include the three processed caches/manifests, all 25 run/selector artifacts and source snapshots, saved evaluation reports, original v1 evidence, and Git commit `7637d6d`. These data/model artifacts are not all distributed in Git; a code-only checkout cannot complete the audit. Restore exact bytes and original file modification times when transferring the artifact bundle. Timestamp checks intentionally fail if those times are lost. No GPU, PyTorch, scikit-learn, training or model inference is required; models are hashed but never unpickled.

The script is a preserved copy of the original scratch auditor with path/output handling adjusted. Metric formulas, SHA-ID tie order, tolerances, hash checks and selection checks are unchanged. Every rerun compares its status, counts, maximum deltas, per-run legacy deltas and chronology with the immutable [original receipt](independent_metric_review.json). A rerun receipt has its own script hash and timestamp, so whole receipt bytes are expected to differ.

The [preserved-code recheck](independent_metric_recheck_committed.json), run from outside the repository on 2026-09-10, passed with exact equality across all 14 comparison fields. Four path-resolution cases and refusal to overwrite that receipt were also checked. Original review SHA-256 remained `44e881765e2f2408e3ac28c2665d78deeddbf47b02f14d388d0392c854835926`.

The [review](independent_metric_review.md) explains scientific limits: legacy-upgrade fitting residuals were not persisted; real data contain no invalid rows; timestamps do not prove absence of unrecorded computation; official source/Sony are regression sets; fresh384 has unverified true scene clusters and mirror identity. The auditor checks per-run and aggregate arithmetic, not bootstrap assumptions, full-dataset claims, skin DeltaE, novelty or commercial rights. It deliberately retains the frozen final-lock digest and original v1 commit rather than treating this as a generic auditor for later experiments.
