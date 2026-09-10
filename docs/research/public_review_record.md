# Independent public-milestone review

2026-09-10. Research/license specialists reviewed original rights and prior art; a separate code/report reviewer checked the implementation and results.

Verified: actual SimpleCube++ uint16 dimensions and exact target rectangle; all2234 timestamps; disjoint internal capture-date splits; held-out residual targets; recovery/reproduction formulas. Reviewer recomputed all80 method/domain result blocks from saved predictions/GT/scores across8 runs. Maximum mean/AURC discrepancy2.7e-15. No metric inflation or concealed stronger control was found.

Resolved findings:

- Data binding: training now fingerprints Cube/Sony NPZ and manifests; evaluation rejects changes. Verification confirms all8 runs match current caches.
- Capacity/cost: report distinguishes966,760 stored versus964,131 baseline-active parameters, and adds full-path CPU/GPU profiling alongside model-only latency.
- Historical source hashes: exact pre-format full source hash recovered by reverting only Sony decoder line wrapping. [Snapshot, diff and AST check](../benchmarks/source_snapshots/verification.json) prove no behavioral difference between paired seed runs.
- Fixed malformed documentation links/commit typography. New local evidence links checked; imported audit references to an unavailable original Luma repository remain untouched.
- Added oracle/random diagnostic curves and conditional Sony/Canon bootstrap; explicitly listed unexecuted follow-up ablations rather than imply exhaustive protocol completion.

[Final verification](public_verification.json):56 tests pass, Ruff lint and format checks pass, git diff whitespace check passes, lockfile check passes. Eight complete real60-epoch trainings/evaluations are preserved. Existing48-test/34-command synthetic evidence is retained separately, not relabeled.

No external publication, upload of dataset images, commercial training on restricted data, imported pretrained weights or paid cloud compute occurred.
