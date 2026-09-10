# V6 execution status

Final update: all three seeds completed. All 24 best/final checkpoints passed
independent CPU replay and FP64 rescoring; all three warmup replays were bitwise
equal. [Final three-seed report](../benchmarks/cc_v6_report.md): generic action
3.2214° full / 2.5211° raw risk80; physical transport 3.2105° / 2.5440°.
The combination loses to V5 on reused source validation. No V6 external-camera
result or inference latency is claimed. Session 99152 finished with exit 0.
The seed17-only status below is preserved as historical progress.

Seed17 completed four120-epoch arms,3.097M parameters, same1126/119 real source
split, shared full20-epoch state, bitwise CUDA warmup replay passed. All8 best/
last checkpoints replayed on CPU and independently rescored. Source GT float32
rounding changes error by at most about2e-6 degrees; this is recorded separately.

|Seed17 arm|Best full mean reproduction|Last mean|Best raw risk80|
|---|---:|---:|---:|
|point|3.3045|3.4246|untrained|
|posterior|3.3227|3.4813|2.5974|
|action|3.3277|3.3885|2.5652|
|physical transport|3.2438|3.3728|2.7220|

Negative preliminary outcome: canonical-frame combination has worse source
accuracy than V5; physical transport has slightly better full accuracy within
V6 but worse selective risk than action/posterior. This is one seed on reused
validation, not a final multi-seed or unseen-camera conclusion. No V6 phone test.
Seeds29 then43 are running sequentially in the existing live execution; do not
launch duplicates. Source numerical scripts match; unrelated phone/docs work
may make whole-repository checkout identities differ and is not hidden.
Training allocations859.98вЂ“864.07MiB include cached data/common state. No V6
inference latency/export measured. Full suite222 passed,14 historical ONNX
warnings,37.17s before subsequent reporting-only edits and Fourier phase2.

Separate radical representation spike: Fourier ridge phase1 completed16 cases,
all independently replayed and rescored. Best12,288-coefficient variant achieved
2.7468 full mean/2.0714 raw risk80 on119, selected by full validation mean.
This is promising size/accuracy evidence, not a victory over the best CNN or a
novelty claim. Phase2 completed18 additional cases: reducing regularization changes mean only
2.7468 to2.7446; removing the learned prior worsens best mean to3.3792. All34
filters were independently replayed. See ../benchmarks/fourier_representation_report.md.
This adaptive validation search is disclosed and is not fresh-test evidence.

A further audit found the inherited oracle helper clips absolute coordinates,
whereas V6 searches residual coordinates. Oracle diagnostics are not used by
training, checkpoint selection or any reported achieved error/risk. A separate
cc_v6_oracle_audit.py reconstructs the correct candidates without editing frozen
training code: seed17 all8 checkpoints have zero change in the oracle2 minimum.
The synthetic boundary regression test demonstrates why the original helper
is not generally valid. Use corrected receipts for any future oracle claims.
After the222-test suite, phase2 bias tests and the oracle boundary test passed
separately; no scientific outcome depends on the old oracle fields.
