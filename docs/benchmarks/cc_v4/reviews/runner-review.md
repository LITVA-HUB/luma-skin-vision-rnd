# CC V4 Task 2 runner review

Review date: 2026-09-10. Scope: manual, read-only review of `docs/research/cc_v4_spec.md` Task 2, `docs/research/cc_v4_source_lock.md`, `scripts/cc_v4_experiment.py`, and `tests/test_cc_v4_experiment.py`, plus CPU integration against the newly available model/geometry modules. I did not run GPU work, load the source cache, or inspect any dataset numeric row. The only repository change from this review is this report.

## Verdict

**Do not treat the current runner as the final locked evidence producer until findings R1 and R2 are resolved.** The fitting/evaluation logic itself has no observed role leakage or GT-dependent action sampling, and its value targets and checkpoint decision keys match the frozen protocol. The outstanding problems concern required compute evidence and cryptographic correspondence of the multiple selected artifacts. R3 should also be resolved before issuing the first command so a run carrying the frozen schema cannot silently become the later derivative ablation.

Scoped CPU verification:

```text
.venv/Scripts/python.exe -m pytest \
  tests/test_cc_v4_experiment.py tests/test_cc_v4_geometry.py tests/test_cc_v4_model.py -q
37 passed in 4.70s
```

`git diff --check` also reported no whitespace errors in the reviewed runner, tests, specification, or source lock.

Reviewed-byte SHA-256 identities:

```text
cc_v4_spec.md             5bbce09aee93c0af60cb2ab1bab7ae145208434b61ff440b6931b242b05bcccf
cc_v4_source_lock.md      4aa00196eba451565272ea8e24a6e3e423cfcf38579815e6fe1bd3c7bb569b19
cc_v4_experiment.py       b9332140c0d6e0fc7f1e82f7ca3fdad4ca842c74028258c33d502213cf13e934
test_cc_v4_experiment.py  ee4c21256ce0dc5f59390d3273bdf8f1024c69aa1c53f5829d287cd822cbc1d8
cc_v4_model.py            a85ec84df9b0bbeeeef25cdff192a85e10a8ee35bced7a6949edfb4a5843cc148
cc_v4_geometry.py         5f0aceda35209b40d4f93be6825c329b0c274f83671aaaeb945ab6d1bc95408f
test_cc_v4_model.py       96cb7624f00570f7c087bf1c6fc38b1c9d42d77669032c5fc5d50d3150a85820
test_cc_v4_geometry.py    861c6f42b119a86d75bf29d505d51069afe1ba34ae310ad70398889dd87efc77
```

## Findings

### R1 — High: required actual inference latency is never measured

Task 2 requires actual latency alongside parameter count and allocator peak. The runner evaluates only `select(..., steps=4)` at `scripts/cc_v4_experiment.py:93-99`; it neither times this call nor performs separately timed steps 1, 2, and 4. The result records training wall time, parameters, allocator peak, and cache size at lines 249-255, but no inference latency. It also discards the model's returned `query_count`; the only compute record is the declarative string at line 158.

Consequently, an otherwise complete run cannot substantiate the locked 25/51/103-query compute comparison or the specification's actual-latency claim. A later ad hoc timing run would need its own exact checkpoint, device, warmup/repetition method, synchronization, batch size, and code hashes to be scientifically comparable. Record observed query counts and synchronized warmup/timed latency for steps 1, 2, and 4 in the immutable run artifacts.

### R2 — High: the independently selected point checkpoint is not cryptographically bound to its evidence

The runner correctly saves two independently selected checkpoints and their validation artifacts at lines 226-235: `best.pt` for searched step 2 and `best_point.pt` for the direct point. It also saves `last.pt` at line 246. However, `result.json` records bytes/hash only for `best.pt` at line 255. There is no hash for `best_point.pt`, `last.pt`, either metrics JSON, either validation NPZ, the history, copied lock/spec, or config.

This leaves the locked independent direct-point result without a machine-checkable checkpoint-to-metrics association. It also prevents a report consumer from distinguishing later file replacement from the original completed run. At minimum, the completion manifest should hash all three checkpoints and bind each selected checkpoint to its epoch, metrics JSON, and prediction NPZ. Hashing config/history and the copied lock/spec would make the complete evidence bundle self-verifying.

### R3 — Medium: one schema accepts protocol-divergent budgets and derivative training

The frozen source screen is seed 17, 120 epochs, batch 32, and value-only training. The CLI defaults match those values at lines 265-268, but `train()` accepts any positive epoch count, any batch at least 2, any seed, and any nonnegative `gradient_weight` at lines 126-127. A positive derivative weight changes training at lines 203-213 while the run still declares schema `cc-v4-source-screen-1` and copies the value-only source lock. `LOSS_WEIGHTS` and the `loss_weights` config field also omit the applied derivative coefficient; it survives only inside raw CLI arguments.

Arguments make such a deviation discoverable, so this is not hidden numeric leakage. It is still easy to launch an artifact whose embedded frozen lock contradicts its actual training. Reject nonzero derivative weight and nonlocked seed/epoch/batch for this schema, or issue a distinct ablation schema/lock and include the effective derivative coefficient in the normalized loss configuration.

### R4 — Medium: runner tests do not exercise the scientific boundary or artifact policy

The four tests in `tests/test_cc_v4_experiment.py` cover detached action sampling, refinement count summaries, fixed-coverage arithmetic, and warmup (`lines 23-62`). They do not execute or isolate:

- source role selection and numeric materialization boundaries;
- frozen data-hash rejection;
- immutable-output and failure-artifact behavior;
- epoch-0 eligibility and the 99% validity checkpoint gate;
- separate searched/direct checkpoint selection and correspondence;
- exact action value targets as wired into the training loop;
- oracle reconstruction for the actual 25/51/103 candidate sets;
- runtime query-count capture, artifact hashes, or final claim fields.

The model/geometry tests independently establish much of the mathematical core, and manual inspection found the current runner wiring correct. These absent runner regressions nevertheless leave the highest-cost protocol behavior unprotected immediately before long GPU runs. Add focused CPU tests with stubbed model/data helpers and tiny synthetic tensors; no real cache decode is needed.

### R5 — Low: oracle evidence is incomplete for the declared stage-1/2/4 comparison

`oracle_errors()` returns only the first 51-candidate minimum and the full 103-candidate minimum (`lines 73-86`), and metrics expose only `oracle2_mean`/`oracle4_mean` (`lines 107-114`). The selected-action results cover stages 1, 2, and 4, but there is no 25-candidate stage-1 oracle or explicit selected-minus-oracle regret summary. The saved arrays preserve enough information to derive step-2/4 regret, but not stage-1 oracle without reconstructing candidates again. Record oracle 1/2/4 and regret at each stage in the primary metrics.

### R6 — Low: the module docstring overstates the compressed-row boundary

The first line says the runner “never reads test/camera GT rows.” `read_npz_rows()` does avoid deserializing excluded rows into numeric arrays, and `source_rows()` supplies only train/validation indices. Because the NPZ member is a compressed sequential stream, the helper advances by reading and discarding intervening decompressed bytes. Thus the defensible claim is that excluded rows are never **numerically decoded, retained, transferred, or used**, not literally never read. The source lock already uses the better word “decoding.”

## Confirmed-correct behavior

- **Role isolation:** `source_rows()` selects only manifest `train` and `val`, enforces disjoint capture groups, and returns 1,126/119 relative indices. The two `read_npz_rows` calls at runner lines 172-173 receive only their sorted union. No source test/risk/cal row or INTEL file is indexed into an array, sent to the model, or evaluated. The manifest is read for role/group/ID integrity and output provenance only.
- **Frozen source identity:** lines 131-137 require the exact inherited V3 `cube.npz` and manifest hashes plus the expected manifest/role counts before output creation or numeric materialization.
- **GT-independent action design:** `sample_actions()` at lines 26-31 uses a detached model point, independent uniform local/global draws, and no GT argument. GT enters only after encoding, to compute supervised point and field targets.
- **Value-target correctness:** lines 193-213 convert the real illuminant to the declared log(R/G), log(B/G) coordinates. `analytic_costs(GT, action)` supplies reproduction degrees and exact sine-squared reproduction cost. The point term is actual reproduction degrees; field losses are squared errors in the locked units and weights. The independent NumPy geometry tests cover exact null, joint action/illuminant translation, extreme GT scale, and analytic gradients.
- **Warmup:** epochs 1-20 are point-only; field weight rises linearly from epoch 21 to 40 and remains one afterward. Epoch 0 is evaluation-only and records `None` for unmeasured training losses.
- **Checkpoint policy:** lines 222-235 select `best.pt` by minimum step-2 validation mean only when valid fraction is at least 0.99, with strict `<` retaining the earliest exact tie. Epoch 0 participates. `best_point.pt` is selected independently by direct-point validation mean under the same validity gate.
- **Repeated refinement diagnostics:** evaluation performs one cached encode followed by a four-stage selection, retains per-row actions, predicted risks, reproduction and recovery errors, and reports 1-to-2, 2-to-4, and 1-to-4 true-error changes. Model tests confirm 25/51/103 queries, preceding-center retention, original-point retention, deterministic selection, and nonincreasing predicted risk. The runner correctly warns through its fields that true error may worsen.
- **Matched mode execution:** the same runner, loss, optimizer, augmentations, action sampler, evaluation, and checkpoint rules apply to posterior and transport. Model tests confirm equal parameter graphs and the declared 128 inactive first-layer scalars in posterior/direct. This supports a matched declared budget while making no identical-FLOPs claim.
- **Claims:** output limitations explicitly call the validation reused development evidence, one seed, uncalibrated, and neither independent camera/test nor skin evidence. Risk summaries repeat the uncalibrated status. No Delta E, surface-color, universality, or novelty claim is emitted.
- **Output protection:** a pre-existing output path is rejected. For a newly created run directory, ordinary caught failures after creation write one `failure.json` and never overwrite an existing completed or failed artifact. Abrupt process termination and validation failures before directory creation naturally cannot emit a failure record.

## Release decision

The runner's data and target paths are scientifically suitable for the declared source-only screen. Resolve R1 and R2 before first full GPU training so the costly artifacts contain the required compute evidence and self-verifying checkpoint correspondence. Resolve R3 in the launch interface or use an exact reviewed command whose arguments are preserved. R4 and R5 are lower-cost protections that should be addressed before relying on the screen report, even if training itself begins after R1-R3 are fixed.

## Addendum — 2026-09-11 amended runner and benchmark

This addendum reviews the amended runner and new `scripts/cc_v4_benchmark.py`. I did not interrupt or inspect the numeric contents of the active source runs, run GPU work, or decode dataset rows. The conclusions below supersede R1-R3 for the amended bytes; R4-R6 remain applicable except where explicitly updated.

Reviewed-byte SHA-256 identities:

```text
cc_v4_experiment.py       d0c7aaf08fedc98bb2db37598eff730713dcd3a8c8e3e4f41fc1a00368de6eac
cc_v4_benchmark.py        83234833af884bb67ff8871f6d64e5215cbc100b3002a5dad521fc970b99bbb7
cc_v4_model.py            1c7c4b54c3b07bfbaf9e26a689e08ca0e511532b62403421664ed7c18054ae68
test_cc_v4_experiment.py  ee4c21256ce0dc5f59390d3273bdf8f1024c69aa1c53f5829d287cd822cbc1d8
cc_v4_spec.md             2ae0c34bfd6d2c4898ea1e916510912ce0dbc96b2ddc8563cbd62841b17ae53a
cc_v4_source_lock.md      b9d0175675dff3f893858318a535d5e70f3ccf9ddc1dd6c6799c92fd19aa6bf2
cc_v4_pilot_notes.md      4d05ccee6ce9458baf22f0fbab490f87a228162afc35acde47490be6b254b38a
```

Verification:

```text
.venv/Scripts/python.exe -m pytest \
  tests/test_cc_v4_experiment.py tests/test_cc_v4_geometry.py tests/test_cc_v4_model.py -q
47 passed in 5.68s

.venv/Scripts/python.exe -m py_compile \
  scripts/cc_v4_experiment.py scripts/cc_v4_benchmark.py
passed

completed pilot artifact manifest: 48 entries, 0 hash mismatches
```

### Prior findings resolved

- **R1 compute evidence — addressed for the intended workflow.** Training evaluation now observes every batch's four-stage `query_count` and rejects anything other than `{103}` at runner lines 93-116. The separate benchmark measures stages 1, 2, and 4 with 20 warmups each, at least 20 measured repetitions, synchronization before and after every sample, device-resident batch-one input, explicit scope/units, raw samples, median/p95/min, and observed 25/51/103 counts (`cc_v4_benchmark.py:19-96`). This is an appropriate fresh-process latency design. Definitive latency remains unmeasured until the declared uncontended post-training runs finish; no pilot number may fill that field.
- **R2 artifact binding — resolved.** After writing the completed result, the runner hashes every existing file recursively before creating `artifact_manifest.json` (`cc_v4_experiment.py:252-266`). This covers best searched, independently selected point, last checkpoint, both metrics/prediction pairs, history, config, copied locks, source tree, dependency locks, and copied scripts. An independent check of the completed pilot found all 48 entries present with zero mismatches. The manifest intentionally cannot include itself; it is an integrity receipt against accidental mutation, not an external signature.
- **R3 protocol variants — addressed by explicit classification.** Config now sets `is_frozen_primary=true` only for `(epochs,batch,seed,gradient_weight)=(120,32,17,0)` at line 155, while retaining exact arguments. All permitted modes are members of the amended three-way screen. Nonlocked experiments remain possible but are explicitly classified, which is sufficient if every report filters on this field. The completed full posterior config inspected at the metadata level records mode posterior, CUDA, 120/32/17/0, and `is_frozen_primary=true`.

### A1 — Medium: benchmark receipt omits the run identity and primary-status fields needed for three-way attribution

The benchmark verifies every manifest entry, the current model hash, data hashes, checkpoint, inputs, observed query counts, device, precision and timing scope. Its output at `cc_v4_benchmark.py:54-64`, however, does not record:

- `config["arguments"]["mode"]` or the source run identity;
- `config["is_frozen_primary"]`;
- source config/artifact-manifest hashes;
- confirmation that `result.json` says `status=complete`.

Therefore a standalone JSON cannot say whether it measures posterior, generic action, or transport, and it cannot mechanically distinguish a frozen full run from the pilot/nonlocked run. The checkpoint hash is unique but opaque, and filenames are not internal provenance. Before definitive timing, validate the expected schema, completed status, and frozen-primary flag, then write mode, run/config/manifest identity, and status into the benchmark receipt. This is benchmark-only and does not invalidate or require interrupting active training.

### A2 — Medium: one executed benchmark dependency is not checked against the trained snapshot

The script checks the current `cc_v4_model.py` against the trained hash before loading the checkpoint (`cc_v4_benchmark.py:33-34`). It also imports and executes `read_npz_rows` from the current `cc_v2_statistics.py` but never checks that file against `config["scripts_sha256"]["cc_v2_statistics.py"]`. The benchmark script's own hash records the import statement, not the imported helper bytes. A helper change could alter which bytes are decoded or how the four inputs are constructed while leaving the output's model/checkpoint provenance apparently intact.

Check the current helper hash against the trained snapshot or import the snapshotted helper. Record that hash in the benchmark output. The current workspace helper does match the completed run snapshot; this is a future-mutation guard, not evidence of present contamination.

### A3 — Medium: the benchmark has execution evidence but no focused protocol test

There is no `test_cc_v4_benchmark.py`. The contended pilot proves the current happy path executes on this GPU and observes 25/51/103 queries, but it does not regression-test rejection of a modified manifest/model/data hash, existing output, too few repetitions, wrong query count, non-primary/incomplete run, or malformed validation IDs. Refactor the metadata validation and result construction into CPU-testable helpers or test them with monkeypatched CUDA/model calls before the definitive receipts are treated as durable evidence.

### A4 — Low: direct point-readout latency is not measured

The benchmark measures searched stages 1/2/4 for the trained mode. It does not measure the same checkpoint in direct readout mode (`query_count=1`), even though source results compare the original point against searched output. Historical C+ timings use different code and cannot isolate V4 search overhead. Add one direct-readout timing if the final report compares latency per decision or attributes overhead to search. This is not needed merely to report absolute searched-mode latency.

### Contended pilot disposition

`docs/benchmarks/cc_v4_pilot_notes.md` is explicit: the 20-repeat pilot benchmark overlapped the first full posterior training, every latency number is **CONTENDED/INVALID**, and all are excluded from reported model timing. It preserves the file only as an execution/query-accounting receipt and states that definitive 100-repeat benchmarks run sequentially after training. This is scientifically adequate provided the final report never copies the pilot timing values. The pilot JSON itself has no invalid-status field, reinforcing A1's need for complete status/provenance in definitive receipts.

### Addendum release decision

No load-bearing numerical, label-leakage, checkpoint-selection, or training-immutability defect was found in the amended runner, so the active sequential source runs do not require revision. The prior R1-R3 blockers are substantively addressed. Resolve A1 and A2 before the definitive benchmark, and preferably A3; these changes affect only future timing receipts. Preserve the contended pilot and its exclusion note exactly as an erratum.

## Closure addendum — 2026-09-11 benchmark provenance fix

Scoped byte reviewed:

```text
scripts/cc_v4_benchmark.py
sha256 9f338d9dcd901a3d3d2c73cc86dfba142054ae71f4c9eca6df49d01ef2065a75
```

**A1 and A2 are closed.** Before loading the checkpoint or moving the model/input to CUDA, the benchmark now requires `config.is_frozen_primary=true` and `result.status=complete` (`lines 28-31`). It then verifies every manifest-bound run artifact, the live model implementation, and the live `cc_v2_statistics.py` row decoder against their trained snapshot hashes (`lines 32-40`). The output records mode, absolute source run, primary flag, training status, config hash, run-manifest hash, decoder hash, model hash, checkpoint hash, and benchmark-script hash (`lines 61-66`). A definitive receipt can therefore identify and mechanically distinguish posterior/action/transport, reject the non-primary pilot, and bind its executable dependencies to the completed training evidence.

The source-validity metadata receipt `docs/benchmarks/cc_v4/source_validity.json` (SHA-256 `06698e4f628298985969a1ebd0ceaec399f87532d782ee959a75e8acfee6efec`) records 1,126 train and 119 validation image rows, with zero invalid rows in either role and no GT key read. This resolves the practical concern that invalid/absent-channel inputs might silently dilute training under the current frozen source population. It is a source-input validity audit, not accuracy evidence.

Verification on the final scoped bytes:

```text
.venv/Scripts/python.exe -m py_compile scripts/cc_v4_benchmark.py
passed

.venv/Scripts/python.exe -m pytest \
  tests/test_cc_v4_experiment.py tests/test_cc_v4_geometry.py tests/test_cc_v4_model.py -q
47 passed in 5.52s

git diff --check -- scripts/cc_v4_benchmark.py \
  .superpowers/sdd/cc_v4_spec/runner-review.md
passed
```

**Final scoped verdict: clean for the planned definitive post-training benchmark.** No GPU run was performed in this review. A3 remains a defense-in-depth test-coverage recommendation and A4 remains optional unless the report attributes latency overhead specifically to search; neither blocks the declared stage-1/2/4 measurements. The contended pilot remains excluded exactly as recorded.
