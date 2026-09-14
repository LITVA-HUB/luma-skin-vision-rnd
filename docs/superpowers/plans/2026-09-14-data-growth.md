# Luma dataset growth and quality experiments

Progress update, 2026-09-14 05:01 Moscow: data intake, AS verification and Seg1 training/evaluation are COMPLETE. All launched jobs exited0. Seg1 heldmeanIoU94.20% on2000images;4.42Mparameters,17.68MB,21m15s training. Additional16TRAIN hyperspectral faces acquired;19total. See docs/benchmarks/facial_skin_v1/report.md and docs/data/skin_data_growth_2026-09-14.md. Overall native-phone color accuracy remains unvalidated; goal active. The original checklist below is retained as historical intake context; completed follow-through is recorded in2026-09-14-data-growth-followthrough.md.

User explicitly authorizes external data acquisition, dataset assembly, testing and quality improvement. Five supplied photographs are additionally authorized for local analysis. User confirms they have no measured color or same-light color-card reference. This supersedes earlier series restrictions on acquiring new data, while all old sealed sources, original split assignments and exposed test exclusions remain intact.

**Goal:** acquire useful permitted training inputs, assemble traceable data, test improvements without conflating skin segmentation with instrument-referenced skin-color accuracy.

**Architecture:** separate source registry and immutable local originals on `D:/Luma-RnD/data_growth_2026_09_14`; bounded Python acquisition/validation; independent component experiments. No participant photographs in Git or uploads to external services. Existing research environment and AS sources remain frozen. Execute inline in the current isolated worktree.

**Scope:** first acquire UCI Skin Segmentation and the user photographs, investigate additional measured-color sources, finish AS verification, then continue the prepared head-range study. CHROMA-FIT/DAST require verified author access/license; ENCoDE requires credentialing; MST-E forbids ML training. No emails or agreements are sent/accepted by this task.

- [x] Verify AS primary terminal exit 0; launch independent CPU audit after it completes.
- [ ] Finish AS audit, then sequential selected-bank reconstruction/runtime and final report; preserve any failure and never adjust tolerances to pass.
- [ ] Add `scripts/skin_data_growth.py` and `tests/test_skin_data_growth.py`. Test invalid BGR/labels, deterministic duplicate-group partitions, malformed JPEG rejection, byte-preserving private ingestion, deduplication and missing-ground-truth semantics before implementation. Run the test file with the existing explicit Python interpreter.
- [ ] Acquire UCI through its original download link after saving original CC BY 4.0 evidence. Store URL, byte count, SHA256 and original citation. Validate 245057 rows, 50859 skin and 194198 non-skin, finite integer BGR in 0..255; report exact duplicate colors and conflicting labels.
- [ ] Freeze group partitions before training: SHA256 of each exact BGR triplet with fixed salt, 70/15/15 by hash bucket; all duplicates including conflicting labels remain together. Pixel-only evaluation lacks person/source identities and must not be called independent face generalization.
- [ ] Copy the five supplied files to private hash-named originals, preserving bytes; record decoded dimensions, checksum, source, missing color reference and diagnostic-only role. Treat the entire supplied batch as one conservative leakage group because identities are not provided. Never infer identity grouping from appearance.
- [ ] Add a bounded classifier experiment: compare fitted RGB logistic baseline to a fixed small nonlinear classifier grid; select solely on validation balanced accuracy and freeze the selected recipe before test access. Save class-wise recall/precision, balanced accuracy, AUC and selected predictions. This measures skin-pixel classification only, not Lab accuracy.
- [ ] Inspect source terms for NIST reflectance and additional face sources; acquire permitted useful originals, retain measured/derived/subjective/unlabeled label types separately. Reuse prior He/ISSA/UMINHO manifests rather than recounting old files as new.
- [ ] Run diagnostic inference on supplied photos without assigning pseudo ground-truth Lab; report inputs/limitations and maintain photos outside Git. Measured physical truth is absent.
- [ ] Record receipts, source/code hashes, tested results, pending work and usable paths. Broader goal remains active until useful ordinary-phone color quality is demonstrated.

Validation rules: selection uses train/validation only; test labels never tune thresholds. Duplicated samples cannot cross splits. Missing references remain null. Published dataset counts and locally validated counts are distinct. Existing AS mixed/camera roles are historically reused and confounded; new color labels must retain their illuminant/observer conventions.

No new GPU training runs concurrently with AS audit/runtime. CPU acquisition/profile work can proceed during verification. Storage has approximately 353 GB free on D at intake; C has about 38 GB, so new arrays go to D.
