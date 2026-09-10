# V5 completed; phone benchmark next, 2026-09-11

Overall R&D goal remains active. No GPU training is currently running.

## Completed V5 work

Training session74620 exited successfully: seeds17/29/43, six arms each,
120 epochs per arm. All18 primary arms completed. All36 best/final prediction
records passed independent GT scoring and CPU checkpoint replay. Sessions14721
(two-seed replay),76511 (two-seed search control) and19345 (full pytest) ended
successfully. The final source screen is
[cc_v5_report.md](../benchmarks/cc_v5_report.md). Full pytest:206 passed,
14 historical ONNX warnings,32.66s. Do not restart these completed jobs.

Mean best reproduction over three seeds: point2.4814°, posterior2.5612°,
generic action2.4136°, transport random2.4548°, transport selected2.5346°,
transport derivatives2.4812°. Mean raw risk80: posterior2.1307°, action2.1333°,
transport random2.0192°, selected2.1895°, derivatives2.1448°. This is reused119
validation images, not independent accuracy or calibrated C+ superiority.
Selected-action/derivative training failed to give a consistent advantage.
Physical transport retains a modest selective-ranking signal with mixed seeds.

Equal-query refinement control completed for all15 trained-critic best models.
At stage1 only one of1785 model/image cases moved from its original point;
51-query adaptive/fixed actions agree exactly except one model's FP32 difference
up to1.193e-7. No useful feedback-specific gain is established. More passes
are not the next priority. All raw outcomes and negative evidence are preserved.

Training scripts have identical hashes across all seeds. Seed43 records repo
d738641 with dirty phone-development work; source-snapshot differences from
seed17 are pyproject.toml/uv.lock adding h5py. No torch/numpy/model-code upgrade
occurred. Do not describe every seed as starting from one clean Git checkout.
V5 deployment timing, inference VRAM, ONNX and calibrated thresholds remain
unmeasured. V2 remains the strongest completed positive transfer milestone.

## Phone data ready; test labels still reserved

Acquisition session98768 exited successfully. Earlier92820 failed withHTTP429;
polite bounded retries resumed verified files. Complete frozen subset:47 scenes,
560 records,3,229,998,120 payload bytes. Original-author CC BY4.0. Archive
[verification receipt](../data/provenance/mobile_screen_2026_09_11/verification_all.json)
SHA256a346e735673a43320c6e2bffd5b93268544f77dae25e90f51963920ebbc078ac.
The downloader's NOT DECODED label describes its byte-only action; three loader
TRAIN scenes have separately been numerically inspected. All44 paired-phone test
scenes (88 NT inputs) remain untouched numerically. No final phone metric exists.

Loader preparation session14686 completed for six TRAIN captures. Released HDF5
is demosaiced HWC camera RGB; sampled values lie on1/255 grid. Reference polygons
align; a white patch saturated and dark patches disagree. Reference protocol
v1 was frozen before any reserved decoding, SHA256
c641f1ae17a86d6597b0f629879bd25765a8fff6e0559ef4f5ab8789b5116b19.
[Findings](../data/phone_loader_findings.md),
[quality protocol](phone_reference_protocol_v1.md). All six loader references
passed; classical diagnostics/model thumbnails prepared. These are not held-out
phone results. Never change reference selection based on model performance.

CC0 iPhone SE2/XS Max archive76.45MB is acquired;30 real object inputs are only
an auxiliary repeatability candidate, without verified absolute-color GT.
No iPhone image decoding/training yet. Restricted S24/LSMI/RenderedWB data was
not adopted. Optional Apache2.0 DINOv2-S teacher was acquired and load-checked,
but no image features or teacher-supervised training were run.

## Next actions

Freeze a phone evaluation method/weight/calibration lock, then prepare/scoring
code using the existing reference protocol and byte-verified reserved records.
Compare frozen V2 strongest matched direct C+, SoG residual, strong cheap/classical
controls; any V5 exploratory matrix must be predeclared and retained in full.
Do not alter old V2 manifests or overwrite their external predictions; use new
outputs and verify exact numerical source dependencies against their snapshots.
Report per-camera and scene-clustered outcomes, scorable counts, fixed-coverage
risk and actual source-threshold coverage. Reference-quality exclusions and
model refusals have different denominators. No per-camera adaptation or CCM
is allowed in the camera-independent input path. No synthetic/facial DeltaE.

Then test stronger scene-context training, retaining ordinary/no-teacher controls
and a small local inference model. The FFCC-inspired numerical control remains
untrained; the newINTEL384 group/LOCO protocol is still pending. Distillation,
semantic weighting and recurrence already have prior art. The project goal is
not complete and is not blocked.
