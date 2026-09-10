# V5 execution and phone-data status, 2026-09-11

The overall R&D goal is active. Revalidate handles before restarting any work.

- Training session74620 runs seeds17,29,43 serially. Seeds17 and29 completed all
  six120-epoch arms each; seed43 is now running within the same live
  PowerShell process. No duplicate training should be launched merely because
  a later seed directory does not yet exist.
- Independent scoring of all24 best/final prediction records for seeds17/29
  passed; [two-seed report](../benchmarks/cc_v5/two_seed_screen/report.md).
  All24 checkpoints also passed CPU replay against saved GPU predictions;
  largest absolute discrepancy across checked actions, risks and degree errors
  was3.745e-5 (predeclared tolerance1e-3).
  [Replay receipt](../benchmarks/cc_v5/two_seed_cpu_replay.json). Seed17 mean best
  reproduction: point2.5192°, posterior2.7126°, generic action2.4461°,
  transport random2.3649°, selected-action2.4607°, selected-action+gradient2.4676°.
  The special new training schemes lose to ordinary random-action supervision
  in this seed. This is reused119-image development validation, not a final
  three-seed effect, independent phone benchmark or novelty finding.
- Seed29 best means: point2.5158°, posterior2.4950°, action2.4039°,
  transport random2.5243°, policy2.4804°, gradient2.5212°. The most successful
  arm changes between seeds; no stable large mechanism improvement is shown.
- Seed43 starts at repository d738641 with dirty phone-development docs; its
  seven training-script hashes equal seed17 exactly. Source-snapshot differences
  are pyproject.toml/uv.lock adding h5py for independent CPU phone loading.
  Training torch/numpy/model code were not upgraded. Per-seed provenance is
  retained; do not describe every seed as launched from one clean Git checkout.
- V5 inference timing/export/calibrated reliability remains unmeasured.
- Equal-query search control completed on all10 trained-critic best checkpoints
  from seeds17/29. Every25-query first stage retained the original point on all
  119 images;51-query adaptive and fixed selections were bitwise identical.
  At103 queries, differences are small and mixed (largest absolute mean-error
  difference0.01061°). This does not support crediting feedback/recentering for
  the25-to51-query gain. [Control report](../benchmarks/cc_v5/two_seed_search_control/report.md).
  Session76511 finished successfully; no new training or phone test data used.
- Smartphone acquisition session92820 terminated with HTTP429 after partial
  acquisition. Resume session98768 now uses a tested transport wrapper with
  request spacing and bounded Retry-After retries; existing files are reverified
  and skipped. The original downloader and selection remain byte-identical.
  The frozen3.23GB Beyond RGB subset contains
  subset:3 original training scenes for loader development and44 paired-phone
  test scenes reserved for later use. Verify data/public/beyond_rgb_phone/
  verification_all.json and session completion before declaring acquisition done.
  Data/code/selection hashes and exact limits are in
  [the phone plan](../data/smartphone_benchmark_plan.md). Do not decode reserved
  test pixels/GT to tune the model. Both phones from a scene stay in one role.
- iPhone SE2/XS Max CC0 archive acquired,76.45MB.30 real object input photos
  are an auxiliary repeatability candidate, not verified absolute-color GT.
  No image decoding/training yet. Initial phone list also records restricted
  S24/LSMI/RenderedWB/Flash-Ambient sources; these were not adopted.
- Phone loader audit and preparation completed on all six TRAIN captures only.
  Files contain already demosaiced HWC camera RGB, sampled on a1/255 grid.
  Reference polygons visually align; one white patch is saturated and dark
  neutral patches disagree. [Findings](../data/phone_loader_findings.md).
  A fixed gray-reference quality rule is committed before any test decoding:
  [protocol](phone_reference_protocol_v1.md), SHA256
  c641f1ae17a86d6597b0f629879bd25765a8fff6e0559ef4f5ab8789b5116b19.
  Six references passed, and classical methods plus model thumbnails prepared.
  Their loader-only diagnostic numbers are not a held-out phone benchmark.
  Reserved44 test scenes remain untouched numerically. Any test preparation
  additionally needs completed acquisition verification and a model/weight lock.
- Optional standard Apache2.0 DINOv2-S teacher acquired and CPU load-smoke-checked;
  no teacher image features or training yet. It is22.06M parameters and intended
  only for a future training/reference experiment, not compact deployment.
  [Exact provenance](../ip/dinov2_teacher_adoption.md).

Next: complete all queued V5 arms, independently score both best/final outcomes
across all seeds, preserve negatives and decide training direction. Finish and
verify phone acquisition and freeze method/weight/calibration identities before
scoring reserved scenes under the now-fixed reference protocol. Continue independent
iPhone ground-truth search; no proprietary facial collection is required now.
The original FFCC-inspired numerical control remains untrained. Teacher/EMA and
an independent-domain search comparison remain future work. The equal-query
nonadaptive control is now measured on the two-seed source-development screen.
The focused2026 prior-art update confirms compact teacher distillation and
semantic weighting already exist; no claim that those ideas alone are novel.
