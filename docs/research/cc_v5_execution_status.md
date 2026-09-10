# V5 execution and phone-data status, 2026-09-11

The overall R&D goal is active. Revalidate handles before restarting any work.

- Training session74620 runs seeds17,29,43 serially. Seed17 has completed all
  six120-epoch arms; seed29 has started. Seed43 is queued within the same live
  PowerShell process. No duplicate training should be launched merely because
  a later seed directory does not yet exist.
- Independent scoring of all12 best/final seed17 prediction records passed;
  [interim report](../benchmarks/cc_v5/seed17_screen/report.md). Mean best
  reproduction: point2.5192°, posterior2.7126°, generic action2.4461°,
  transport random2.3649°, selected-action2.4607°, selected-action+gradient2.4676°.
  The special new training schemes lose to ordinary random-action supervision
  in this seed. This is reused119-image development validation, not a final
  three-seed effect, independent phone benchmark or novelty finding.
- V5 inference timing/export/calibrated reliability remains unmeasured.
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
- Optional standard Apache2.0 DINOv2-S teacher acquired and CPU load-smoke-checked;
  no teacher image features or training yet. It is22.06M parameters and intended
  only for a future training/reference experiment, not compact deployment.
  [Exact provenance](../ip/dinov2_teacher_adoption.md).

Next: complete all queued V5 arms, independently score both best/final outcomes
across all seeds, preserve negatives and decide training direction. Finish and
verify phone acquisition, inspect only the three loader scenes to implement
correct RAW/CFA/black-white and target extraction. Freeze a meaningful phone
evaluation protocol before scoring the reserved scenes. Continue independent
iPhone ground-truth search; no proprietary facial collection is required now.
The original FFCC-inspired numerical control remains untrained. Teacher/EMA and
equal-query nonadaptive controls remain required future candidates, not results.
