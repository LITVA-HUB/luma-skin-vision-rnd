# Experiments

Each run must be reproducible from frozen code, config, environment, dataset hash, and split hash. Store the method name (A0/A1/A2/C/C+/Proposed/ablation), seed, hardware, dependency versions, model parameters, training/calibration provenance, metrics, and artifact checksums. Keep generated outputs out of source control unless explicitly approved as non-sensitive evidence.

Real experiments follow `docs/research/experiment_plan.md`: repeatability first; subject-disjoint train/validation/calibration/test; error models trained on subject-held-out residuals; calibration thresholds frozen before test; Proposed compared with strong matched C+ at 80% coverage and across the complete risk–coverage curve; locked unseen device/light evaluation; paired subject-cluster bootstrap.

Synthetic runs are engineering smoke tests only. Label their directories, reports, and plots `SYNTHETIC`; they do not establish color accuracy, ROI reliability, calibration, generalization, novelty, or any passed research gate. Real efficacy remains **NOT MEASURED** until instrument-paired evaluation is complete.
