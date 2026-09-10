# Luma R&D implementation plan — 2026-09-10

Goal: deliver the entire local research machine; real scientific efficacy remains blocked on measured data.
Architecture: independent Python package with schema-validated manifests and split enforcement, deterministic measurement, interchangeable compact regression and residual prediction, separately fitted calibration and versioned conservative inference contract.
Spec: docs/context/user_request.txt (the user's instructions); other imported files are PROVIDED CONTEXT only.
Stack: Python 3.11+, NumPy/Pillow/Pydantic, optional OpenCV; PyTorch/torchvision for training, optional ONNX/ORT. No downloaded pretrained weights by default.

## Decisions and repository map

Prefer modular local package over notebook-only prototype (difficult to validate) or service platform (premature). New nested Git repository isolates unrelated existing workspace content. Use a codex/bootstrap branch. No publication or upstream integration.

- src/luma_skin_vision/color.py: explicit sRGB transfer, D65/2° Lab and CIEDE2000.
- data.py + data/schemas/: validated region records, repeat metadata, rooted file integrity, reproducible subject splits and leakage checks.
- preprocessing/, face/, roi/, photometry/: image normalization, optional YuNet adapter, cheek geometry, bounded hypotheses and deterministic A0/A1/A2.
- evaluation/: equal coverage risk, subject bootstrap and subgroup reports.
- models/, uncertainty/, calibration/, selective/: compact C, matched C+ infrastructure, ambiguity features, held-out residuals and conservative decision.
- training.py + scripts/: immutable run directories, seeded train/evaluate/calibrate/benchmark/export commands.
- docs/: research, protocols, licensing and technical evidence; missing results say NOT MEASURED.

## Execution tasks

- [x] 1. Bootstrap: inspect environment and supplied archive without running it; verify archive manifest, save source context and hash. Initialize Git, dependency groups and lockfile.
- [x] 2. Independent research: verify at least 10 strong analogues from primary sources, separately assess code/weights/data; narrow novelty. Review current compact backbones and official Skolkovo sources.
- [x] 3. Data and mathematics: write tests first for Sharma reference pairs, sRGB landmarks/roundtrip, schema and cross-split rejection, repeatability, deterministic subject allocation. Implement only after failure. Gate: targeted pytest passes.
- [x] 4. Measurement and evaluation: test ROI bounds, color transforms, matched coverage, tied-score policy and paired subject bootstrap. Implement A0/A1 and train-only ridge CCM A2; synthetic generator uses known reference colors and camera/light perturbations. Gate: generated manifest validates and baselines produce labeled reports.
- [x] 5. Compact model framework: test one end-to-end synthetic run, save/load predictions, split-safe error fitting and calibration persistence. Implement C with torchvision MobileNetV3-small (random initialization smoke only); use the same backbone for C+/proposed experiments with documented provisional reliability supervision. Record full config, source/data/split hashes and hardware.
- [x] 6. Integration/export: test reject semantics and uncalibrated deployment refusal; implement local request/response contract, export skeleton and measured CPU/GPU/ORT smoke tooling. Export is engineering validation; optimization for deployment waits for positive real evidence.
- [x] 7. Evidence and review: run all tests, Ruff and full synthetic smoke; inspect independent review, fix findings and repeat affected checks. Save actual commands/results and update all status/evidence documents. Commit local repository; no remote publishing.

## Scientific gates and scope

G1 repeatability, G2 real problem, G3 ML versus classical, G4 proposed versus C+, G5 useful equal-coverage selection and G6 compact deployment all require real held-out data. Not passed by synthetic tests. A2 requires paired calibration samples; learned measurement-aware ROI needs reliable labels or residual supervision. Generic skin masks do not supply these labels. No classification-temperature score will be presented as regression uncertainty. Synthetic C+/proposed are infrastructure only, not strong scientifically tuned baselines. Pilot next: 10–15 people × 3 camera pipelines × 4 lighting conditions × 2 repeats, with repeated instrument readings at both cheeks.
