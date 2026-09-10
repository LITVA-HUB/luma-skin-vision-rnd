# Luma Selective Skin Color Engine — R&D

Standalone research infrastructure for estimating a defined facial-region color and deciding when the measurement is unreliable. **There is no validated real-world skin-color model yet.** All included training/evaluation examples are SYNTHETIC engineering checks. No participant photographs, pretrained weights, identity recognition, medical interpretation, ethnicity inference or undertone classifier are included.

The proposed target is continuous CIELAB under D65 / CIE 1931 2° conditions, linked to repeated instrument measurements at registered cheek sites. Ordinary smartphone JPEGs cannot identify physical skin reflectance under arbitrary unknown lighting/camera processing. This project tests a bounded operating domain and must be allowed to abstain.

The original Luma application is unavailable. `luma_style_a0` is a clean-room approximation from provided audit context, not a reproduction verified against its source. See [status](docs/research/CURRENT_RND_STATUS.md), [research plan](docs/research/implementation_plan.md), [prior art](docs/research/prior_art.md) and [what to collect](docs/data/DATA_REQUIRED.md).

## Run on Windows / RTX 4060

Python 3.12 is pinned in `.python-version`; package supports 3.11+. The default training wheels use the official CUDA 12.8 PyTorch index. An NVIDIA driver is required for GPU training; CPU execution is supported. No separate CUDA toolkit is needed for these wheel-based runs.

```powershell
uv sync --all-extras
uv run --all-extras pytest -q
uv run --all-extras ruff check .
uv run --all-extras python scripts/run_smoke.py
uv run --all-extras python scripts/render_smoke_report.py
```

`run_smoke.py` generates 60 toy subjects, validates all records, runs A0/A1/A2/C/C+/proposed, fits separate calibration, evaluates locked test data, exports the three learned models and measures CPU/GPU/ORT batch-1 execution. It preserves every subprocess command, exit code, stdout and stderr under `artifacts/smoke_*/commands.json`. This is not a claim of useful accuracy or a required size for a real dataset.

For minimal production-interface dependencies use `uv sync` (no PyTorch/OpenCV/ONNX); for training use `uv sync --extra train --extra dev`. Use `--all-extras` on later `uv run` commands to retain optional tools, or invoke `.venv/Scripts/python` directly. The training environment is separate from a future inference bundle.

## Individual commands

```powershell
uv run --all-extras python scripts/inspect_environment.py
uv run --all-extras python scripts/generate_synthetic_demo.py --subjects 60
uv run --all-extras python scripts/validate_dataset.py --manifest data/synthetic_demo/manifest.jsonl
uv run --all-extras python scripts/repeatability.py --manifest data/synthetic_demo/manifest.jsonl
uv run --all-extras python scripts/train.py --config configs/experiments/baseline_c.yaml
```

Training prints an experiment directory. Substitute that exact path for `RUN`:

```powershell
uv run --all-extras python scripts/calibrate.py --experiment RUN
uv run --all-extras python scripts/evaluate.py --experiment RUN
uv run --all-extras python scripts/export_onnx.py --experiment RUN
uv run --all-extras python scripts/benchmark.py --experiment RUN --device cuda
uv run --all-extras python scripts/benchmark_onnx.py --experiment RUN
```

All commands run from the repository root. `prepare_dataset.py --manifest captures.jsonl --output manifest.jsonl` imports schema-complete JSONL records without split labels and assigns deterministic subject-level splits. Output shares the input data root; existing splits/output are never overwritten. Keep real datasets outside Git, preferably a separate controlled directory.

## Implemented boundaries

| Component | Current state |
|---|---|
| A0 | Clean-room robust bilateral cheek measurement; no correction |
| A1 | Bounded linear-sRGB Gray World; Shades of Gray implementation available |
| A2 | Train-only regularized global affine color matrix; matched physical instrument target still required |
| C | MobileNetV3-small features, random initialization, continuous Lab regression |
| C+ | Same backbone plus weakly supervised spatial attention and generic quality features; provisional comparator, not a tuned strong baseline |
| Proposed | Same backbone/attention, 12 ambiguity features and OOF residual model; contribution unverified |
| Uncertainty | Subject-OOF residual regression and subject-max split calibration; no shift or conditional accepted-risk guarantee |
| Export | Tested FP32 ONNX numerical equivalence; no FP16/INT8 deployment accuracy claim |
| API | Versioned conservative local contract; all current artifacts return `UNSUPPORTED` |

Core code is in `src/luma_skin_vision`; configs hold reproducible settings; runs contain hashes/checkpoints/OOF audit/calibration/reports; `docs/data` describes real collection; `docs/ip` separates software, weights and data licenses; `docs/skolkovo` maps evidence gaps without promising application outcomes.

Model attention is not anatomical skin segmentation and not proven measurement reliability. Bbox-based cheeks require a sufficiently frontal, correctly registered photograph. Collection must explicitly resolve selfie mirroring. A small smoke model's parameter count, speed and synthetic errors do not establish real skin color performance or cosmetics shade-match quality.

The code is kept local. No public project license or IP clearance is asserted; review [licensing inventory](docs/ip/licensing_inventory.md) before redistribution.
