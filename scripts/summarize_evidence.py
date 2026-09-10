"""Build status documents only from recorded local verification and smoke artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    latest = json.loads((ROOT / "artifacts/latest_smoke.json").read_text())
    summary_path = Path(latest["summary"])
    summary = json.loads(summary_path.read_text())
    verification = json.loads((ROOT / "docs/research/verification.json").read_text())
    assert verification["all_passed"]
    results, deploy = summary["results"], summary["deployment"]
    run_metadata = {
        name: json.loads((ROOT / path / "run.json").read_text())
        for name, path in summary["runs"].items()
    }
    assert len({m["source_hash"] for m in run_metadata.values()}) == 1
    assert len({m["dataset_hash"] for m in run_metadata.values()}) == 1
    assert all(m["data_kind"] == "SYNTHETIC" for m in run_metadata.values())
    dataset = summary["validation"]

    def save(path, text):
        (ROOT / path).write_text(text, encoding="utf-8")

    table = "| Method | Mean ΔE00 | Requested 80% risk | Actual coverage | Accepted images |\n|---|---:|---:|---:|---:|\n"
    for name, result in results.items():
        risk = result["risk_coverage"][1]
        table += f"| {name} | {result['metrics']['mean_delta_e00']:.4f} | {risk['mean_delta_e00']:.4f} | {risk['coverage']:.4%} | {risk['accepted']} |\n"
    intro = "# Synthetic engineering baseline report — 2026-09-10\n\n**SYNTHETIC ONLY. Not real skin accuracy, technological advantage or a passed scientific gate.**\n\n"
    intro += f"Measured data volume: {dataset['subjects']} toy subjects, {dataset['images']} JPEG patch images, {dataset['records']} region records; subject split 24/12/12/12. Two synthetic camera-gain settings and two synthetic light-gain settings are not real smartphone/light pipelines.\n\n"
    intro += "All six methods use identical subject splits and complete paired cheek records. Learned models ran two epochs at 64px from random initialization. This is intentionally a pipeline check, not a tuned model comparison. Requested 80% is rounded down to a common whole-image count; actual coverage is shown. Errors average both cheeks; selectors retain entire photos.\n\n"
    save(
        "docs/benchmarks/baseline_report.md",
        intro
        + table
        + "\nA2 is numerically strongest on this toy generator. The proposed neural prototype does not beat the strongest classical baseline here. This says nothing conclusive about real-data potential, and no architecture/config was tuned to make the comparison look successful. Gray World is worse than no correction on skin-only toy patches, consistent with its violated scene-average assumption.\n\nRaw provenance: [synthetic_smoke_evidence.json](synthetic_smoke_evidence.json). Full curve: [plot](synthetic_risk_coverage.png). Each experiment keeps its OOF membership audit, model hash, frozen calibration and test predictions.\n",
    )
    save(
        "docs/benchmarks/benchmark_results.md",
        "# Real-world benchmark status\n\n**NOT MEASURED.** Real participants N, camera pipelines M, lighting conditions K, ΔE00 X/Y, relative improvement Z and subject confidence interval are all NOT MEASURED. No real-data scientific gate has passed.\n\nLocal synthetic pipeline results are separately recorded in [baseline_report.md](baseline_report.md). The initial pilot must first establish instrument repeatability and genuine camera/light variation. Only then run locked A0/A1/A2/C/strong-C+/proposed comparisons and real image-level risk/coverage curves.\n",
    )
    dtable = "| Model | Parameters | ONNX bytes | CPU PyTorch median ms | RTX 4060 median ms | CPU ORT median ms | Export max ΔE00 discrepancy |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, value in deploy.items():
        dtable += f"| {name} | {value['cpu']['parameters']} | {value['export']['onnx_bytes']} | {value['cpu']['median_ms']:.4f} | {value['cuda']['median_ms']:.4f} | {value['onnx_cpu']['median_ms']:.4f} | {value['export']['max_delta_e00_difference']:.8g} |\n"
    train_table = "| Model | Complete training + OOF seconds | Peak CUDA allocated bytes | Training device |\n|---|---:|---:|---|\n"
    for name in deploy:
        m = run_metadata[name]
        train_table += f"| {name} | {m['training_duration_seconds']:.3f} | {m['peak_vram_bytes']} | {m['training_device']} |\n"
    save(
        "docs/benchmarks/deployment_benchmark.md",
        "# Measured deployment engineering smoke — 2026-09-10\n\n**SYNTHETIC, FP32, 64×64 crops, batch 1. Model-only timing: excludes decoding, face detection, ROI selection, ambiguity computation, residual model and policy. These are not production/phone latency claims.** Ten warmup calls and 50 timed iterations per backend, four CPU threads, GPU synchronized around timing. The machine was an ordinary active workstation, not a controlled performance lab.\n\n"
        + dtable
        + "\n"
        + train_table
        + "\nExport equivalence uses all frozen synthetic test-region inputs; the ONNX artifact is bound to the originating checkpoint and dataset by SHA256. GPU VRAM denotes peak allocated tensors across training/OOF, not all GPU memory in use.\n\nFP16 production, INT8, TensorRT, teacher/student: NOT STARTED pending useful real-data evidence. Parameter count and file size are measured artifact properties, not evidence of novelty. Raw p95 timings/environment/hashes: [synthetic_smoke_evidence.json](synthetic_smoke_evidence.json).\n",
    )
    save(
        "docs/benchmarks/ablation_report.md",
        "# Ablation status\n\nControlled real-data ablation results: **NOT MEASURED**. The C/C+/proposed smoke exercises different interfaces but is not a tuned causal ablation. In particular C+ changes attention and quality inputs together; it cannot isolate ROI value.\n\nAfter real gates: hold data/seed/backbone/ROI/preprocessing/tuning budget fixed; remove measurement-aware ROI, hypotheses, ambiguity features, error head and replace with standard residual confidence separately. Include no correction, classical correction only, no teacher, then FP32/FP16 and optional INT8 after a useful model exists. Report identical-image coverage, ΔE00/tail error/subject-paired CI, parameters and measured resource costs. Reject any novelty claim if advantage over strong C+ disappears.\n",
    )
    registry = "# Experiment registry\n\nEach `experiments/runs/<id>/run.json` is a transparent registry entry. Failed runs retain FAILED with reason; successful runs have unique directories. `python scripts/list_experiments.py` lists all historical runs. This table identifies the canonical final smoke; earlier smoke records are retained locally as superseded engineering runs.\n\n| Method | Run | Git commit | Source SHA256 | Status |\n|---|---|---|---|---|\n"
    for name, m in run_metadata.items():
        registry += f"| {name} | [{m['experiment_id']}](../../{summary['runs'][name]}/run.json) | {m['git_commit']} | {m['source_hash']} | {m['status']} / SYNTHETIC |\n"
    registry += f"\nCommand transcript: `{summary['command_log']}`. Configs, optimizer settings, precision, seed, resolution, augmentation, dataset/split hashes, pretrained provenance, environment, VRAM, duration, checkpoint and evaluations are stored in the linked run directories. Those directories are excluded from Git because models/private-data derivatives belong under controlled artifact storage.\n"
    save("docs/research/experiment_registry.md", registry)
    status = f"""# CURRENT R&D STATUS — 2026-09-10

**Runnable research infrastructure exists. Real skin-color accuracy and proposed technological advantage are NOT MEASURED.**

## COMPLETED

- Isolated Git repository, root AGENTS, README, locked dependencies and versioned schema. Evidence: source tree, uv.lock, [plan](implementation_plan.md).
- Archive context imported without execution and nine file hashes verified. Evidence: [provenance](../context/provenance.json). No original Luma source was independently verified.
- sRGB/D65/2° conversions and ΔE00 validated on all 34 Sharma reference pairs; schema/leakage/EXIF/mirroring/ROI/CCM/selection/calibration/model-loading/ONNX/memory/artifact-tamper tests. Evidence: **{verification["tests_passed"]} passing tests**, [verification.json](verification.json).
- A0/A1/A2 and compact C/C+/proposed train → OOF error model → calibration → test evaluation execute on deterministic synthetic data. Evidence: [registry](experiment_registry.md), [baseline report](../benchmarks/baseline_report.md).
- FP32 ONNX parity and batch-1 CPU/RTX4060/ORT timing measured for three learned models. Evidence: [deployment benchmark](../benchmarks/deployment_benchmark.md). No production validity follows.
- Data acquisition/repeatability/privacy protocols, real-training gate, independent review fixes and conservative API contract. Evidence: [DATA_REQUIRED](../data/DATA_REQUIRED.md), [review](review_record.md), [contract](../architecture/api_contract.md).
- Primary-source research: 13 close analogues, narrowed novelty hypothesis, separate code/weights/data licensing and technical Skolkovo/IP evidence map. Evidence: [prior_art](prior_art.md), [licensing](../ip/licensing_inventory.md), [evidence map](../skolkovo/evidence_map.md).

## IN PROGRESS

Scientific definition of measurement-aware reliability and strong C+ tuning protocol. The current attention is weakly supervised through Lab loss; it is not an independently validated measurement mask. Some recent prior-art full texts/quantitative tables and exact third-party artifact rights still need additional review before adoption/public claims.

## BLOCKED

G1 instrument validity/repeatability, G2 actual cross-camera/light problem, G3 real ML value, G4 proposed versus strong C+, G5 useful selective risk and G6 faithful compact deployment require a physical dataset. No real protocol-approval file, participant images or measured targets were fabricated. All six gates remain NOT MEASURED.

## FAILED / NEGATIVE OBSERVATIONS

No final engineering check failed. During development, tests exposed cheek inversion, EXIF mismatch, nonfinite acceptance, ambiguous bootstrap units, excessive cache memory and stale ONNX attribution; these are fixed. A Windows encoding failure in the verification transcript writer was also fixed by explicit UTF-8 child-process settings.

The synthetic two-epoch neural prototype is worse than classical A2 on the toy generator. This is an engineering observation, not a real-world negative result and not evidence for commercial deployment. See [negative results](negative_results.md).

## NOT STARTED

Instrument-target dataset collection, real segmentation/reliability supervision, tuned strong C+, rigorous unseen-device/light tests, full real ablations, paired-consistency objectives, risk-control guarantees under a preregistered protocol, teacher/student, production FP16/INT8/TensorRT, product-shade matching validation and live Luma integration.

## NEXT BEST ACTION

Run the 10–15 participant physical pilot with three camera pipelines, four lighting conditions, two capture repeats and at least three instrument readings per anatomical cheek per session. Lock canonical mirroring/registration and the repeatability threshold first. Run validation and repeatability before granting G1; then estimate study variance/sample size. Do not buy more compute or tune on a test set to compensate for missing measurement evidence.
"""
    save("docs/research/CURRENT_RND_STATUS.md", status)
    print(
        json.dumps(
            {
                "status": str(ROOT / "docs/research/CURRENT_RND_STATUS.md"),
                "tests": verification["tests_passed"],
                "source_hash": next(iter(run_metadata.values()))["source_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
