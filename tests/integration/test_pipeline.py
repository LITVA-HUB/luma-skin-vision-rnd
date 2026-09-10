import json

import numpy as np
import pytest

from luma_skin_vision.api import analyze
from luma_skin_vision.synthetic import generate
from luma_skin_vision.training import calibrate_run, evaluate_run, load_run, train


def config(path, method="baseline_c"):
    return {
        "schema_version": "1.0",
        "dataset": str(path),
        "method": method,
        "seed": 7,
        "resolution": 32,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_accumulation": 1,
        "precision": "fp32",
        "device": "cpu",
        "oof_folds": 2,
        "correction": "none",
    }


def test_train_calibrate_evaluate_load_and_conservative_api(tmp_path):
    manifest = generate(tmp_path / "data", subjects=12, size=32)
    run = train(config(manifest), tmp_path / "runs")
    model, metadata = load_run(run)
    assert metadata["data_kind"] == "SYNTHETIC"
    assert metadata["parameters"] < 10_000_000
    assert metadata["status"] == "COMPLETED"
    oof = json.loads((run / "oof_audit.json").read_text())
    assert all(not set(f["fit_subjects"]) & set(f["held_out_subjects"]) for f in oof)
    calibration = calibrate_run(run)
    assert calibration.domain_validated is False
    report = evaluate_run(run)
    assert report["data_kind"] == "SYNTHETIC"
    assert report["split"] == "test"
    assert np.isfinite(report["metrics"]["mean_delta_e00"])
    import torch

    x = torch.ones(1, 3, 32, 32) * 0.5
    aux = torch.zeros(1, 12)
    model.eval()
    second, _ = load_run(run)
    with torch.no_grad():
        np.testing.assert_array_equal(model(x, aux).numpy(), second(x, aux).numpy())
    result = analyze(run, manifest.parent / "images" / "syn_000_d0_l0_r0.jpg", bbox=[0, 0, 32, 32])
    assert result["status"] == "UNSUPPORTED"
    assert result["measurement"] is None and result["profile_update"]["may_update"] is False
    from luma_skin_vision.export import benchmark_run, export_run

    export_run(run)
    onnx = run / "model.onnx"
    onnx.write_bytes(onnx.read_bytes() + b"stale-artifact")
    with pytest.raises(ValueError, match="ONNX artifact"):
        benchmark_run(run, onnx=True, iterations=5)


def test_preparation_retains_compact_crops_not_full_photos(tmp_path):
    import tracemalloc

    from luma_skin_vision.data import validate_records
    from luma_skin_vision.training import prepare

    manifest = generate(tmp_path, subjects=8, size=256)
    rows = validate_records(manifest)
    tracemalloc.start()
    prepared = prepare(manifest, rows, config(manifest))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert prepared["images"].shape == (128, 3, 32, 32)
    assert peak < 48 * 1024 * 1024


def test_baseline_train_and_test_fingerprint(tmp_path):
    manifest = generate(tmp_path / "data", subjects=12, size=32)
    run = train(config(manifest, "baseline_a0"), tmp_path / "runs")
    assert evaluate_run(run)["metrics"]["mean_delta_e00"] >= 0
    with manifest.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate_run(run)
