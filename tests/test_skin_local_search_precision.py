"""Checks for frozen post-training storage formats and their evaluation boundary."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _model(method, rows=7):
    common = {
        "method": np.asarray(method),
        "x_mean": np.linspace(-2, 2, 36, dtype=np.float32),
        "x_std": np.linspace(0.5, 1.5, 36, dtype=np.float32),
        "y_mean": np.array([50, 5, 10], dtype=np.float32),
        "y_std": np.array([4, 3, 2], dtype=np.float32),
    }
    if method == "ridge":
        common["beta"] = np.linspace(-1, 1, 37 * 3, dtype=np.float32).reshape(37, 3)
    elif method == "krr":
        common.update(
            centers=np.linspace(-1, 1, rows * 36, dtype=np.float32).reshape(rows, 36),
            widths=np.asarray(1.25, dtype=np.float32),
            beta=np.linspace(-0.8, 0.9, rows * 3, dtype=np.float32).reshape(rows, 3),
        )
    elif method in ("random_rbf", "guided_rbf"):
        atoms = 4
        common.update(
            centers=np.linspace(-1, 1, atoms * 36, dtype=np.float32).reshape(atoms, 36),
            widths=np.array([0.5, 1, 2, 1], dtype=np.float32),
            beta=np.linspace(-0.6, 0.7, (37 + atoms) * 3, dtype=np.float32).reshape(
                37 + atoms, 3
            ),
        )
    else:
        common.update(
            hidden_w=np.linspace(-1, 1, 64 * 36, dtype=np.float32).reshape(64, 36),
            hidden_b=np.zeros(64, dtype=np.float32),
            out_w=np.linspace(-0.4, 0.5, 3 * 64, dtype=np.float32).reshape(3, 64),
            out_b=np.zeros(3, dtype=np.float32),
            skip_w=np.linspace(-0.2, 0.3, 3 * 36, dtype=np.float32).reshape(3, 36),
        )
    return common


def _populate_final_models(run):
    methods = {
        "ridge": (17,),
        "krr": (17,),
        "random_rbf": (17, 29, 43),
        "guided_rbf": (17, 29, 43),
        "mlp": (17, 29, 43),
    }
    for protocol in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        folder = run / protocol / "final"
        folder.mkdir(parents=True)
        for method, seeds in methods.items():
            for seed in seeds:
                np.savez(folder / f"{method}_s{seed}.npz", **_model(method))


def test_storage_formats_and_int8_axis_scales_are_explicit():
    import skin_local_search_precision as precision

    source = _model("mlp")
    fp32 = precision.pack_model(source, "fp32_reference")
    fp16 = precision.pack_model(source, "fp16")
    int8 = precision.pack_model(source, "int8")

    for key, value in source.items():
        if value.dtype.kind == "f":
            assert fp32[key].dtype == np.float32
            assert fp16[key].dtype == np.float16
    assert precision.payload_stats(fp16) == {
        "numeric_scalars": 2749,
        "numeric_bytes": 5498,
        "int8_scalars": 0,
        "float16_scalars": 2749,
        "float32_scalars": 0,
        "quantization_scale_scalars": 0,
    }

    for key in ("x_mean", "x_std", "y_mean", "y_std"):
        assert int8[key].dtype == np.float32
    for key in ("hidden_w", "hidden_b", "out_w", "out_b", "skip_w"):
        assert int8[key].dtype == np.int8
        assert np.all(int8[f"{key}_scale"] > 0)
    assert int8["hidden_w_scale"].shape == (64,)
    assert int8["out_w_scale"].shape == (3,)
    assert int8["skip_w_scale"].shape == (3,)
    assert int8["hidden_b_scale"].shape == ()
    assert int8["out_b_scale"].shape == ()

    restored = precision.dequantize_model(int8)
    assert all(value.dtype == np.float32 for value in restored.values() if value.dtype.kind == "f")
    assert np.isfinite(restored["hidden_w"]).all()
    assert np.max(np.abs(restored["hidden_w"] - source["hidden_w"])) <= float(
        int8["hidden_w_scale"].max()
    ) / 2 + 1e-7


def test_rbf_int8_uses_beta_columns_centers_features_and_one_width_scale():
    import skin_local_search_precision as precision

    packed = precision.pack_model(_model("guided_rbf"), "int8")
    assert packed["beta_scale"].shape == (3,)
    assert packed["centers_scale"].shape == (36,)
    assert packed["widths_scale"].shape == ()
    assert np.all(packed["beta_scale"] > 0)
    assert np.all(packed["centers_scale"] > 0)
    assert float(packed["widths_scale"]) > 0


def test_prepare_freezes_exactly_33_primary_models_without_outer_arrays(tmp_path):
    import skin_local_search_precision as precision

    run = tmp_path / "run"
    _populate_final_models(run)
    outer_decoy = run / "mixed" / "final" / "ridge_s17_outer.npz"
    np.savez(outer_decoy, prediction=np.zeros((2, 3)), target=np.ones((2, 3)))

    manifest = precision.prepare(run)
    assert manifest["source_model_count"] == 33
    assert len(manifest["models"]) == 33
    assert all("_outer" not in entry["source_model"] for entry in manifest["models"])
    artifacts = list((run / "precision").glob("*/*/*/quantized.npz"))
    assert len(artifacts) == 99
    assert (run / "precision" / "manifest.sha256").read_text().strip() == hashlib.sha256(
        (run / "precision" / "manifest.json").read_bytes()
    ).hexdigest()
    assert precision.prepare(run) == manifest

    source = run / manifest["models"][0]["source_model"]
    with source.open("ab") as stream:
        stream.write(b"changed")
    try:
        precision.prepare(run)
    except ValueError as error:
        assert "source model hash" in str(error)
    else:
        raise AssertionError("a frozen source mutation must be rejected")


def test_evaluate_reports_every_format_without_selecting_one(tmp_path, monkeypatch):
    import skin_local_search_precision as precision

    run = tmp_path / "run"
    _populate_final_models(run)
    rows = 6
    rng = np.random.default_rng(91)
    data = {
        "color": rng.normal(size=(rows, 36)).astype(np.float64),
        "target": rng.normal(size=(rows, 3)).astype(np.float64),
        "patient": np.array(["a", "a", "b", "b", "c", "c"]),
        "site": np.array(["x", "y", "x", "y", "x", "y"]),
        "device": np.array(["SLR", "SLR", "ipod", "ipod", "SLR", "ipod"]),
    }
    for protocol in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        folder = run / protocol
        np.savez(folder / "roles.npz", fit=np.array([1, 1, 1, 0, 0, 0], dtype=bool),
                 held=np.array([0, 0, 0, 1, 1, 1], dtype=bool), folds=np.array([0, 1, 2]))
    cache = tmp_path / "train.npz"
    np.savez(cache, **data)
    monkeypatch.setattr(precision, "CACHE_HASH", hashlib.sha256(cache.read_bytes()).hexdigest())

    precision.prepare(run)
    result = precision.evaluate(run, cache)
    assert result["formats"] == ["fp32_reference", "fp16", "int8"]
    assert result["format_was_selected_on_outer"] is False
    assert len(result["models"]) == 99
    assert {row["format"] for row in result["models"]} == set(result["formats"])
    assert all("person_delta_e" in row and "deviation_from_fp32" in row for row in result["models"])
    stored = json.loads((run / "precision" / "evaluation.json").read_text())
    assert stored == result
