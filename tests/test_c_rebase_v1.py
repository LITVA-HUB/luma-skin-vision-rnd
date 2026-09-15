"""Synthetic-only scientific invariants for the frozen OOF trainer."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("c_rebase_v1_trainer", ROOT / "scripts/c_rebase_v1/train.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


def fixture_config():
    # Deliberately small synthetic fixture, not used by the actual experiment CLI.
    return {"experiment": "C_REBASE_V1", "seed": 17, "num_threads": 1, "epochs": 4,
            "phase_boundary": 2, "batch_size": 4, "gradient_clip_norm": 5.,
            "normalization": {"std_floor": 1e-6},
            "optimizer": {"name": "AdamW", "lr": .001, "weight_decay": .0001,
                          "betas": [.9, .999], "eps": 1e-8}}


def fixture_data():
    rng = np.random.default_rng(90210)
    x = rng.normal(size=(24, 36))
    target = np.array([55., 13., 17.]) + x[:, :3] * [3., 2., 1.]
    return {"color36": x, "target": target, "row_index": np.arange(24),
            "patient": np.repeat(["P01", "P02", "P03"], 8),
            "site": np.repeat(["S001", "S002", "S003", "S004", "S005", "S006"], 4),
            "image": np.array([f"I{x:04d}" for x in range(24)]),
            "fold": np.repeat([0, 1, 2], 8)}


def test_normalization_excludes_every_holdout_value():
    data = fixture_data()
    rows = np.flatnonzero(data["fold"] != 0)
    a = TRAIN.fit_normalization(data["color36"], data["target"], rows, 1e-6)
    data["color36"][data["fold"] == 0] += 1e8
    data["target"][data["fold"] == 0] -= 1e8
    b = TRAIN.fit_normalization(data["color36"], data["target"], rows, 1e-6)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_independent_fits_checkpoint_continuity_and_save_load(tmp_path):
    data = fixture_data()
    first = TRAIN.train_fold(fixture_config(), data, 0, tmp_path / "first", {"fixture": "synthetic"})
    second = TRAIN.train_fold(fixture_config(), data, 0, tmp_path / "second", {"fixture": "synthetic"})
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2]["final_state_sha256"] == second[2]["final_state_sha256"]
    assert first[2]["parameter_count"] == 2563
    assert all(first[2]["parameter_changed"].values())
    assert first[2]["save_load_predictions_bitwise_equal"]
    mid = torch.load(tmp_path / "first/epoch_002.pt", map_location="cpu", weights_only=True)
    final = torch.load(tmp_path / "first/epoch_004.pt", map_location="cpu", weights_only=True)
    assert mid["epoch"] == 2 and final["epoch"] == 4
    assert all(float(s["step"]) == 8 for s in mid["optimizer_state_dict"]["state"].values())
    assert all(float(s["step"]) == 16 for s in final["optimizer_state_dict"]["state"].values())
    assert np.array_equal(final["normalization"]["x_mean"].numpy(), data["color36"][8:].mean(0))
    assert final["numpy_shuffle_state"] != mid["numpy_shuffle_state"]


def test_patient_leakage_stops_before_fitting(tmp_path):
    data = fixture_data()
    data["patient"][8] = "P01"
    with pytest.raises(ValueError, match="Patient leakage"):
        TRAIN.train_fold(fixture_config(), data, 0, tmp_path / "bad", {})


def test_freeze_tampering_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("original")
    receipt = tmp_path / "freeze.json"
    receipt.write_text(json.dumps({"experiment": "C_REBASE_V1", "phase": "BEFORE_ANY_OOF_RESULTS",
                                   "files": {"config": {"sha256": TRAIN.sha256(path)}}}))
    TRAIN.verify_freeze({"config": path}, receipt)
    path.write_text("changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TRAIN.verify_freeze({"config": path}, receipt)


def test_holdout_target_does_not_change_training_or_prediction(tmp_path):
    data = fixture_data()
    first = TRAIN.train_fold(fixture_config(), data, 0, tmp_path / "first", {})
    data["target"][data["fold"] == 0] += [50., -100., 20.]
    second = TRAIN.train_fold(fixture_config(), data, 0, tmp_path / "second", {})
    assert np.array_equal(first[1], second[1])
    assert first[2]["final_state_sha256"] == second[2]["final_state_sha256"]
    assert first[2]["metrics"]["mean"] != second[2]["metrics"]["mean"]
