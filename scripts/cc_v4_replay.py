"""Replay both selected checkpoints on source validation, never on test rows."""

import json
from pathlib import Path

import numpy as np
import torch
from cc_v2_statistics import read_npz_rows
from cc_v4_experiment import evaluate
from cc_v4_model import CorrectionEvidenceNet

from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "docs/benchmarks/cc_v4/checkpoint_replay.json"
    if out.exists():
        raise FileExistsError("Replay receipt already exists")
    torch.set_num_threads(4)
    data = ROOT / "data/processed/cc128"
    rows = json.loads((data / "cube_manifest.json").read_text(encoding="utf-8"))
    selected = np.array([i for i, row in enumerate(rows) if row["subset"] == "val"])
    if len(selected) != 119:
        raise ValueError("Unexpected validation population")
    x = torch.from_numpy(read_npz_rows(data / "cube.npz", "images", selected, 2234).astype(np.float32))
    gt = torch.from_numpy(read_npz_rows(data / "cube.npz", "gt", selected, 2234).astype(np.float32))
    receipt = {"device": "cpu", "torch": torch.__version__, "script_sha256": sha256(Path(__file__)),
               "scope": "Only119source-development validation rows; no other GT decoded", "models": {}}
    for mode in ("posterior", "action", "transport"):
        run = ROOT / "experiments/runs" / f"ccv4_{mode}_e120_s17"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if [rows[i]["id"] for i in selected] != config["validation_ids"]:
            raise ValueError("Validation identity mismatch")
        for name, expected in config["data_hashes"].items():
            if sha256(data / name) != expected:
                raise ValueError("Changed data")
        for name, expected in config["scripts_sha256"].items():
            if sha256(ROOT / "scripts" / name) != expected:
                raise ValueError(f"Changed executable: {name}")
        manifest = json.loads((run / "artifact_manifest.json").read_text(encoding="utf-8"))["sha256"]
        for name, expected in manifest.items():
            if sha256(run / name) != expected:
                raise ValueError(f"Changed run artifact: {name}")
        model = CorrectionEvidenceNet(mode=mode).eval()
        receipt["models"][mode] = {}
        for checkpoint, predictions in (("best.pt", "best_validation.npz"), ("best_point.pt", "best_point_validation.npz")):
            model.load_state_dict(torch.load(run / checkpoint, map_location="cpu", weights_only=True))
            _, actual = evaluate(model, x, gt, 32)
            with np.load(run / predictions, allow_pickle=False) as saved:
                defects = {}
                for name in ("point_action", "trajectory_actions", "trajectory_risk", "base_reproduction", "trajectory_reproduction"):
                    defects[name] = float(np.max(np.abs(actual[name].astype(np.float64) - saved[name].astype(np.float64))))
                if not np.array_equal(actual["valid"], saved["valid"]):
                    raise ValueError("Replayed validity mismatch")
            # Separate CPU/GPU convolution rounding is expected. A different
            # grid decision causes a much larger log-action mismatch and fails.
            if max(defects.values()) > 1e-3:
                raise ValueError(f"CPU checkpoint replay mismatch: {mode}/{checkpoint}: {defects}")
            receipt["models"][mode][checkpoint] = {
                "checkpoint_sha256": sha256(run / checkpoint), "predictions_sha256": sha256(run / predictions),
                "max_absolute_defects": defects, "tolerance": 1e-3, "status": "passed",
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
