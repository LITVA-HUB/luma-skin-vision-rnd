"""Independent alignment/aggregation audit; no model fitting or selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chromaseed import predict  # noqa: E402

from luma_skin_vision.color import delta_e00, srgb_to_lab  # noqa: E402

RUN = ROOT / "experiments/runs/chromaseed_v1"
FROZEN = ROOT / "experiments/runs/chromaseed_frozen_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_v1"
CACHE = ROOT.parents[1] / "luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def main():
    global RUN, FROZEN, OUT, CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--frozen-run", type=Path, default=FROZEN)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    RUN, FROZEN, OUT, CACHE = args.run, args.frozen_run, args.output, args.cache
    source_lock = read(RUN / "source_lock.json")
    assert digest(CACHE) == source_lock["cache_sha256"]
    for base in (RUN, FROZEN):
        for path, expected in read(base / "source_lock.json")["source_hashes"].items():
            assert digest(ROOT / path) == expected
    with np.load(CACHE, allow_pickle=False) as d:
        real = {k: d[k] for k in ("color", "target", "patient", "site", "device")}
    palette_receipt = read(RUN / "palette/receipt.json")
    palette = {}
    max_lab_error = 0.
    for role in ("train", "validation"):
        path = RUN / "palette" / f"{role}.npz"
        assert digest(path) == palette_receipt[role]["sha256"]
        d = load(path)
        max_lab_error = max(max_lab_error, float(np.max(abs(srgb_to_lab(d["canonical_rgb"]) - d["target"]))))
        palette[role] = {tuple(row) for row in d["canonical_rgb"]}
    assert not (palette["train"] & palette["validation"])
    frozen_primary = read(RUN / "frozen_selections.json")
    evaluation = read(RUN / "evaluation.json")
    assert digest(RUN / "frozen_selections.json") == evaluation["frozen_selection_sha256"]
    for group in ("final_trace_hashes", "role_hashes"):
        for path, expected in frozen_primary[group].items():
            assert digest(RUN / path) == expected
    means = {}
    checks, largest_prediction_gap = 0, 0.
    for row in evaluation["models"]:
        protocol, arm, seed, step = row["protocol"], row["arm"], row["seed"], row["step"]
        role = load(RUN / protocol / "roles.npz")
        a, held = role["fit"], role["held"]
        assert set(real["patient"][a]).isdisjoint(real["patient"][held])
        for fold in range(3):
            fit_person = real["patient"][a]
            assert set(fit_person[role["folds"] == fold]).isdisjoint(fit_person[role["folds"] != fold])
        folder = RUN / protocol / "final" / f"{arm}_s{seed}"
        receipt = read(folder / "trace.json")
        model_path = folder / f"step{step}.npz"
        assert digest(model_path) == receipt["artifacts"][str(step)]["sha256"]
        outer = load(folder / f"outer{step}.npz")
        for key, original in (("target", "target"), ("person", "patient"), ("site", "site"), ("camera", "device")):
            np.testing.assert_array_equal(outer[key], real[original][held])
        predicted = predict(load(model_path), real["color"][held])
        largest_prediction_gap = max(largest_prediction_gap, float(np.max(abs(predicted - outer["prediction"]))))
        np.testing.assert_allclose(predicted, outer["prediction"], atol=3e-5, rtol=0)
        errors = delta_e00(outer["prediction"], outer["target"])
        person_means = np.array([errors[outer["person"] == p].mean() for p in np.unique(outer["person"])])
        assert abs(person_means.mean() - row["metrics"]["person_mean"]) < 1e-12
        assert abs(errors.mean() - row["metrics"]["image_mean"]) < 1e-12
        means[(protocol, arm, seed, step)] = person_means
        checks += 1
    frozen_checks = 0
    feval = read(FROZEN / "evaluation.json")
    assert digest(FROZEN / "frozen_selections.json") == feval["frozen_selection_sha256"]
    for row in feval["models"]:
        model_path = FROZEN / row["path"]
        assert digest(model_path) == row["sha256"]
        model = load(model_path)
        source_path = (RUN / "pretrain" / f"initial_s{row['seed']}.npz" if row["arm"] == "random_basis" else
                       RUN / "pretrain" / f"{row['arm']}_s{row['seed']}" / "step2048.npz")
        source = load(source_path)
        for key in ("hidden_w", "hidden_b"):
            np.testing.assert_array_equal(model[key], source[key])
        outer = load(model_path.with_name(f"{row['arm']}_s{row['seed']}_outer.npz"))
        held = load(FROZEN / row["protocol"] / "roles.npz")["held"]
        for key, original in (("target", "target"), ("person", "patient"), ("site", "site"), ("camera", "device")):
            np.testing.assert_array_equal(outer[key], real[original][held])
        e = delta_e00(outer["prediction"], outer["target"])
        pm = np.mean([e[outer["person"] == p].mean() for p in np.unique(outer["person"])])
        assert abs(pm - row["metrics"]["person_mean"]) < 1e-12
        frozen_checks += 1
    bootstrap = []
    rng = np.random.default_rng(905731)
    for protocol in ("mixed", "slr_to_ipod", "ipod_to_slr"):
        for candidate in ("clean_palette", "rendered_palette"):
            for control in ("scratch", "skin_long", "shuffled_palette"):
                for step in (64, 256, 1024):
                    delta = np.mean([means[(protocol, candidate, seed, step)] - means[(protocol, control, seed, step)]
                                     for seed in (17, 29, 43)], axis=0)
                    draws = rng.integers(0, len(delta), size=(20000, len(delta)))
                    distribution = delta[draws].mean(1)
                    bootstrap.append({"protocol": protocol, "candidate": candidate, "control": control, "step": step,
                                      "difference": float(delta.mean()), "people": len(delta),
                                      "descriptive_95_percentile": np.quantile(distribution, [.025, .975]).tolist(),
                                      "people_improved": int(np.sum(delta < 0))})
    result = {"status": "COMPLETE", "primary_checkpoint_alignments": checks,
              "frozen_basis_alignments_and_hidden_layer_identity": frozen_checks,
              "max_replayed_prediction_component_gap": largest_prediction_gap,
              "synthetic_target_lab_error": max_lab_error, "exact_palette_fit_validation_overlap": 0,
              "bootstrap_scope": "Descriptive only: reused people, overlapping protocols, optimization seeds are not new people; hyperparameter selection uncertainty omitted",
              "paired_differences": bootstrap}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "paired_differences"}, indent=2))


if __name__ == "__main__":
    main()
