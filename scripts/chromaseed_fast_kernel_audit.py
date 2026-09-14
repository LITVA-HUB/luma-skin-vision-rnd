"""Independent width/role/decision/SVD/prediction checks for the frozen KF run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from chromaseed_kernel_audit import direct_kernel, independent_predict, norm, unpack
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, weights_for, write_json

ROOT = Path(__file__).resolve().parents[1]


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nz(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def key_for(arm, rank, seed, wi, ai):
    return f"{arm}_k{rank:03d}_s{seed}_w{wi}_a{ai}"


def row_hash(rows):
    return hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest()


def positive_median(distances):
    values = np.sort(np.asarray(distances)[np.asarray(distances) > 1e-10])
    return max(float(values[(len(values) - 1) // 2]), 1e-6) if len(values) else None


def independent_widths(z):
    distances = np.concatenate([np.sqrt(np.mean((z[i + 1:] - z[i]) ** 2, axis=1)) for i in range(len(z) - 1)])
    exact = positive_median(distances) or 1e-6
    result = {("dense_exact", s): exact for s in (17, 29, 43)}
    result.update({("column_exact", s): exact for s in (17, 29, 43)})
    streams = {}
    for seed in (17, 29, 43):
        pairs = np.random.default_rng(seed + 104729).integers(0, np.array([len(z), len(z) - 1]), size=(16384, 2), dtype=np.int64)
        pairs[:, 1] = np.where(pairs[:, 1] >= pairs[:, 0], pairs[:, 1] + 1, pairs[:, 1])
        distance = np.sqrt(np.mean((z[pairs[:, 0]] - z[pairs[:, 1]]) ** 2, axis=1))
        for count in (1024, 4096, 16384):
            width = positive_median(distance[:count])
            if width is None:
                width = positive_median(np.sqrt(np.mean((z - z[0]) ** 2, axis=1))) or 1e-6
            result[(f"column_pairs{count}", seed)] = width
            streams[(f"column_pairs{count}", seed)] = hashlib.sha256(pairs[:count].tobytes()).hexdigest()
    return result, streams


def summarize(predictions, data, rows):
    records = [error_summary(p, data["target"][rows], data["patient"][rows], data["site"][rows])[0] for p in predictions]
    return float(np.mean([r["person_mean"] for r in records])), float(np.mean([r["p90"] for r in records]))


def refit_svd(model, z, yn, weights, indices, alpha):
    columns = direct_kernel(z, z[indices], float(model["width"]))
    sub = columns[indices]
    u, singular, _ = np.linalg.svd(.5 * (sub + sub.T), full_matrices=False)
    use = singular > 1e-8 * singular.max()
    whitening = u[:, use] / np.sqrt(singular[use])
    phi = columns @ whitening
    gram = phi.T @ (weights[:, None] * phi) + alpha * np.eye(phi.shape[1])
    cross = phi.T @ (weights[:, None] * yn)
    return (whitening @ np.linalg.solve(gram, cross)).astype(np.float32)


def audit(run, cache, parent, output):
    lock = js(run / "source_lock.json")
    choices = js(run / "selections.json")
    results = js(run / "results.json")
    for path, expected in lock["sources"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in lock["parent_bindings"].items():
        assert sha(parent / path) == expected, path
    assert sha(cache) == CACHE_HASH
    assert results["source_lock_sha256"] == choices["source_lock_sha256"] == sha(run / "source_lock.json")
    assert results["selection_sha256"] == sha(run / "selections.json")
    with np.load(cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "site", "device")}
    counts = {"bank_hashes": 0, "normalizers_widths_centers": 0, "oof_probe_rows": 0,
              "matched_dense_column_controls": 0, "primary_choices": 0, "fast_choices": 0, "size_choices": 0,
              "selected_artifact_hashes": 0, "svd_refits": 0, "final_prediction_rows": 0}
    maxima = {"prediction_component_drift": 0., "svd_component_drift": 0., "matched_dense_column_component_drift": 0.}
    paired = []
    for role, (fit, held) in roles(data["patient"], data["device"]).items():
        fit_rows, held_rows = np.flatnonzero(fit), np.flatnonzero(held)
        folds = folds_for(data["patient"][fit], data["device"][fit])
        files, oof_rows = [], []
        final_bank, final_receipt = None, None
        for fold in (0, 1, 2, None):
            idx = fit_rows if fold is None else fit_rows[folds != fold]
            query_rows = None if fold is None else fit_rows[folds == fold]
            directory = run / "final" / role / "bank" if fold is None else run / "inner" / role / f"fold{fold}"
            receipt = js(directory / "receipt.json")
            assert receipt["source_lock_sha256"] == sha(run / "source_lock.json")
            assert receipt["fit_rows_sha256"] == row_hash(idx)
            if query_rows is not None:
                assert receipt["query_rows_sha256"] == row_hash(query_rows)
                assert not set(data["patient"][idx]) & set(data["patient"][query_rows])
            for name, expected in receipt["files"].items():
                assert sha(directory / name) == expected
                counts["bank_hashes"] += 1
            arrays = nz(directory / "models.npz")
            first = unpack(arrays, next(iter(receipt["models"])))
            x, y = data["color"][idx].astype(np.float64), data["target"][idx].astype(np.float64)
            prep = {"x_mean": x.mean(0).astype(np.float32), "x_std": np.maximum(x.std(0), 1e-6).astype(np.float32),
                    "y_mean": y.mean(0).astype(np.float32), "y_std": np.maximum(y.std(0), 1e-6).astype(np.float32)}
            z = norm(prep, data["color"][idx])
            widths, streams = independent_widths(z)
            probes = data["color"][idx[:3]]
            if query_rows is not None:
                oof = nz(directory / "oof.npz")
                np.testing.assert_array_equal(oof["row_indices"], query_rows)
                positions = np.linspace(0, len(query_rows) - 1, 3, dtype=int)
                probes = data["color"][query_rows[positions]]
                files.append(oof)
                oof_rows.append(query_rows)
            for key, config in receipt["models"].items():
                model = unpack(arrays, key)
                for field in prep:
                    np.testing.assert_array_equal(model[field], prep[field])
                indices = np.asarray(config["fit_center_indices"])
                assert len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(idx)))
                np.testing.assert_array_equal(model["centers"], z[indices].astype(np.float32))
                base = widths[(config["arm"], config["seed"])]
                np.testing.assert_allclose(base, config["base_width"], atol=1e-12, rtol=0)
                assert model["width"] == np.float32(base * config["width_factor"])
                if config["arm"].startswith("column_pairs"):
                    assert config["width_info"]["pair_stream_sha256"] == streams[(config["arm"], config["seed"])]
                counts["normalizers_widths_centers"] += 1
                prediction = independent_predict(model, probes)
                if query_rows is not None:
                    drift = float(np.max(np.abs(prediction - oof[f"pred__{key}"][positions])))
                    assert drift <= 1e-8, (key, drift)
                    maxima["prediction_component_drift"] = max(maxima["prediction_component_drift"], drift)
                    counts["oof_probe_rows"] += len(probes)
                if config["arm"] == "column_exact":
                    dense = unpack(arrays, key_for("dense_exact", config["rank"], config["seed"], config["width_index"], config["alpha_index"]))
                    drift = float(np.max(np.abs(prediction - independent_predict(dense, probes))))
                    assert drift <= .002, (key, drift)
                    maxima["matched_dense_column_component_drift"] = max(maxima["matched_dense_column_component_drift"], drift)
                    counts["matched_dense_column_controls"] += 1
            if fold is None:
                final_bank, final_receipt = arrays, receipt
            del first
        joined_rows = np.concatenate(oof_rows)
        order = np.argsort(joined_rows)
        np.testing.assert_array_equal(joined_rows[order], fit_rows)
        oof = {key: np.concatenate([f[key] for f in files])[order] for key in files[0] if key != "row_indices"}
        choice = choices["roles"][role]
        scores = {}
        for arm, ranks in choice["primary"].items():
            for rank, selected in ranks.items():
                values = []
                for wi, factor in enumerate((.5, 1., 2.)):
                    for ai, alpha in enumerate((.1, 1., 10.)):
                        predictions = [oof[f"pred__{key_for(arm, int(rank), seed, wi, ai)}"] for seed in (17, 29, 43)]
                        mean, tail = summarize(predictions, data, fit_rows)
                        values.append((mean, alpha, factor, wi, ai, tail))
                best = min(values, key=lambda c: c[:3])
                assert best[3:5] == (selected["width_index"], selected["alpha_index"])
                np.testing.assert_allclose([best[0], best[5]], [selected["selected"]["person_mean"], selected["selected"]["p90"]], rtol=0, atol=1e-10)
                scores[(arm, int(rank))] = (best[0], best[5])
                counts["primary_choices"] += 1
        fast_choices = {}
        for rank in (64, 128, 256):
            mean, tail = scores[("column_exact", rank)]
            expected_arm = "column_exact"
            for budget in (1024, 4096, 16384):
                arm = f"column_pairs{budget}"
                candidate = scores[(arm, rank)]
                if candidate[0] <= mean + .05 and candidate[1] <= tail + .10:
                    expected_arm = arm
                    break
            assert choice["fast"][str(rank)]["selected"]["arm"] == expected_arm
            fast_choices[rank] = expected_arm
            counts["fast_choices"] += 1
        best_rank = min(fast_choices, key=lambda k: (scores[(fast_choices[k], k)][0], k))
        mean, tail = scores[(fast_choices[best_rank], best_rank)]
        eligible = [k for k in fast_choices if scores[(fast_choices[k], k)][0] <= mean + .05 and scores[(fast_choices[k], k)][1] <= tail + .10]
        assert choice["size"]["selected"]["rank"] == min(eligible)
        assert choice["size"]["selected"]["arm"] == fast_choices[min(eligible)]
        counts["size_choices"] += 1
        base_model = unpack(final_bank, next(iter(final_receipt["models"])))
        z = norm(base_model, data["color"][fit_rows])
        yn = (data["target"][fit_rows] - base_model["y_mean"]) / base_model["y_std"]
        weights = weights_for(data["patient"][fit_rows], data["site"][fit_rows])
        person_values = {}
        for record in [r for r in results["records"] if r["role"] == role]:
            arm, rank, seed = record["arm"], record["rank"], record["seed"]
            name = f"{arm}_k{rank:03d}_s{seed}.npz"
            path, prediction_path = run / "selected" / role / name, run / "evaluated" / role / name
            assert sha(path) == record["model_sha256"] and sha(prediction_path) == record["prediction_sha256"]
            counts["selected_artifact_hashes"] += 2
            model, saved = nz(path), nz(prediction_path)
            np.testing.assert_array_equal(saved["row_indices"], held_rows)
            prediction = independent_predict(model, data["color"][held_rows])
            drift = float(np.max(np.abs(prediction - saved["prediction"])))
            assert drift <= 1e-8
            maxima["prediction_component_drift"] = max(maxima["prediction_component_drift"], drift)
            measures, per_person = error_summary(prediction, data["target"][held_rows], data["patient"][held_rows], data["site"][held_rows])
            for metric, value in measures.items():
                np.testing.assert_allclose(value, record["metrics"][metric], atol=1e-9, rtol=0)
            key = key_for(arm, rank, seed, record["width_index"], record["alpha_index"])
            config = final_receipt["models"][key]
            expected_payload = unpack(final_bank, key)
            for field in model:
                np.testing.assert_array_equal(model[field], expected_payload[field])
            beta = refit_svd(model, z, yn, weights, np.asarray(config["fit_center_indices"]), config["alpha"])
            replayed = {**model, "coefficient": beta}
            reference = independent_predict(replayed, data["color"][held_rows[:7]])
            drift = float(np.max(np.abs(reference - prediction[:7])))
            assert drift <= .002, (name, drift)
            maxima["svd_component_drift"] = max(maxima["svd_component_drift"], drift)
            counts["svd_refits"] += 1
            counts["final_prediction_rows"] += len(held_rows)
            person_values[(arm, rank, seed)] = per_person
        rng = np.random.default_rng(738413)
        for rank, arm in fast_choices.items():
            new = np.mean([person_values[(arm, rank, seed)] for seed in (17, 29, 43)], axis=0)
            reference = np.mean([person_values[("dense_exact", rank, seed)] for seed in (17, 29, 43)], axis=0)
            difference = new - reference
            draws = rng.integers(len(difference), size=(20000, len(difference)))
            paired.append({"role": role, "rank": rank, "arm": arm, "n_people": len(difference),
                           "mean_delta_vs_matched_dense": float(difference.mean()),
                           "descriptive_person_bootstrap_95": np.quantile(difference[draws].mean(1), [.025, .975]).tolist()})
        print(f"AUDITED {role}", flush=True)
    assert counts["normalizers_widths_centers"] == 4860 and counts["svd_refits"] == 135
    write_json(output / "audit.json", {"passed": True, "source_lock_sha256": sha(run / "source_lock.json"),
               "selection_sha256": sha(run / "selections.json"), "audit_source_sha256": sha(Path(__file__)),
               "dependencies": {path: sha(ROOT / path) for path in ("scripts/chromaseed_kernel_audit.py", "scripts/chromaseed_refine_audit.py", "scripts/skin_local_search_train.py", "src/luma_skin_vision/color.py")},
               "checks": counts, "maxima": maxima, "paired": paired,
               "evidence": "separate width/decision/direct-kernel/SVD calculation on reused exploratory people, not external replication or fresh confirmation"})


def main():
    parser = argparse.ArgumentParser()
    for arg in ("run", "cache", "parent-run", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.cache, args.parent_run, args.output)


if __name__ == "__main__":
    main()
