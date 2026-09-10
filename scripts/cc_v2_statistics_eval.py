"""Source-only statistics selectors; target evaluation requires a separate head lock."""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from luma_skin_vision.cc.core import reproduction, selective_curve
from luma_skin_vision.cc.v2_experiment import split_indices
from luma_skin_vision.data import sha256
from luma_skin_vision.experiment import source_identity, write_json


def sibling(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


statistics, selector = sibling("cc_v2_statistics"), sibling("cc_v2_select")
GRID = ("ridge1", "ridge10", "ridge100", "hgb3", "hgb7")
BLOCKS = ("context", "cheap", "combined")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bindings(paths):
    return {str(Path(path).resolve()): sha256(Path(path)) for path in paths}


def check_bindings(expected):
    if any(
        not Path(path).is_file() or sha256(Path(path)) != digest
        for path, digest in expected.items()
    ):
        raise ValueError("Frozen artifact/input binding changed")


def prediction(run, candidate, npz, manifest, output):
    sidecar = output.with_suffix(".manifest.json")
    if not output.exists():
        statistics.predict_frozen(run, candidate, npz, manifest, output)
    bound = load(sidecar)
    screen = load(run / "screen.json")
    record = next(r for r in screen["candidates"] if r["candidate"] == candidate)
    if (
        bound["candidate"] != candidate
        or bound["screen_sha256"] != sha256(run / "screen.json")
        or bound["model_sha256"] != record["model_sha256"]
        or bound["script_sha256"] != sha256(Path(statistics.__file__))
        or bound["prediction_sha256"] != sha256(output)
        or Path(bound["input_npz"]).resolve() != npz.resolve()
        or Path(bound["input_manifest"]).resolve() != manifest.resolve()
        or bound["input_hashes"] != {"npz": sha256(npz), "manifest": sha256(manifest)}
    ):
        raise ValueError("Prediction input/artifact binding mismatch")
    rows = load(manifest)
    with np.load(output, allow_pickle=False) as cache:
        values = {key: cache[key] for key in cache.files}
    if values["ids"].tolist() != [row["id"] for row in rows]:
        raise ValueError("Prediction IDs differ from manifest")
    return rows, values


def fit(run, candidate, data, out, estimator_lock, *, expected_counts=(259, 268)):
    run, data, out, estimator_lock = map(Path, (run, data, out, estimator_lock))
    lock, screen = load(estimator_lock), load(run / "screen.json")
    if candidate not in lock["statistics_controls"] or lock[
        "statistics_source_screen_sha256"
    ] != sha256(run / "screen.json"):
        raise ValueError("Estimator selection lock mismatch")
    if out.exists():
        raise FileExistsError(out)
    if str(data.resolve()) != screen["data_root"] or screen["data_hashes"] != {
        name: sha256(data / name) for name in screen["data_hashes"]
    }:
        raise ValueError("Source fitting dataset binding mismatch")
    model = next(r for r in screen["candidates"] if r["candidate"] == candidate)
    frozen = bindings(
        [
            Path(__file__),
            Path(statistics.__file__),
            Path(selector.__file__),
            run / "screen.json",
            run / model["model_file"],
            estimator_lock,
            data / "cube.npz",
            data / "cube_manifest.json",
        ]
    )
    rows, values = prediction(
        run,
        candidate,
        data / "cube.npz",
        data / "cube_manifest.json",
        out / "source_predictions.npz",
    )
    ix = split_indices(rows, "official")
    if (len(ix["risk"]), len(ix["cal"])) != tuple(expected_counts):
        raise ValueError("Unexpected immutable risk/cal split sizes")
    selected = np.r_[ix["risk"], ix["cal"]]
    ordered = np.sort(selected)
    gt = statistics.read_npz_rows(data / "cube.npz", "gt", ordered, expected_rows=len(rows))[
        np.searchsorted(ordered, selected)
    ]
    if not values["valid"][selected].all():
        raise ValueError("Unsupported source risk/cal inputs require protocol revision")
    errors = reproduction(values["pred"][selected], gt)
    n, ids = len(ix["risk"]), [rows[i]["id"] for i in ix["risk"]]
    folds = list(GroupKFold(5).split(np.arange(n), groups=[rows[i]["group"] for i in ix["risk"]]))
    frozen.update(
        bindings([out / "source_predictions.npz", out / "source_predictions.manifest.json"])
    )
    state = {
        "candidate": candidate,
        "run": str(run.resolve()),
        "data": str(data.resolve()),
        "source_hash": screen["source_identity"]["source_hash"],
        "bindings": frozen,
        "scope": "Source risk-fit/calibration only; no target labels or errors",
        "selection": "minimum pooled 5-date-group OOF risk80, AURC tie-break; identical CNN grid",
        "calibration": "positive scalar mean(error)/mean(predicted error) on separate source cal only; preserves ranking",
        "fit_ids": ids,
        "cal_ids": [rows[i]["id"] for i in ix["cal"]],
        "fit_errors": errors[:n].tolist(),
        "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds],
        "heads": {},
    }
    with threadpool_limits(limits=4):
        for block in BLOCKS:
            x, candidates = selector.features(values, block)[selected], []
            for name in GRID:
                oof = np.zeros(n)
                for tr, va in folds:
                    head = selector.estimator(name).fit(x[tr], np.log1p(errors[tr]))
                    oof[va] = selector.raw_predict(head, x[va])
                curve = selective_curve(errors[:n], oof, ids)
                candidates.append(
                    {
                        "name": name,
                        "risk80": curve["fixed"]["80"]["mean"],
                        "aurc": curve["aurc"],
                        "oof_scores": oof.tolist(),
                    }
                )
            best = min(candidates, key=lambda c: (c["risk80"], c["aurc"]))
            head = selector.estimator(best["name"]).fit(x[:n], np.log1p(errors[:n]))
            raw_cal = selector.raw_predict(head, x[n:])
            scale = max(1e-8, float(errors[n:].mean() / raw_cal.mean()))
            artifact = out / (block + ".joblib")
            joblib.dump({"model": head, "scale": scale, "block": block}, artifact)
            state["heads"][block] = {
                "selected": best["name"],
                "features": x.shape[1],
                "candidates": candidates,
                "scale": scale,
                "cal_scores": (raw_cal * scale).tolist(),
                "cal_errors": errors[n:].tolist(),
                "cal_mae": float(np.abs(raw_cal * scale - errors[n:]).mean()),
                "artifact_sha256": sha256(artifact),
            }
    check_bindings(frozen)
    if source_identity()["source_hash"] != state["source_hash"]:
        raise ValueError("Source implementation changed during fit")
    write_json(out / "selection.json", state)
    print(
        json.dumps(
            {"candidate": candidate, "heads": {b: h["selected"] for b, h in state["heads"].items()}}
        ),
        flush=True,
    )
    return state


def evaluate(out, head_lock, output, *, external=None, sony=False):
    out, head_lock, output = map(Path, (out, head_lock, output))
    state, lock = load(out / "selection.json"), load(head_lock)
    if lock.get("statistics_selectors", {}).get(state["candidate"]) != sha256(
        out / "selection.json"
    ):
        raise ValueError("Final head lock mismatch")
    if output.exists():
        raise FileExistsError(output)
    check_bindings(state["bindings"])
    if source_identity()["source_hash"] != state["source_hash"]:
        raise ValueError("Source implementation binding changed")
    run, data = Path(state["run"]), Path(state["data"])
    specs = [
        (
            "source_regression",
            data / "cube.npz",
            data / "cube_manifest.json",
            "source_predictions.npz",
        )
    ]
    if sony:
        specs.append(
            (
                "sony30_regression",
                data / "sony.npz",
                data / "sony_manifest.json",
                "sony_predictions.npz",
            )
        )
    if external:
        specs.append(
            ("fresh_all", Path(external[0]), Path(external[1]), "external_predictions.npz")
        )
    result, audit = {}, bindings([head_lock, out / "selection.json"])
    with threadpool_limits(limits=4):
        for domain, npz, manifest, filename in specs:
            before = bindings([npz, manifest])
            rows, values = prediction(run, state["candidate"], npz, manifest, out / filename)
            idx = (
                split_indices(rows, "official")["test"]
                if domain == "source_regression"
                else np.arange(len(rows))
            )
            slices = {domain: idx}
            if domain == "fresh_all":
                slices.update(
                    {
                        "fresh_" + camera: np.array(
                            [i for i, r in enumerate(rows) if r["camera"] == camera]
                        )
                        for camera in sorted({r["camera"] for r in rows})
                    }
                )
            # Only requested evaluation rows are decoded. This path never runs during fit.
            gt = statistics.read_npz_rows(npz, "gt", np.sort(idx), expected_rows=len(rows))
            for block, head in state["heads"].items():
                artifact = out / (block + ".joblib")
                check_bindings({str(artifact): head["artifact_sha256"]})
                audit[str(artifact.resolve())] = head["artifact_sha256"]
                payload = joblib.load(artifact)  # Locally produced, hash-bound artifacts only.
                scores = (
                    selector.raw_predict(payload["model"], selector.features(values, block))
                    * payload["scale"]
                )
                for name, subset in slices.items():
                    record = selector.selective_result(
                        values["pred"][subset],
                        gt[np.searchsorted(np.sort(idx), subset)],
                        scores[subset],
                        [rows[i]["id"] for i in subset],
                        np.asarray(head["cal_scores"]),
                        values["valid"][subset],
                    )
                    record.update(
                        groups=[rows[i]["group"] for i in subset],
                        cameras=[rows[i]["camera"] for i in subset],
                        mandatory_refusal_note="Invalid inputs excluded from every acceptance mask; fallback errors diagnostic only.",
                    )
                    result.setdefault(name, {})[block] = record
            check_bindings(before)
            audit.update(before)
            audit.update(bindings([out / filename, (out / filename).with_suffix(".manifest.json")]))
    check_bindings(state["bindings"])
    check_bindings(audit)
    write_json(output, {"candidate": state["candidate"], "bindings": audit, "domains": result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "evaluate"))
    parser.add_argument("--run", type=Path, default=Path("experiments/runs/ccv2_statistics_fixed"))
    parser.add_argument("--candidates", nargs="+", default=["direct_hgb7", "gw_ridge1"])
    parser.add_argument("--data", type=Path, default=Path("data/processed/cc128"))
    parser.add_argument(
        "--out-root", type=Path, default=Path("experiments/runs/ccv2_statistics_eval")
    )
    parser.add_argument(
        "--estimator-lock",
        type=Path,
        default=Path("docs/benchmarks/cc_v2/statistics_fixed_selection_lock.json"),
    )
    parser.add_argument("--head-lock", type=Path)
    parser.add_argument("--external", nargs=2)
    parser.add_argument("--sony", action="store_true")
    parser.add_argument(
        "--output-root", type=Path, default=Path("docs/benchmarks/cc_v2/statistics_evaluations")
    )
    args = parser.parse_args()
    if args.action == "evaluate" and args.head_lock is None:
        parser.error("evaluate requires --head-lock")
    for candidate in args.candidates:
        out = args.out_root / candidate
        if args.action == "fit":
            fit(args.run, candidate, args.data, out, args.estimator_lock)
        else:
            args.output_root.mkdir(parents=True, exist_ok=True)
            evaluate(
                out,
                args.head_lock,
                args.output_root / (candidate + ".json"),
                external=args.external,
                sony=args.sony,
            )


if __name__ == "__main__":
    main()
