"""Reproduce the frozen v2 audit, writing a new receipt without changing evidence."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument(
    "--receipt",
    type=Path,
    help="New receipt path, relative to repository root; existing files are refused.",
)
args = parser.parse_args()
ROOT = args.repo_root.resolve()
BASE = ROOT / "docs/benchmarks/cc_v2"
RUNS = ROOT / "experiments/runs"
receipt_path = args.receipt or BASE / (
    "independent_metric_recheck_"
    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    + ".json"
)
if not receipt_path.is_absolute():
    receipt_path = ROOT / receipt_path
if receipt_path.exists():
    raise FileExistsError("Audit receipts are append-only: " + str(receipt_path))

# Frozen manifests retain their original absolute workspace prefix. Resolve only
# that recorded prefix into this checkout; never alter the serialized evidence.
seed_config = json.loads(
    (RUNS / "ccv2_direct_large_g0_s17/config.json").read_text(encoding="utf-8")
)
recorded_root = seed_config["out"].replace("\\", "/").rsplit("/experiments/runs/", 1)[0]


def local_path(path):
    value = str(path).replace("\\", "/")
    if value == recorded_root or value.startswith(recorded_root + "/"):
        value = value[len(recorded_root) :].lstrip("/")
        return ROOT / value
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


issues, receipts, maxima = [], {}, defaultdict(float)
checks = Counter()
TOL = 1e-7


def load(path):
    return json.loads(local_path(path).read_text(encoding="utf-8"))


def sha(path):
    path = local_path(path).resolve()
    key = str(path)
    if key not in receipts:
        with path.open("rb") as f:
            receipts[key] = hashlib.file_digest(f, "sha256").hexdigest()
    return receipts[key]


def ok(condition, label):
    checks[label.split(":")[0]] += 1
    if not condition:
        issues.append(label)


def binding(path, expected):
    path = local_path(path)
    ok(path.is_file() and sha(path) == expected, "hash:" + str(path))


def diff(a, b, label, category="metric", tolerance=TOL):
    if isinstance(a, dict):
        ok(isinstance(b, dict) and set(a) <= set(b), "schema:" + label)
        for k, v in a.items():
            if k in b:
                diff(v, b[k], label + "/" + k, category, tolerance)
    elif a is None or isinstance(a, (str, bool)):
        ok(a == b, "exact:" + label)
    else:
        try:
            x, y = np.asarray(a), np.asarray(b)
            if x.dtype.kind not in "biuf" or y.dtype.kind not in "biuf":
                ok(a == b, "exact:" + label)
                return
            ok(x.shape == y.shape, "shape:" + label)
            if x.shape != y.shape:
                return
            delta = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
            maxima[category] = max(maxima[category], delta)
            ok(
                np.isfinite(x).all() and np.isfinite(y).all() and delta <= tolerance,
                "numeric:" + label + f" delta={delta:.12g}",
            )
        except (TypeError, ValueError):
            ok(a == b, "exact:" + label)


def angle(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    dot = (
        a / np.sqrt((a * a).sum(-1, keepdims=True)) * (b / np.sqrt((b * b).sum(-1, keepdims=True)))
    ).sum(-1)
    return np.arccos(np.clip(dot, -1.0, 1.0)) * (180.0 / np.pi)


def numeric_delta(a, b):
    if isinstance(a, dict):
        return max((numeric_delta(v, b[k]) for k, v in a.items()), default=0.0)
    if a is None or isinstance(a, (str, bool)):
        return 0.0
    x, y = np.asarray(a), np.asarray(b)
    return (
        float(np.max(np.abs(x.astype(float) - y.astype(float))))
        if x.size and x.dtype.kind in "biuf"
        else 0.0
    )


def errors(p, g):
    ratio = np.asarray(g, dtype=np.float64) / np.asarray(p, dtype=np.float64)
    return angle(p, g), angle(ratio, np.ones_like(ratio))


def summary(e):
    e = sorted(float(x) for x in e)
    n, q = len(e), np.percentile(e, [25, 50, 75, 90, 95])
    k = math.ceil(n / 4)
    return dict(
        n=n,
        mean=sum(e) / n,
        median=q[1],
        trimean=(q[0] + 2 * q[1] + q[2]) / 4,
        best25=sum(e[:k]) / k,
        worst25=sum(e[-k:]) / k,
        p90=q[3],
        p95=q[4],
        max=e[-1],
        over10_fraction=sum(v > 10 for v in e) / n,
        over20_fraction=sum(v > 20 for v in e) / n,
    )


def order_for(scores, ids, valid=None):
    valid = np.ones(len(ids), bool) if valid is None else np.asarray(valid, bool)
    return np.array(
        sorted(
            np.flatnonzero(valid),
            key=lambda i: (scores[i], hashlib.sha256(ids[i].encode()).digest()),
        ),
        dtype=int,
    )


lock = load(BASE / "final_head_lock.json")
lock_time = datetime.fromisoformat(lock["timestamp_utc"]).timestamp()
binding(
    BASE / "final_head_lock.json",
    "3bb2219e4ccf2b4db43842e3e6666de701988a5af9bdf92f28faa4b3b4e29184",
)
for name, h in lock["amendments"].items():
    binding(BASE / name, h)
    item = load(BASE / name)
    ok(datetime.fromisoformat(item["timestamp_utc"]).timestamp() < lock_time, "chronology:" + name)
    ok(item.get("target_errors_observed") is False, "pretarget_declaration:" + name)
for name, h in lock["script_hashes"].items():
    binding(ROOT / "scripts" / name, h)

rows = {
    "source": load(ROOT / "data/processed/cc128/cube_manifest.json"),
    "sony": load(ROOT / "data/processed/cc128/sony_manifest.json"),
    "fresh": load(ROOT / "data/processed/cc_v2_fresh128/fresh_manifest.json"),
}
cache_paths = {
    "source": ROOT / "data/processed/cc128/cube.npz",
    "sony": ROOT / "data/processed/cc128/sony.npz",
    "fresh": ROOT / "data/processed/cc_v2_fresh128/fresh.npz",
}
gt, input_valid = {}, {}
for name, path in cache_paths.items():
    with np.load(path, allow_pickle=False) as data:
        gt[name] = data["gt"]
        x = data["images"]
        ok(np.isfinite(x).all() and (x >= 0).all(), "input_validity:" + name)
        input_valid[name] = (x.mean(axis=(2, 3), dtype=np.float32) > 1e-8).all(axis=1)
        del x
    ok(len(rows[name]) == len(gt[name]), "manifest_count:" + name)
    ok(len(set(r["id"] for r in rows[name])) == len(rows[name]), "unique_ids:" + name)
binding(cache_paths["fresh"], lock["fresh_data"]["cache"])
binding(ROOT / "data/processed/cc_v2_fresh128/fresh_manifest.json", lock["fresh_data"]["manifest"])
ok(
    Counter(r["camera"] for r in rows["fresh"])
    == {"Canon_5DSR": 128, "Nikon_D810": 128, "Sony_IMX135_BLCCSC": 128},
    "fresh_camera_counts",
)
split = {
    part: [i for i, r in enumerate(rows["source"]) if r["subset"] == part]
    for part in ["train", "val", "risk", "cal", "test"]
}
split_ids = {part: [rows["source"][i]["id"] for i in ids] for part, ids in split.items()}
ok(
    {k: len(v) for k, v in split.items()}
    == {"train": 1126, "val": 119, "risk": 259, "cal": 268, "test": 462},
    "source_split_counts",
)
for ai, a in enumerate(["train", "val", "risk", "cal"]):
    for b in ["train", "val", "risk", "cal"][ai + 1 :]:
        ok(
            not (
                {rows["source"][i]["group"] for i in split[a]}
                & {rows["source"][i]["group"] for i in split[b]}
            ),
            "source_group_separation:" + a + "/" + b,
        )
domain_specs = {
    "source_regression": ("source", split["test"]),
    "sony30_regression": ("sony", list(range(30))),
    "fresh_all": ("fresh", list(range(384))),
}
for camera in sorted(set(r["camera"] for r in rows["fresh"])):
    domain_specs["fresh_" + camera] = (
        "fresh",
        [i for i, r in enumerate(rows["fresh"]) if r["camera"] == camera],
    )

source_paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "uv.lock"]
h = hashlib.sha256()
for p in source_paths:
    h.update(p.relative_to(ROOT).as_posix().encode())
    h.update(p.read_bytes())
ok(h.hexdigest() == lock["source_hash"], "live_source_hash")
registered = json.loads(
    subprocess.check_output(
        ["git", "show", "7637d6d:docs/benchmarks/public_evidence_manifest.json"], cwd=ROOT
    )
)
registered = {r["run"]: r for r in registered}

states, outputs, result_counts, legacy_deltas = {}, {}, Counter(), {}
expected_names = (
    set(lock["cnn_selectors"]) | set(lock["legacy_selectors"]) | set(lock["statistics_selectors"])
)
found = {p.stem for p in (BASE / "runs").glob("*.json")}
ok(found == expected_names and len(found) == 25, "complete_frozen_run_inventory")
chronology = {
    "final_lock_utc": lock["timestamp_utc"],
    "first_output_mtime_utc": None,
    "last_selector_mtime_utc": None,
}
output_times, selector_times = [], []
oof_checked = 0
legacy_comparisons = 0
for name in sorted(found):
    output_path = BASE / "runs" / f"{name}.json"
    r = outputs[name] = load(output_path)
    sha(output_path)
    output_times.append(output_path.stat().st_mtime)
    ok(output_path.stat().st_mtime > lock_time, "chronology:output_" + name)
    if name in lock["cnn_selectors"]:
        kind, run = "cnn", RUNS / name
        sp = run / "risk_v2/selection.json"
        config = load(run / "config.json")
        diff(config, r["config"], name + "/config", "binding", 0)
        ok(config["all_split_ids"] == split_ids, "cnn_splits:" + name)
        ok(config["source_hash"] == lock["source_hash"], "cnn_source:" + name)
        for filename, digest in config["data_hashes"].items():
            binding(ROOT / "data/processed/cc128" / filename, digest)
        snapshot = config["source_snapshot"]
        hh = hashlib.sha256()
        for p in source_paths:
            rel = p.relative_to(ROOT).as_posix()
            saved = run / snapshot["directory"] / rel
            binding(saved, snapshot["files"][rel])
            binding(p, snapshot["files"][rel])
            hh.update(rel.encode())
            hh.update(saved.read_bytes())
        ok(hh.hexdigest() == lock["source_hash"], "snapshot_hash:" + name)
        pm = load(run / "predictions_manifest.json")
        checkpoint = load(run / "checkpoint_manifest.json")
        for meta in [pm, checkpoint]:
            binding(run / "model.pt", meta["checkpoint_sha256"])
            binding(run / "config.json", meta["config_sha256"])
        ok(
            pm["training_source_hash"]
            == pm["prediction_source_identity"]["source_hash"]
            == lock["source_hash"],
            "prediction_source:" + name,
        )
        training = load(run / "training.json")
        ok(training.get("test_or_external_errors_computed") is False, "training_scope:" + name)
        best_epoch = min(training["history"], key=lambda v: v["val_reproduction"])
        diff(
            best_epoch["val_reproduction"],
            training["best_validation"],
            name + "/checkpoint_selection",
            "validation_selection",
            0,
        )
        ok(best_epoch["epoch"] == training["checkpoint_epoch"], "checkpoint_best_epoch:" + name)
        for filename, item in pm["outputs"].items():
            binding(run / filename, item["sha256"])
            binding(Path(item["input_npz"]), item["input_hashes"]["npz"])
            binding(Path(item["input_manifest"]), item["input_hashes"]["manifest"])
        state = states[name] = load(sp)
        binding(run / "predictions_manifest.json", state["predictions_manifest_sha256"])
        binding(run / "predictions.npz", state["prediction_sha256"])
        binding(run / "model.pt", state["checkpoint_sha256"])
        binding(ROOT / "scripts/cc_v2_select.py", state["script_sha256"])
        ok(r["evaluation_source"] == state["script_sha256"], "evaluation_source:" + name)
        predpaths = {
            d: run / f
            for d, f in [
                ("source", "predictions.npz"),
                ("sony", "sony_predictions.npz"),
                ("fresh", "external_predictions.npz"),
            ]
        }
    elif name in lock["legacy_selectors"]:
        kind, run = "legacy", local_path(r["legacy_run"])
        sp = RUNS / name / "selection.json"
        state = states[name] = load(sp)
        config = load(run / "config.json")
        ok(config["all_split_ids"] == split_ids, "legacy_splits:" + name)
        binding(run / "model.pt", registered[run.name]["model_sha256"])
        for fname, field in [
            ("model.pt", "checkpoint_sha256"),
            ("config.json", "legacy_config_sha256"),
            ("risk_heads.json", "legacy_risk_heads_sha256"),
        ]:
            binding(run / fname, state[field])
        for fname in ["config.json", "risk_heads.json", "evaluation.json"]:
            preserved = ROOT / "docs/benchmarks/public_runs" / run.name / fname
            original = subprocess.check_output(
                ["git", "show", f"7637d6d:docs/benchmarks/public_runs/{run.name}/{fname}"], cwd=ROOT
            )
            ok(
                preserved.read_bytes().replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"),
                "immutable_legacy:" + name + "/" + fname,
            )
            if fname != "evaluation.json":
                ok(
                    (run / fname).read_bytes().replace(b"\r\n", b"\n")
                    == original.replace(b"\r\n", b"\n"),
                    "legacy_artifact:" + name + "/" + fname,
                )
        for fname, digest in state["source_hashes"].items():
            binding(ROOT / "data/processed/cc128" / fname, digest)
        binding(sp.parent / "heads.joblib", state["artifact_sha256"])
        binding(ROOT / "scripts/cc_v2_legacy.py", state["script_sha256"])
        binding(ROOT / "scripts/cc_v2_select.py", state["helper_sha256"])
        ok(r["script_sha256"] == state["script_sha256"], "legacy_eval_script:" + name)
        ok(r["source_identity"]["source_hash"] == lock["source_hash"], "legacy_eval_source:" + name)
        for domain, cn in [("sony30_regression", "sony"), ("fresh_all", "fresh")]:
            binding(cache_paths[cn], r["input_hashes"][domain]["cache"])
        predpaths = {}
    else:
        kind = "statistics"
        sp = RUNS / "ccv2_statistics_eval" / name / "selection.json"
        state = states[name] = load(sp)
        for collection in [state["bindings"], r["bindings"]]:
            for filename, digest in collection.items():
                binding(filename, digest)
        ok(state["source_hash"] == lock["source_hash"], "statistics_source:" + name)
        predpaths = {
            d: sp.parent / f
            for d, f in [
                ("source", "source_predictions.npz"),
                ("sony", "sony_predictions.npz"),
                ("fresh", "external_predictions.npz"),
            ]
        }
        screen = load(local_path(state["run"]) / "screen.json")
        ok(
            screen["source_ids"] == {k: split_ids[k] for k in ["train", "val"]},
            "statistics_fitting_partition:" + name,
        )
        for candidate in screen["candidates"]:
            _, ve = errors(candidate["val_predictions"], gt["source"][split["val"]])
            diff(
                summary(ve),
                candidate["val_reproduction"],
                name + "/" + candidate["candidate"] + "/validation",
                "statistics_validation",
            )
        for mode, selected in screen["selected_by_mode"].items():
            ok(
                selected
                == min(
                    (v for v in screen["candidates"] if v["mode"] == mode),
                    key=lambda v: v["val_mean_reproduction"],
                )["candidate"],
                "statistics_mode_selection:" + name + "/" + mode,
            )
        for domain, predpath in predpaths.items():
            pm = load(predpath.with_suffix(".manifest.json"))
            binding(predpath, pm["prediction_sha256"])
            binding(local_path(state["run"]) / "screen.json", pm["screen_sha256"])
            binding(local_path(state["run"]) / (name + ".joblib"), pm["model_sha256"])
            binding(ROOT / "scripts/cc_v2_statistics.py", pm["script_sha256"])
            binding(Path(pm["input_npz"]), pm["input_hashes"]["npz"])
            binding(Path(pm["input_manifest"]), pm["input_hashes"]["manifest"])
            ok(
                pm["candidate"] == name and pm["gt_read_or_errors_computed"] is False,
                "statistics_prediction_scope:" + name + "/" + domain,
            )
    binding(sp, lock[kind + "_selectors" if kind != "cnn" else "cnn_selectors"][name])
    if kind != "statistics":
        ok(r["selector_sha256"] == sha(sp), "evaluation_selector:" + name)
    selector_times.append(sp.stat().st_mtime)
    ok(sp.stat().st_mtime < lock_time, "chronology:selector_" + name)
    if "timestamp_utc" in state:
        ok(
            datetime.fromisoformat(state["timestamp_utc"]).timestamp() < lock_time,
            "chronology:selector_timestamp_" + name,
        )
    ok(
        state["fit_ids"] == split_ids["risk"] and state["cal_ids"] == split_ids["cal"],
        "selector_source_ids:" + name,
    )
    if "folds" in state:
        seen = []
        groups = [rows["source"][i]["group"] for i in split["risk"]]
        for fold in state["folds"]:
            tr, va = fold["train"], fold["validation"]
            ok(
                not (set(tr) & set(va)) and set(tr) | set(va) == set(range(259)),
                "oof_partition:" + name,
            )
            ok(
                not ({groups[i] for i in tr} & {groups[i] for i in va}),
                "oof_group_separation:" + name,
            )
            seen.extend(va)
        ok(sorted(seen) == list(range(259)), "oof_exactly_once:" + name)
    saved_predictions = {}
    for domain, path in predpaths.items():
        with np.load(path, allow_pickle=False) as v:
            saved_predictions[domain] = {k: v[k] for k in ["pred", "valid", "ids"]}
        ok(
            saved_predictions[domain]["ids"].tolist() == [a["id"] for a in rows[domain]],
            "prediction_ids:" + name + "/" + domain,
        )
        diff(
            saved_predictions[domain]["valid"],
            input_valid[domain],
            name + "/" + domain + "/validity",
            "validity",
            0,
        )
    fit_err = None
    if "source" in saved_predictions:
        _, fit_err = errors(
            saved_predictions["source"]["pred"][split["risk"]], gt["source"][split["risk"]]
        )
    if "fit_errors" in state:
        diff(fit_err, state["fit_errors"], name + "/fit_errors", "source_oof")
    for block, head in state["heads"].items():
        selected = min(head["candidates"], key=lambda v: (v["risk80"], v["aurc"]))["name"]
        ok(head["selected"] == selected, "source_head_selection:" + name + "/" + block)
        ok(
            head["scale"] > 0 and len(head["cal_scores"]) == 268,
            "source_calibration:" + name + "/" + block,
        )
        if "artifact_sha256" in head:
            binding(sp.parent / (block + ".joblib"), head["artifact_sha256"])
        if fit_err is not None:
            for candidate in head["candidates"]:
                order = order_for(candidate["oof_scores"], state["fit_ids"])
                ranked = fit_err[order]
                diff(
                    float(ranked[: math.floor(259 * 0.8)].mean()),
                    candidate["risk80"],
                    name + "/" + block + "/" + candidate["name"] + "/risk80",
                    "source_oof",
                )
                diff(
                    float(np.mean(np.cumsum(ranked) / np.arange(1, 260))),
                    candidate["aurc"],
                    name + "/" + block + "/" + candidate["name"] + "/aurc",
                    "source_oof",
                )
                oof_checked += 1
        if "cal_errors" in head:
            _, ce = errors(
                saved_predictions["source"]["pred"][split["cal"]], gt["source"][split["cal"]]
            )
            diff(ce, head["cal_errors"], name + "/" + block + "/cal_errors", "source_calibration")
            diff(
                float(np.mean(head["cal_scores"])),
                float(ce.mean()),
                name + "/" + block + "/cal_mean",
                "source_calibration",
            )
            diff(
                float(np.mean(np.abs(np.asarray(head["cal_scores"]) - ce))),
                head["cal_mae"],
                name + "/" + block + "/cal_mae",
                "source_calibration",
            )
    ok(set(r["domains"]) == set(domain_specs), "domain_inventory:" + name)
    old = (
        load(ROOT / "docs/benchmarks/public_runs" / run.name / "evaluation.json")
        if kind == "legacy"
        else None
    )
    per_run_legacy = defaultdict(float)
    for domain, methods in r["domains"].items():
        cn, ix = domain_specs[domain]
        for method, rec in methods.items():
            label = name + "/" + domain + "/" + method
            result_counts[kind] += 1
            for field, rowfield in [("ids", "id"), ("cameras", "camera"), ("groups", "group")]:
                ok(
                    rec[field] == [rows[cn][i][rowfield] for i in ix],
                    "manifest_membership:" + label + "/" + field,
                )
            diff(gt[cn][ix], rec["gt"], label + "/gt", "ground_truth", 0)
            diff(input_valid[cn][ix], rec["valid"], label + "/valid", "validity", 0)
            if cn in saved_predictions:
                diff(
                    saved_predictions[cn]["pred"][ix],
                    rec["pred"],
                    label + "/saved_pred",
                    "prediction_binding",
                    0,
                )
            p, g, s = np.asarray(rec["pred"]), np.asarray(rec["gt"]), np.asarray(rec["scores"])
            ok(
                np.isfinite(p).all()
                and (p > 0).all()
                and np.isfinite(g).all()
                and (g > 0).all()
                and np.isfinite(s).all(),
                "finite_positive:" + label,
            )
            recovery, rep = errors(p, g)
            diff(summary(recovery), rec["recovery"], label + "/recovery", "recovery")
            diff(summary(rep), rec["reproduction"], label + "/reproduction", "reproduction")
            diff(rep, rec["errors"], label + "/errors", "reproduction")
            valid = np.asarray(rec["valid"], bool)
            n = len(valid)
            order = order_for(s, rec["ids"], valid)
            diff(int((~valid).sum()), rec["invalid_n"], label + "/invalid_n", "validity", 0)
            diff(
                float(valid.mean()),
                rec["max_supported_coverage"],
                label + "/max_supported",
                "validity",
                0,
            )
            expected_risk = np.cumsum(rep[order]) / np.arange(1, len(order) + 1)
            diff(order, rec["selective"]["order"], label + "/order", "order", 0)
            diff(
                np.arange(1, len(order) + 1) / n,
                rec["selective"]["coverage"],
                label + "/coverage",
                "coverage",
                1e-15,
            )
            diff(expected_risk, rec["selective"]["risk"], label + "/risk", "selective")
            diff(
                float(expected_risk.mean()) if len(order) else None,
                rec["selective"]["aurc"],
                label + "/aurc",
                "aurc",
            )
            for percent in [100, 95, 90, 80, 70, 60]:
                k = math.floor(n * percent / 100)
                item = rec["selective"]["fixed"][str(percent)]
                if k <= len(order):
                    expect = {**summary(rep[order[:k]]), "coverage": k / n}
                    if not valid.all():
                        expect["attainable"] = True
                else:
                    expect = {
                        "mean": None,
                        "n": len(order),
                        "coverage": len(order) / n,
                        "attainable": False,
                    }
                diff(expect, item, label + "/fixed/" + str(percent), "fixed")
                threshold = rec["frozen_source_thresholds"][str(percent)]["threshold"]
                if kind != "legacy" or method.startswith("upgraded_"):
                    block = method.removeprefix("upgraded_")
                    expected_threshold = (
                        None
                        if percent == 100
                        else float(np.quantile(state["heads"][block]["cal_scores"], percent / 100))
                    )
                    diff(
                        expected_threshold,
                        threshold,
                        label + "/threshold/" + str(percent),
                        "threshold",
                        1e-12,
                    )
                accepted = valid & (True if threshold is None else s <= threshold)
                expect = {
                    "threshold": threshold,
                    "n": int(accepted.sum()),
                    "coverage": float(accepted.mean()),
                    "mean_reproduction": float(rep[accepted].mean()) if accepted.any() else None,
                }
                diff(
                    expect,
                    rec["frozen_source_thresholds"][str(percent)],
                    label + "/source_mask/" + str(percent),
                    "source_mask",
                )
            if (
                old
                and domain in ["source_regression", "sony30_regression"]
                and method.startswith("v1_")
            ):
                legacy_comparisons += 1
                old_domain = "source" if domain == "source_regression" else "sony_pilot"
                original = old[old_domain][method[3:]]
                for field in [
                    "pred",
                    "gt",
                    "scores",
                    "errors",
                    "recovery",
                    "reproduction",
                    "selective",
                    "frozen_source_thresholds",
                    "ids",
                ]:
                    tolerance = 1e-5 if field in ["scores", "frozen_source_thresholds"] else 1e-7
                    diff(
                        original[field],
                        rec[field],
                        label + "/v1/" + field,
                        "legacy_" + field,
                        tolerance,
                    )
                    per_run_legacy[field] = max(
                        per_run_legacy[field], numeric_delta(original[field], rec[field])
                    )
    if old:
        legacy_deltas[name] = dict(per_run_legacy)
    print("audited", name, flush=True)

for family, item in lock["primary_source_selected"].items():
    for block, expected in item["source_oof_risk80"].items():
        actual = []
        for seed in lock["seeds"]:
            head = states[f"{family}_s{seed}"]["heads"][block]
            actual.append(
                next(c["risk80"] for c in head["candidates"] if c["name"] == head["selected"])
            )
        diff(actual, expected, family + "/" + block + "/locked_oof", "primary_selection", 0)
        diff(
            float(np.mean(actual)),
            item["source_oof_means"][block],
            family + "/" + block + "/mean",
            "primary_selection",
            0,
        )
    ok(
        item["selected_block"] == min(item["source_oof_means"], key=item["source_oof_means"].get),
        "primary_block_selection:" + family,
    )
ok(lock["seeds"] == [17, 29, 43] and lock["representative_export_seed"] == 17, "prespecified_seeds")

# Check reporting aggregation separately from evaluator routines.
aggregate = load(BASE / "aggregate.json")
binding(ROOT / "scripts/cc_v2_report.py", aggregate["script_sha256"])
for fname, digest in aggregate["inputs"].items():
    binding(BASE / "runs" / fname, digest)
grouped = defaultdict(list)
for name, r in sorted(outputs.items()):
    family = re.sub(r"_s\d+$", "", name)
    for domain, methods in r["domains"].items():
        for method, rec in methods.items():
            grouped[(domain, family + "::" + method)].append((name + ".json", rec))
for (domain, method), items in grouped.items():
    a = aggregate["domains"][domain][method]
    ok(
        a["source_files"] == [x[0] for x in items] and a["runs"] == len(items),
        "aggregate_membership:" + domain + "/" + method,
    )
    paths = {
        "mean": ["reproduction", "mean"],
        "median": ["reproduction", "median"],
        "p95": ["reproduction", "p95"],
        "aurc": ["selective", "aurc"],
        "source80_actual_coverage": ["frozen_source_thresholds", "80", "coverage"],
        "source80_error": ["frozen_source_thresholds", "80", "mean_reproduction"],
    }
    paths.update(
        {"risk" + str(p): ["selective", "fixed", str(p), "mean"] for p in [100, 95, 90, 80, 70, 60]}
    )
    for metric, path in paths.items():
        values = []
        for _, rec in items:
            v = rec
            for key in path:
                v = v[key]
            values.append(v)
        expected = {
            "values": values,
            "mean": float(np.mean(values)),
            "std_across_runs": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
        diff(expected, a[metric], domain + "/" + method + "/aggregate/" + metric, "aggregate")
chronology["first_output_mtime_utc"] = datetime.fromtimestamp(
    min(output_times), timezone.utc
).isoformat()
chronology["last_selector_mtime_utc"] = datetime.fromtimestamp(
    max(selector_times), timezone.utc
).isoformat()
summary_out = {
    "status": "PASS_WITH_SCOPE_CAVEATS" if not issues else "NEEDS_REVISION",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "Independent read-only recomputation from saved predictions and GT; no fitting, tuning, target inference or source imports.",
    "run_count": len(found),
    "domain_method_records": dict(result_counts),
    "check_counts": dict(checks),
    "maximum_absolute_deltas": dict(maxima),
    "legacy_regression_comparison_count": legacy_comparisons,
    "legacy_per_run_maxima": legacy_deltas,
    "legacy_numerical_tolerance": {
        "scores_and_source_thresholds": 1e-5,
        "predictions_errors_and_metrics": 1e-7,
        "reason": "Observed context-only differences are consistent with persisted center/scale loading as JSON float64 after original float32 feature normalization. They do not change any ranking, acceptance count or angular metric.",
    },
    "source_oof_candidate_metrics_recomputed": oof_checked,
    "invalid_rows_by_base_domain": {k: int((~v).sum()) for k, v in input_valid.items()},
    "fresh_camera_counts": dict(Counter(r["camera"] for r in rows["fresh"])),
    "source_split_counts": {k: len(v) for k, v in split.items()},
    "chronology": chronology,
    "issues": issues,
    "limitations": [
        "Local timestamps and frozen hash chains support recorded pretarget ordering; they cannot prove absence of unrecorded earlier computation.",
        "Legacy upgrade fit-time neural outputs/errors were not persisted; candidate selection rule and source IDs are checked, but their OOF numeric errors cannot be independently recomputed without rerunning inference, which this audit does not do.",
        "All measured rows are valid. Invalid-refusal branches receive no real invalid-row coverage here; their behavior is covered by the earlier synthetic engineering review.",
        "Official source test is a regression split with inherited capture-date overlap; Sony30 is already observed regression evidence.",
        "Fresh384 is a custom camera-balanced mirror subset; actual scene clusters and original-mirror file identity are unverified. No full INTEL-TAU benchmark, skin DeltaE, commercial rights or novelty conclusion follows.",
    ],
    "sha256_receipts": {
        str(Path(k).relative_to(ROOT)): v
        for k, v in receipts.items()
        if Path(k).is_relative_to(ROOT)
    },
    "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
original_path = BASE / "independent_metric_review.json"
original = load(original_path)
reproduced_fields = [
    "status",
    "run_count",
    "domain_method_records",
    "check_counts",
    "maximum_absolute_deltas",
    "legacy_regression_comparison_count",
    "legacy_per_run_maxima",
    "legacy_numerical_tolerance",
    "source_oof_candidate_metrics_recomputed",
    "invalid_rows_by_base_domain",
    "fresh_camera_counts",
    "source_split_counts",
    "chronology",
    "issues",
]
mismatched_fields = [field for field in reproduced_fields if summary_out[field] != original[field]]
summary_out["original_review_comparison"] = {
    "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
    "fields": reproduced_fields,
    "exact_match": not mismatched_fields,
    "mismatched_fields": mismatched_fields,
}
if mismatched_fields:
    summary_out["status"] = "NEEDS_REVISION"
receipt_path.parent.mkdir(parents=True, exist_ok=True)
with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(summary_out, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: v
            for k, v in summary_out.items()
            if k not in ["sha256_receipts", "legacy_per_run_maxima", "limitations"]
        },
        indent=2,
    ),
    flush=True,
)
raise SystemExit(1 if issues or mismatched_fields else 0)
