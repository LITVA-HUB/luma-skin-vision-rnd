"""Secondary frozen-prefix compression study over completed local-search models."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_local_search_core import rbf_features, ridge_solve
from skin_local_search_train import (
    CACHE_HASH,
    GRIDS,
    SEEDS,
    load_model,
    metrics,
    sha,
    synchronize,
    weights_for,
    write_json,
)

FAMILIES = ("random_rbf", "guided_rbf")
PREFIX_COUNTS = (8, 16, 32)
REFERENCE_COUNT = 64
ALL_COUNTS = (*PREFIX_COUNTS, REFERENCE_COUNT)
PROTOCOL = ROOT / "docs/research/skin_local_search_compact_protocol.md"
DEFAULT_SOURCE_RUN = ROOT / "experiments/runs/skin_local_search_v1"
DEFAULT_RUN = ROOT / "experiments/runs/skin_local_search_compact_v1"
FIT_SCRIPT_SHA256 = "f24dfe35370b7c871fcced3540c7b366167b6c85613c34cdac842042fbad3c86"
FIT_PROTOCOL_SHA256 = "950ca0e35380d6484924b8f860d16710d023967e54f8b9d0b1f789f737620e9c"


def _method(model):
    value = np.asarray(model["method"])
    if value.ndim != 0:
        raise ValueError("method must be a scalar")
    return str(value.item())


def _validate_source_model(model, feature_count):
    method = _method(model)
    if method not in FAMILIES:
        raise ValueError("source model must be an RBF family")
    required = {"x_mean", "x_std", "y_mean", "y_std", "centers", "widths", "beta"}
    if not required.issubset(model):
        raise ValueError("source RBF payload is incomplete")
    centers = np.asarray(model["centers"])
    widths = np.asarray(model["widths"])
    if centers.ndim != 2 or centers.shape[1] != feature_count or widths.shape != (len(centers),):
        raise ValueError("source centers or widths have inconsistent shape")
    if len(centers) != REFERENCE_COUNT:
        raise ValueError("source model must contain the frozen 64-atom order")
    for key, shape in (("x_mean", (feature_count,)), ("x_std", (feature_count,)),
                       ("y_mean", (3,)), ("y_std", (3,))):
        value = np.asarray(model[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {key}")
    if not np.isfinite(centers).all() or not np.isfinite(widths).all() or np.any(widths <= 0):
        raise ValueError("source RBF atoms must be finite with positive widths")
    if np.any(np.asarray(model["x_std"]) <= 0) or np.any(np.asarray(model["y_std"]) <= 0):
        raise ValueError("source scales must be positive")
    return method


def prefix_predict(model, x):
    """Independent NumPy inference for a compact RBF-prefix payload."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("x must be a matrix")
    method = _method(model)
    if method not in FAMILIES:
        raise ValueError("model must be an RBF family")
    mean = np.asarray(model["x_mean"], dtype=np.float32)
    scale = np.asarray(model["x_std"], dtype=np.float32)
    centers = np.asarray(model["centers"], dtype=np.float32)
    widths = np.asarray(model["widths"], dtype=np.float32)
    beta = np.asarray(model["beta"], dtype=np.float32)
    if values.shape[1] != len(mean) or scale.shape != mean.shape or centers.shape[1:] != mean.shape:
        raise ValueError("feature payload shape mismatch")
    if widths.shape != (len(centers),) or beta.shape != (1 + len(mean) + len(centers), 3):
        raise ValueError("beta or atom payload shape mismatch")
    normalized = (values - mean) / scale
    distance = np.maximum(
        (normalized * normalized).sum(1)[:, None]
        + (centers * centers).sum(1)[None, :]
        - 2 * normalized @ centers.T,
        0,
    ) / normalized.shape[1]
    features = np.exp(-0.5 * distance / widths[None, :] ** 2)
    standardized = np.column_stack(
        (np.ones(len(normalized), dtype=np.float32), normalized, features)
    ) @ beta
    return standardized * np.asarray(model["y_std"], dtype=np.float32) + np.asarray(
        model["y_mean"], dtype=np.float32
    )


def refit_prefix(source_model, x, y, person, site, alpha, count, device="cpu"):
    """Refit only the ridge head over the first K atoms of a frozen 64-atom model."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    person, site = np.asarray(person), np.asarray(site)
    if x.ndim != 2 or y.shape != (len(x), 3) or person.shape != (len(x),) or site.shape != (len(x),):
        raise ValueError("fit arrays have inconsistent shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("fit arrays must be finite")
    method = _validate_source_model(source_model, x.shape[1])
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0 or count >= REFERENCE_COUNT:
        raise ValueError("count must be an integer from 1 through 63")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")

    synchronize(device)
    started = time.perf_counter()
    mean = np.asarray(source_model["x_mean"], dtype=np.float64)
    scale = np.asarray(source_model["x_std"], dtype=np.float64)
    target_mean = np.asarray(source_model["y_mean"], dtype=np.float64)
    target_scale = np.asarray(source_model["y_std"], dtype=np.float64)
    normalized = (x - mean) / scale
    normalized_target = (y - target_mean) / target_scale
    centers = np.asarray(source_model["centers"][:count], dtype=np.float64)
    widths = np.asarray(source_model["widths"][:count], dtype=np.float64)
    weights = weights_for(person, site)
    xt = torch.as_tensor(normalized, dtype=torch.float64, device=device)
    target = torch.as_tensor(normalized_target, dtype=torch.float64, device=device)
    weight = torch.as_tensor(weights, dtype=torch.float64, device=device)
    center_tensor = torch.as_tensor(centers, dtype=torch.float64, device=device)
    width_tensor = torch.as_tensor(widths, dtype=torch.float64, device=device)
    with torch.no_grad():
        features = rbf_features(xt, center_tensor, width_tensor)
        design = torch.cat(
            (torch.ones((len(xt), 1), dtype=torch.float64, device=device), xt, features), dim=1
        )
        beta = ridge_solve(design, target, alpha, weight).cpu().numpy()
    synchronize(device)
    seconds = time.perf_counter() - started
    model = {
        "method": np.asarray(method),
        "x_mean": np.asarray(source_model["x_mean"], dtype=np.float32).copy(),
        "x_std": np.asarray(source_model["x_std"], dtype=np.float32).copy(),
        "y_mean": np.asarray(source_model["y_mean"], dtype=np.float32).copy(),
        "y_std": np.asarray(source_model["y_std"], dtype=np.float32).copy(),
        "centers": np.asarray(source_model["centers"][:count], dtype=np.float32).copy(),
        "widths": np.asarray(source_model["widths"][:count], dtype=np.float32).copy(),
        "beta": beta.astype(np.float32),
    }
    prediction = prefix_predict(model, x)
    normalized_error = (prediction.astype(np.float64) - y) / target_scale
    receipt = {
        "prefix_atoms": count,
        "source_atoms": len(source_model["centers"]),
        "prefix_refit_seconds": seconds,
        "weighted_train_normalized_sse": float(
            np.sum(weights[:, None] * normalized_error**2)
        ),
        "numeric_scalars": sum(value.size for value in model.values() if value.dtype.kind == "f"),
        "numeric_bytes": sum(value.nbytes for value in model.values() if value.dtype.kind == "f"),
    }
    return model, receipt


def _verified_model(path):
    receipt_path = path.with_suffix(".json")
    if not path.exists() or not receipt_path.exists():
        raise ValueError(f"missing frozen source model: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha(path) != receipt["artifact_sha256"]:
        raise ValueError(f"changed frozen source model: {path}")
    return load_model(path), receipt


def _relative(path):
    return str(path.relative_to(ROOT))


def _load_roles(source_run, protocol):
    path = source_run / protocol / "roles.npz"
    with np.load(path, allow_pickle=False) as archive:
        return archive["fit"], archive["held"], archive["folds"]


def _fixed_config(alpha, family):
    values = list(GRIDS[family])
    matches = [index for index, value in enumerate(values) if float(value) == float(alpha)]
    if len(matches) != 1:
        raise ValueError("frozen alpha is absent or ambiguous in the v1 grid")
    return matches[0]


def _lock_sources(cache, source_run, run, device):
    required = [source_run / "fit_completion.json", source_run / "frozen_selections.json",
                source_run / "source_lock.json", PROTOCOL, Path(__file__),
                ROOT / "scripts/skin_local_search_train.py", ROOT / "scripts/skin_local_search_core.py"]
    if any(not path.exists() for path in required):
        raise ValueError("compact study requires complete v1 fit artifacts and protocol")
    if sha(cache) != CACHE_HASH:
        raise ValueError("unauthorized cache or changed data")
    source_lock = json.loads((source_run / "source_lock.json").read_text(encoding="utf-8"))
    lock = {
        "status": "secondary prefix study conceived after v1 inner results and before outer evaluation",
        "cache_sha256": CACHE_HASH,
        "source_fit_completion_sha256": sha(source_run / "fit_completion.json"),
        "source_frozen_selections_sha256": sha(source_run / "frozen_selections.json"),
        "source_lock_sha256": sha(source_run / "source_lock.json"),
        "source_code_hashes": source_lock["source_hashes"],
        "compact_hashes": {_relative(path): sha(path) for path in required[3:]},
        "families": list(FAMILIES),
        "prefix_counts": list(PREFIX_COUNTS),
        "reference_count": REFERENCE_COUNT,
        "alpha_policy": "fixed v1-selected alpha; no new alpha grid",
        "device": device,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    destination = run / "source_lock.json"
    normalized = json.loads(json.dumps(lock))
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != normalized:
            script_key = _relative(Path(__file__))
            protocol_key = _relative(PROTOCOL)
            comparison = json.loads(json.dumps(normalized))
            comparison["compact_hashes"][script_key] = FIT_SCRIPT_SHA256
            comparison["compact_hashes"][protocol_key] = FIT_PROTOCOL_SHA256
            can_amend = (
                existing == comparison
                and existing["compact_hashes"].get(script_key) == FIT_SCRIPT_SHA256
                and existing["compact_hashes"].get(protocol_key) == FIT_PROTOCOL_SHA256
                and (run / "fit_completion.json").exists()
                and (run / "frozen_prefixes.json").exists()
                and not (run / "evaluation.json").exists()
                and not (run / "evaluation_status.json").exists()
            )
            if not can_amend:
                raise ValueError("compact source binding changed: use a new run directory")
            archive = run / "source_lock_fit_archive.json"
            if archive.exists() and json.loads(archive.read_text(encoding="utf-8")) != existing:
                raise ValueError("fit-time source-lock archive mismatch")
            if not archive.exists():
                write_json(archive, existing)
            old_lock_sha = sha(destination)
            frozen_sha = sha(run / "frozen_prefixes.json")
            write_json(destination, normalized)
            write_json(run / "source_lock_amendment.json", {
                "reason": "evaluation bookkeeping now writes a separate status and preserves frozen_prefixes.json byte-for-byte",
                "scope": "source-only bookkeeping change after compact fit and before compact outer evaluation",
                "old_source_lock_sha256": old_lock_sha,
                "new_source_lock_sha256": sha(destination),
                "old_compact_hashes": existing["compact_hashes"],
                "new_compact_hashes": normalized["compact_hashes"],
                "frozen_prefixes_sha256": frozen_sha,
                "compact_models_or_parameters_changed": False,
                "compact_outer_evaluation_was_opened": False,
                "primary_outer_status": "already evaluated independently; no primary outer arrays were read by this amendment",
            })
    else:
        write_json(destination, normalized)
    return normalized


def _save_prefix(path, model, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **model)
    record = {**record, "artifact_sha256": sha(path), "serialized_bytes": path.stat().st_size}
    write_json(path.with_suffix(".json"), record)
    return record


def fit_phase(data, source_run, run, device):
    frozen_path = source_run / "frozen_selections.json"
    source_choices = json.loads(frozen_path.read_text(encoding="utf-8"))
    all_records = []
    aggregate = {"scope": "secondary adaptive prefix compression; original TRAIN only",
                 "outer_evaluation_run": False, "protocols": {}}
    final_references = {}
    unique_source_costs = {}
    total_refit_seconds = 0.0
    x_all, y_all = data["color"].astype(np.float64), data["target"].astype(np.float64)
    person_all, site_all = data["patient"], data["site"]

    for protocol in source_choices:
        fit, _, folds = _load_roles(source_run, protocol)
        x, y, person, site = x_all[fit], y_all[fit], person_all[fit], site_all[fit]
        protocol_result = {}
        final_references[protocol] = {}
        for family in FAMILIES:
            choice = source_choices[protocol][family]
            alpha = float(choice["parameter"])
            config = _fixed_config(alpha, family)
            oof = {count: {seed: np.empty_like(y) for seed in SEEDS} for count in ALL_COUNTS}
            for seed in SEEDS:
                for fold in range(3):
                    source_path = source_run / protocol / "inner" / f"{family}_c{config}_s{seed}_f{fold}.npz"
                    source_model, source_record = _verified_model(source_path)
                    if (source_record["method"] != family or float(source_record["parameter"]) != alpha
                            or source_record["seed"] != seed or source_record["fold"] != fold):
                        raise ValueError("source inner model does not match frozen choice")
                    source_key = sha(source_path)
                    unique_source_costs[_relative(source_path)] = float(source_record["fit_seconds"])
                    train, valid = folds != fold, folds == fold
                    oof[REFERENCE_COUNT][seed][valid] = prefix_predict(source_model, x[valid])
                    for count in PREFIX_COUNTS:
                        destination = run / protocol / "inner" / f"{family}_k{count}_s{seed}_f{fold}.npz"
                        if destination.exists() and destination.with_suffix(".json").exists():
                            record = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
                            if sha(destination) != record["artifact_sha256"]:
                                raise ValueError("changed compact inner artifact")
                            model = load_model(destination)
                        else:
                            model, receipt = refit_prefix(
                                source_model, x[train], y[train], person[train], site[train], alpha,
                                count, device
                            )
                            total_refit_seconds += receipt["prefix_refit_seconds"]
                            prediction = prefix_predict(model, x[valid])
                            record = _save_prefix(destination, model, {
                                "protocol": protocol, "family": family, "alpha": alpha,
                                "prefix_atoms": count, "seed": seed, "fold": fold, **receipt,
                                "source_64_path": _relative(source_path),
                                "source_64_sha256": source_key,
                                "source_64_search_seconds": source_record["fit_seconds"],
                                "source_plus_prefix_seconds_charged": source_record["fit_seconds"]
                                + receipt["prefix_refit_seconds"],
                                "inner_metrics": metrics(
                                    prediction, y[valid], person[valid], site[valid]
                                ),
                            })
                        oof[count][seed][valid] = prefix_predict(model, x[valid])
                        all_records.append(record)

            count_scores = {}
            seed_scores = {}
            for count in ALL_COUNTS:
                values = [metrics(oof[count][seed], y, person, site)["person_mean"] for seed in SEEDS]
                seed_scores[str(count)] = values
                count_scores[str(count)] = float(np.mean(values))
            best_count = min(ALL_COUNTS, key=lambda value: (count_scores[str(value)], value))
            protocol_result[family] = {
                "fixed_v1_alpha": alpha,
                "v1_alpha_inner_person_mean": choice["inner_person_mean"],
                "count_inner_person_mean": count_scores,
                "count_seed_person_means": seed_scores,
                "best_count_by_inner": best_count,
                "counts_are_not_selected_on_outer": True,
            }

            final_references[protocol][family] = {"fixed_v1_alpha": alpha, "best_count_by_inner": best_count,
                                                  "models": {}}
            for seed in SEEDS:
                source_path = source_run / protocol / "final" / f"{family}_s{seed}.npz"
                source_model, source_record = _verified_model(source_path)
                if source_record["method"] != family or float(source_record["parameter"]) != alpha:
                    raise ValueError("source final model does not match frozen choice")
                source_key = sha(source_path)
                unique_source_costs[_relative(source_path)] = float(source_record["fit_seconds"])
                models = final_references[protocol][family]["models"].setdefault(str(seed), {})
                models[str(REFERENCE_COUNT)] = {"path": _relative(source_path), "sha256": source_key,
                                                "source_64_search_seconds": source_record["fit_seconds"],
                                                "numeric_bytes": source_record["numeric_bytes"],
                                                "serialized_bytes": source_record["serialized_bytes"]}
                for count in PREFIX_COUNTS:
                    destination = run / protocol / "final" / f"{family}_k{count}_s{seed}.npz"
                    if destination.exists() and destination.with_suffix(".json").exists():
                        record = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
                        if sha(destination) != record["artifact_sha256"]:
                            raise ValueError("changed compact final artifact")
                    else:
                        model, receipt = refit_prefix(
                            source_model, x, y, person, site, alpha, count, device
                        )
                        total_refit_seconds += receipt["prefix_refit_seconds"]
                        record = _save_prefix(destination, model, {
                            "protocol": protocol, "family": family, "alpha": alpha,
                            "prefix_atoms": count, "seed": seed, **receipt,
                            "source_64_path": _relative(source_path),
                            "source_64_sha256": source_key,
                            "source_64_search_seconds": source_record["fit_seconds"],
                            "source_plus_prefix_seconds_charged": source_record["fit_seconds"]
                            + receipt["prefix_refit_seconds"],
                            "train_metrics": metrics(prefix_predict(model, x), y, person, site),
                        })
                    models[str(count)] = {"path": _relative(destination),
                                          "sha256": record["artifact_sha256"],
                                          "numeric_bytes": record["numeric_bytes"],
                                          "serialized_bytes": record["serialized_bytes"]}
            payload_reference = final_references[protocol][family]["models"][str(SEEDS[0])]
            protocol_result[family]["numeric_bytes_by_count"] = {
                count: payload_reference[count]["numeric_bytes"] for count in map(str, ALL_COUNTS)
            }
            protocol_result[family]["serialized_bytes_by_count"] = {
                count: payload_reference[count]["serialized_bytes"] for count in map(str, ALL_COUNTS)
            }
        aggregate["protocols"][protocol] = protocol_result

    aggregate["cost_accounting"] = {
        "actual_prefix_refit_seconds_this_study": total_refit_seconds,
        "unique_inherited_v1_64_search_seconds": float(sum(unique_source_costs.values())),
        "inherited_v1_64_fit_runs": len(unique_source_costs),
        "note": "Inherited 64-atom search is counted once per source fit path, including separately executed byte-identical fits; per-prefix receipts also show source-plus-refit charged cost.",
    }
    aggregate["prefix_refit_records"] = len(all_records)
    write_json(run / "inner_records.json", all_records)
    write_json(run / "aggregate.json", aggregate)
    write_json(run / "frozen_prefixes.json", {
        "source_frozen_selections_sha256": sha(frozen_path),
        "policy": "fixed v1 alpha; prefix order frozen; best K selected by inner OOF only",
        "outer_evaluation_run": False,
        "protocols": final_references,
    })
    print("All prefix weights and inner K choices frozen; outer evaluation has not run.", flush=True)


def evaluation_phase(data, source_run, run):
    frozen_path = run / "frozen_prefixes.json"
    if not frozen_path.exists():
        raise ValueError("fit and freeze all compact prefixes before outer evaluation")
    status_path = run / "evaluation_status.json"
    if status_path.exists():
        raise ValueError("outer evaluation is already recorded")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    x_all, y_all = data["color"], data["target"]
    person_all, site_all, camera_all = data["patient"], data["site"], data["device"]
    records = []
    for protocol, families in frozen["protocols"].items():
        _, held, _ = _load_roles(source_run, protocol)
        x, y = x_all[held], y_all[held]
        person, site, camera = person_all[held], site_all[held], camera_all[held]
        for family, family_record in families.items():
            for seed, models in family_record["models"].items():
                for count, reference in models.items():
                    path = ROOT / reference["path"]
                    if sha(path) != reference["sha256"]:
                        raise ValueError("frozen final prefix changed before evaluation")
                    model = load_model(path)
                    prediction = prefix_predict(model, x)
                    record = {
                        "protocol": protocol,
                        "family": family,
                        "seed": int(seed),
                        "atoms": int(count),
                        "selected_by_inner": int(count) == family_record["best_count_by_inner"],
                        "outer_metrics": metrics(prediction, y, person, site, camera),
                    }
                    records.append(record)
                    destination = run / protocol / "outer" / f"{family}_k{count}_s{seed}.npz"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(destination, prediction=prediction, target=y, person=person, site=site,
                             camera=camera)
    write_evaluation_receipts(run, frozen_path, {
        "scope": "exploratory reused original-TRAIN outer roles; K frozen by inner OOF",
        "models": records,
    })


def write_evaluation_receipts(run, frozen_path, result):
    """Persist evaluation output and status without mutating the frozen model manifest."""
    frozen_sha = sha(frozen_path)
    payload = {**result, "frozen_prefixes_sha256": frozen_sha}
    evaluation_path = run / "evaluation.json"
    status_path = run / "evaluation_status.json"
    if status_path.exists():
        raise ValueError("outer evaluation status already exists")
    write_json(evaluation_path, payload)
    if sha(frozen_path) != frozen_sha:
        raise ValueError("frozen prefix manifest changed during evaluation bookkeeping")
    write_json(status_path, {
        "outer_evaluation_run": True,
        "frozen_prefixes_sha256": frozen_sha,
        "evaluation_sha256": sha(evaluation_path),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage", choices=("fit", "evaluate"), required=True)
    parser.add_argument("--acknowledge-outer-evaluation", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.run.mkdir(parents=True, exist_ok=True)
    _lock_sources(args.cache, args.source_run, args.run, args.device)
    with np.load(args.cache, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "target", "patient", "site", "device")}
    if data["color"].shape != (966, 36) or not np.isfinite(data["target"]).all():
        raise ValueError("invalid original TRAIN cache")
    started = time.perf_counter()
    if args.stage == "fit":
        fit_phase(data, args.source_run, args.run, args.device)
    else:
        if not args.acknowledge_outer_evaluation:
            raise ValueError("outer evaluation requires explicit acknowledgement")
        evaluation_phase(data, args.source_run, args.run)
    write_json(args.run / f"{args.stage}_completion.json", {
        "seconds_this_invocation": time.perf_counter() - started,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


if __name__ == "__main__":
    main()
