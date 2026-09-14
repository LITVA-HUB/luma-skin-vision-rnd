"""TRAIN-only AS INNER pass diagnostics; no training or adopted exit policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from chromaseed_recurrent_stream import StreamingPredictor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from luma_skin_vision.color import delta_e00  # noqa: E402

OUT = Path("D:/Luma-RnD/chromaseed_inner_passes_v1")
AS = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
SEAL = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1/verification.json"
SEAL_SHA = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
CACHE = Path("C:/Users/dimal/Documents/просто/luma-skin-vision-rnd/data/processed/skin_mskcc_pixels_v1/train.npz")
CACHE_SHA = "d7e7b4b4fc4574d620cbac8364dd8bb91be51769f4045bb8b4ddf2ad840556e0"
STREAM = Path("D:/Luma-RnD/chromaseed_recurrent_stream_v1/qualification.json")
STREAM_SHA = "712520d394f845565d4f6e678070ffdfe04341933b22efb9eb81229ec7eb3fb0"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
SEEDS = (17, 29, 43)
RATES = (1e-5, 1e-4)
THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"preserve existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def check_bindings(mapping):
    for path, expected in mapping.items():
        if digest(path) != expected:
            raise ValueError(f"bound input changed: {path}")


def check_partition(fit, query, allowed, person):
    arrays = [np.asarray(a) for a in (fit, query, allowed)]
    for a in arrays:
        if (a.ndim != 1 or a.dtype.kind not in "iu" or not len(a)
                or np.any(a >= len(person)) or np.any(a < 0)
                or not np.array_equal(a, np.unique(a))):
            raise ValueError("invalid row indices")
    fit, query, allowed = arrays
    if np.intersect1d(fit, query).size or not np.array_equal(np.union1d(fit, query), allowed):
        raise ValueError("fit/query do not partition the allowed INNER population")
    if set(person[fit]) & set(person[query]):
        raise ValueError("fit/query person leakage")


def summarize(predictions, target, person):
    predictions, target, person = map(np.asarray, (predictions, target, person))
    if (target.ndim != 2 or target.shape[1:] != (3,) or not len(target)
            or predictions.shape != (3, len(target), 4, 3) or person.shape != (len(target),)
            or not np.isfinite(predictions).all() or not np.isfinite(target).all()):
        raise ValueError("finite three-seed, four-pass predictions and aligned targets/people required")
    errors = delta_e00(predictions, target[None, :, None, :])
    people = np.unique(person)
    weights = np.empty(len(person), float)
    for p in people:
        mask = person == p
        weights[mask] = 1 / (len(people) * int(mask.sum()))
    seed_weights = np.broadcast_to(weights[None] / 3, errors.shape[:2])
    prefixes = []
    for j in range(4):
        e = errors[:, :, j]
        prefixes.append(dict(passes=j+1, person_mean=float(np.sum(e * seed_weights)),
                             image_mean=float(e.mean()),
                             mean_seed_p90=float(np.quantile(e, 0.9, axis=1).mean()),
                             seed_person_mean=[float(np.sum(value * weights)) for value in e]))
    plateaus = []
    for j in (1, 2):
        change = np.linalg.norm(predictions[:, :, j] - predictions[:, :, j-1], axis=-1)
        delta = errors[:, :, 3] - errors[:, :, j]
        for threshold in THRESHOLDS:
            selected = change <= threshold
            mass = float(seed_weights[selected].sum())
            row = dict(passes=j+1, threshold=threshold, selected_seed_rows=int(selected.sum()),
                       person_weighted_coverage=mass, current_person_weighted_error=None,
                       later_minus_current_error=None, later_harm_gt_point1=None,
                       later_improvement_gt_point1=None)
            if mass > 0:
                w = seed_weights[selected] / mass
                row.update(current_person_weighted_error=float(np.sum(errors[:, :, j][selected] * w)),
                           later_minus_current_error=float(np.sum(delta[selected] * w)),
                           later_harm_gt_point1=float(np.sum((delta[selected] > 0.1) * w)),
                           later_improvement_gt_point1=float(np.sum((delta[selected] < -0.1) * w)))
            plateaus.append(row)
    return dict(n_rows=len(target), n_people=len(people), seeds=3, prefixes=prefixes,
                descriptive_best_prefix=min(prefixes, key=lambda p: (p["person_mean"], p["passes"]))["passes"],
                plateaus=plateaus, adopted_policy=None,
                evidence="OOF development diagnostics after checkpoint/LR selection on these same INNER data")


def register():
    if digest(SEAL) != SEAL_SHA or digest(STREAM) != STREAM_SHA or digest(CACHE) != CACHE_SHA:
        raise ValueError("upstream seal, stream qualification or TRAIN cache changed")
    seal, stream = read(SEAL), read(STREAM)
    if not seal["passed"] or digest(AS / "selections.json") != seal["selection_sha256"]:
        raise ValueError("AS selection is not the sealed selection")
    inherited = {key: value for section in ("sources", "inputs", "postprocess_sources", "artifact_sha256")
                 for key, value in seal[section].items()}
    bindings = {str(path): digest(path) for path in (SEAL, STREAM, CACHE, AS / "selections.json")}
    for name in ("scripts/chromaseed_architecture_scale.py", "scripts/skin_local_search_train.py",
                 "src/luma_skin_vision/color.py"):
        path = ROOT / name
        if digest(path) != inherited[name]:
            raise ValueError(f"frozen source changed: {name}")
        bindings[str(path)] = digest(path)
    streaming_path = ROOT / "scripts/chromaseed_recurrent_stream.py"
    expected = next(row["sha256"] for row in stream["inputs"]
                    if Path(row["path"]) == streaming_path)
    if digest(streaming_path) != expected:
        raise ValueError("streaming consumer changed")
    bindings[str(streaming_path)] = expected
    choices = []
    selection = read(AS / "selections.json")
    for role in ROLES:
        for variant in VARIANTS:
            choice = selection["roles"][role]["policies"][variant]
            if choice["step"] not in (128, 512, 2048) or choice["lr"] not in RATES:
                raise ValueError("unexpected AS checkpoint/rate")
            choices.append(dict(role=role, variant=variant, step=choice["step"],
                                lr=choice["lr"], original_four_pass_clean=choice["clean"]))
            for fold in range(3):
                base = AS / "inner" / role / variant / f"fold{fold}"
                for name in ("receipt.json", "rows.npz", f"models_{choice['step']}.npz", f"oof_{choice['step']}.npz"):
                    path = base / name
                    relative = path.relative_to(ROOT).as_posix()
                    if digest(path) != inherited[relative]:
                        raise ValueError(f"AS INNER file differs from seal: {relative}")
                    bindings[str(path)] = inherited[relative]
    for name in ("scripts/chromaseed_inner_passes.py", "tests/test_chromaseed_inner_passes.py",
                 "docs/research/chromaseed_inner_passes_v1_protocol.md"):
        path = ROOT / name
        bindings[str(path)] = digest(path)
    value = dict(schema="luma.chromaseed.inner-passes.v1", bindings=bindings, choices=choices,
                 inner_banks=36, selected_models=108, seeds=list(SEEDS), thresholds=list(THRESHOLDS),
                 expected_cases=12, adopted_policy=None, outer_inference=False, gpu_context=False)
    write_once(OUT / "registration.json", value)
    print("INNER PASSES REGISTERED", digest(OUT / "registration.json"), len(bindings), flush=True)


def load_data():
    if CACHE.name != "train.npz" or digest(CACHE) != CACHE_SHA:
        raise ValueError("only the original TRAIN cache is allowed")
    with np.load(CACHE, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("color", "tokens", "target", "patient", "device")}
    if (data["color"].shape != (966, 36) or data["tokens"].shape != (966, 64, 18)
            or data["target"].shape != (966, 3) or len(np.unique(data["patient"])) != 24):
        raise ValueError("unexpected original TRAIN population")
    return data


def case_rows(data, choice, fold):
    from skin_local_search_train import folds_for, roles

    allowed = np.flatnonzero(roles(data["patient"], data["device"])[choice["role"]][0])
    assignment = folds_for(data["patient"][allowed], data["device"][allowed])
    base = AS / "inner" / choice["role"] / choice["variant"] / f"fold{fold}"
    with np.load(base / "rows.npz", allow_pickle=False) as archive:
        fit, query = archive["fit_rows"], archive["query_rows"]
    check_partition(fit, query, allowed, data["patient"])
    np.testing.assert_array_equal(fit, allowed[assignment != fold])
    np.testing.assert_array_equal(query, allowed[assignment == fold])
    return allowed, query, base


def saved_oof(data, choice, fold):
    allowed, query, base = case_rows(data, choice, fold)
    with np.load(base / f"oof_{choice['step']}.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(query, archive["row_indices"])
        saved = archive["predictions"][[2*j + RATES.index(choice["lr"]) for j in range(3)]]
    return allowed, query, base, saved


def infer_case(data, choice):
    from chromaseed_architecture_scale import Predictor

    all_predictions = None
    maximum_reference = maximum_saved = 0.0
    for fold in range(3):
        allowed, query, base, saved = saved_oof(data, choice, fold)
        if all_predictions is None:
            all_predictions = np.full((3, len(allowed), 4, 3), np.nan)
        with np.load(base / f"models_{choice['step']}.npz", allow_pickle=False) as archive:
            for si, seed in enumerate(SEEDS):
                prefix = f"{2*si + RATES.index(choice['lr'])}__"
                model = {key[len(prefix):]: archive[key] for key in archive.files if key.startswith(prefix)}
                if str(model["variant"]) != choice["variant"]:
                    raise ValueError("model architecture mismatch")
                stream, reference = StreamingPredictor(model), Predictor(model)
                outputs = np.stack([np.stack([p.prediction for p in stream.iter_passes(data["color"][row], data["tokens"][row])])
                                    for row in query])
                check = np.concatenate([reference(data["color"][query[j:j+16]], data["tokens"][query[j:j+16]], all_passes=True)
                                        for j in range(0, len(query), 16)])
                np.testing.assert_allclose(outputs, check, atol=1e-8, rtol=0)
                np.testing.assert_allclose(outputs[:, -1], saved[si], atol=0.002, rtol=1e-6)
                maximum_reference = max(maximum_reference, float(np.max(np.abs(outputs-check))))
                maximum_saved = max(maximum_saved, float(np.max(np.abs(outputs[:, -1]-saved[si]))))
                all_predictions[si, np.searchsorted(allowed, query)] = outputs
        print("INNER PASSES", choice["role"], choice["variant"], "fold", fold, "models", 3, flush=True)
    summary = summarize(all_predictions, data["target"][allowed], data["patient"][allowed])
    if abs(summary["prefixes"][-1]["person_mean"]-choice["original_four_pass_clean"]) > 2e-5:
        raise ValueError("four-pass metric did not reproduce the AS selection score")
    path = OUT / "traces" / f"{choice['role']}__{choice['variant']}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        np.savez(output, row_indices=allowed, predictions=all_predictions, seeds=np.asarray(SEEDS))
    return dict(**choice, summary=summary, trace_file=str(path), trace_sha256=digest(path),
                maximum_numpy_lab_drift=maximum_reference, maximum_saved_cuda_lab_drift=maximum_saved)


def run():
    registration = read(OUT / "registration.json")
    check_bindings(registration["bindings"])
    if (OUT / "results.json").exists() or (OUT / "traces").exists():
        raise FileExistsError("preserve existing diagnostic run; use verify on a completed run")
    data = load_data()
    import torch

    if torch.cuda.is_initialized():
        raise ValueError("diagnostic must not initialize CUDA")
    with threadpool_limits(limits=1):
        cases = [infer_case(data, choice) for choice in registration["choices"]]
    if torch.cuda.is_initialized():
        raise ValueError("diagnostic initialized CUDA")
    check_bindings(registration["bindings"])
    result = dict(registration_sha256=digest(OUT / "registration.json"), cases=cases,
                  selected_models=108, inner_cases=12, seed_row_predictions=sum(3*c["summary"]["n_rows"] for c in cases),
                  native_lab_vectors=sum(12*c["summary"]["n_rows"] for c in cases),
                  adopted_policy=None, fresh_quality_evidence=False, latency_seconds=None,
                  cuda_initialized=False, outer_inference_rows=0)
    write_once(OUT / "results.json", result)
    print("INNER PASSES COMPLETE", digest(OUT / "results.json"), result["seed_row_predictions"], flush=True)


def verify():
    registration, result = read(OUT / "registration.json"), read(OUT / "results.json")
    if digest(OUT / "registration.json") != result["registration_sha256"]:
        raise ValueError("registration changed")
    check_bindings(registration["bindings"])
    if len(result["cases"]) != 12:
        raise ValueError("incomplete diagnostics")
    data = load_data()
    for choice, case in zip(registration["choices"], result["cases"], strict=True):
        for key, value in choice.items():
            if case[key] != value:
                raise ValueError("diagnostic case selection mismatch")
        path = OUT / "traces" / f"{choice['role']}__{choice['variant']}.npz"
        if Path(case["trace_file"]) != path or digest(path) != case["trace_sha256"]:
            raise ValueError("saved trace changed")
        with np.load(path, allow_pickle=False) as archive:
            rows, predictions = archive["row_indices"], archive["predictions"]
            np.testing.assert_array_equal(archive["seeds"], SEEDS)
        maximum_saved = 0.0
        for fold in range(3):
            allowed, query, _, old = saved_oof(data, choice, fold)
            np.testing.assert_array_equal(rows, allowed)
            actual = predictions[:, np.searchsorted(rows, query), -1]
            np.testing.assert_allclose(actual, old, atol=0.002, rtol=1e-6)
            maximum_saved = max(maximum_saved, float(np.max(np.abs(actual-old))))
        if maximum_saved != case["maximum_saved_cuda_lab_drift"]:
            raise ValueError("saved CUDA comparison mismatch")
        if summarize(predictions, data["target"][rows], data["patient"][rows]) != case["summary"]:
            raise ValueError("summary did not reproduce")
        if abs(case["summary"]["prefixes"][-1]["person_mean"] - choice["original_four_pass_clean"]) > 2e-5:
            raise ValueError("AS four-pass control score mismatch")
    check_bindings(registration["bindings"])
    value = dict(passed=True, results_sha256=digest(OUT / "results.json"),
                 registration_sha256=digest(OUT / "registration.json"), cases=12,
                 source_rows_and_summary_replayed=True,
                 full_network_replay=False,
                 scope="Independent frozen NumPy all-pass comparison ran in primary; this stage replays row roles, saved CUDA OOF and summary arithmetic")
    write_once(OUT / "verification.json", value)
    print("INNER PASSES VERIFIED", digest(OUT / "verification.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run", "verify"))
    command = parser.parse_args().command
    {"register": register, "run": run, "verify": verify}[command]()


if __name__ == "__main__":
    main()
