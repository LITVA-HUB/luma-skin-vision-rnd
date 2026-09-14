"""Qualify streaming execution with sealed AS exports and generated features only.

Does not read image rows, target labels, evaluated predictions or live HR models.
No timing, training, threshold calibration or model selection is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from chromaseed_recurrent_stream import VARIANTS, StreamingPredictor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("D:/Luma-RnD/chromaseed_recurrent_stream_v1")
AS_SEAL = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1/verification.json"
AS_SHA = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
AS_RESULTS = ROOT / "experiments/runs/chromaseed_architecture_scale_v1/results.json"
P3 = Path("D:/Luma-RnD/chromaseed_palette_transfer_v1/registration.json")
P3_SHA = "90a9a78bc46f5ca1f40befe0cd04e8861374868d41710c099c7573da7a2add17"
COMPUTE = Path("D:/Luma-RnD/chromaseed_compute_structure_v1/analysis.json")
COMPUTE_SHA = "107d6d7b926e1d6fe1c5478ef79ee6db2339459ef86349316b58f5acaa532b68"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
SEEDS = (17, 29, 43)
FIXED_LAYERS = ["token1", "token2", "context", "key"]
PASS_LAYERS = ["query", "update1", "update2", "head"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=digest(path))


def check_bindings(records):
    for record in records:
        if digest(record["path"]) != record["sha256"]:
            raise ValueError(f"input changed: {record['path']}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def array_digest(value):
    array = np.ascontiguousarray(value)
    h = hashlib.sha256(str((array.shape, array.dtype.str)).encode("ascii"))
    h.update(array.tobytes())
    return h.hexdigest()


class ObservedPredictor(StreamingPredictor):
    def __init__(self, model):
        super().__init__(model)
        self.calls = []

    def _layer(self, name, value):
        weight, _ = self.layers[name]
        self.calls.append(dict(layer=name, input_shape=list(value.shape),
                               linear_macs=math.prod(value.shape[:-1]) * math.prod(weight.shape)))
        return super()._layer(name, value)


def work(calls, passes):
    if [row["layer"] for row in calls] != FIXED_LAYERS + PASS_LAYERS * passes:
        raise ValueError("stream executed the wrong number/order of layers")
    return sum(row["linear_macs"] for row in calls)


def qualify_case(model, x, tokens):
    from chromaseed_head_range import Predictor

    reference = Predictor(model)(x, tokens, all_passes=True)
    actual = ObservedPredictor(model)
    iterator = actual.iter_passes(x, tokens)
    if actual.calls:
        raise ValueError("stream was not lazy")
    outputs = list(iterator)
    result = np.stack([row.prediction for row in outputs])
    if not np.array_equal(result, reference):
        raise ValueError(f"prefix numerical mismatch: {np.max(np.abs(result-reference))}")
    full_macs = work(actual.calls, 4)
    full_trace = list(actual.calls)
    prefixes = []
    for count in range(1, 5):
        actual.calls.clear()
        trace = actual.predict_trace(x, tokens, max_passes=count)
        if (trace.passes_executed != count or trace.stop_reason != "max_passes"
                or not np.array_equal(trace.prediction, reference[count-1])):
            raise ValueError("fixed prefix did not match the frozen consumer")
        macs = work(actual.calls, count)
        prefixes.append(dict(passes=count, linear_macs=macs,
                             linear_mac_reduction=1-macs/full_macs))
    actual.calls.clear()
    forced = actual.predict_trace(x, tokens, exit_threshold=1e12)
    if (forced.passes_executed != 2 or forced.stop_reason != "delta_threshold"
            or not np.array_equal(forced.prediction, reference[1])):
        raise ValueError("forced synthetic exit failed")
    forced_macs = work(actual.calls, 2)
    actual.calls.clear()
    default = actual.predict_trace(x, tokens)
    if default.passes_executed != 4 or not np.array_equal(default.prediction, reference[-1]):
        raise ValueError("default no longer means fixed four passes")
    work(actual.calls, 4)
    return dict(all_passes_bitwise_equal=True, max_abs_lab_drift=float(np.max(np.abs(result-reference))),
                predictions_sha256=array_digest(result), features_sha256=array_digest(x),
                tokens_sha256=array_digest(tokens),
                successive_changes_l2=[row.change_l2 for row in outputs],
                active_tokens=[row.active_tokens for row in outputs],
                trace=full_trace, prefixes=prefixes,
                forced_exit=dict(threshold=1e12, passes_executed=2, linear_macs=forced_macs,
                                 purpose="Synthetic functional trigger; not a selected threshold"),
                default_passes=default.passes_executed, quality_metric=None, latency_seconds=None)


def prepare():
    for path, expected in ((AS_SEAL, AS_SHA), (P3, P3_SHA), (COMPUTE, COMPUTE_SHA)):
        if digest(path) != expected:
            raise ValueError(f"upstream registration/seal changed: {path}")
    seal = read_json(AS_SEAL)
    if not seal["passed"] or digest(AS_RESULTS) != seal["results_sha256"]:
        raise ValueError("AS results do not match their completed seal")
    registration = read_json(P3)
    originals = [ROOT / "scripts/chromaseed_architecture_scale.py",
                 ROOT / "scripts/chromaseed_head_range.py"]
    for path in originals:
        if digest(path) != registration["bindings"][path.relative_to(ROOT).as_posix()]:
            raise ValueError("frozen reference consumer changed")
    # Fixed full enumeration, never sorted or filtered by any quality score.
    rows = [row for row in read_json(AS_RESULTS)["records"] if row["variant"] in VARIANTS]
    keys = [(row["role"], row["variant"], row["seed"]) for row in rows]
    expected = {(role, variant, seed) for role in ROLES for variant in VARIANTS for seed in SEEDS}
    if len(rows) != 36 or set(keys) != expected or len(set(keys)) != len(keys):
        raise ValueError("AS recurrent export coverage mismatch")
    paths = []
    for row in rows:
        path = ROOT / row["model"]
        fixed_path = ROOT / ("experiments/runs/chromaseed_architecture_scale_v1/models/"
                             f"{row['role']}/{row['variant']}_s{row['seed']}.npz")
        if path != fixed_path or digest(path) != row["model_sha256"]:
            raise ValueError("AS export does not match its sealed model hash/path")
        paths.append(path)
    inputs = [binding(path) for path in [AS_SEAL, AS_RESULTS, P3, COMPUTE, *originals,
              Path(__file__), ROOT / "scripts/chromaseed_recurrent_stream.py",
              ROOT / "tests/test_chromaseed_recurrent_stream.py", *paths]]
    import torch  # Reference consumer imports Torch; no CUDA context is created.

    if torch.cuda.is_initialized():
        raise ValueError("qualification must not initialize CUDA")
    structure = read_json(COMPUTE)["single_model_cases"]
    cases = []
    with threadpool_limits(limits=1):
        for row, path in zip(rows, paths, strict=True):
            with np.load(path, allow_pickle=False) as archive:
                model = {key: archive[key] for key in archive.files}
            if str(model.get("head_mode", "unit")) != "unit":
                raise ValueError("sealed AS export unexpectedly has a different head")
            # Same generated normalized probes for each model; no real dataset rows.
            rng = np.random.default_rng(9142602)
            probes = []
            for _ in range(2):
                x = (model["base_x_mean"] + model["base_x_std"] * rng.normal(0, 0.7, 36)).astype(np.float32)
                t = (model["t_mean"] + model["t_std"] * rng.normal(0, 0.7, (64, 18))).astype(np.float32)
                probe = qualify_case(model, x, t)
                previous = next(item for item in structure
                                if item["variant"] == row["variant"] and item["head_mode"] == "unit")
                for actual, projected in zip(probe["prefixes"], previous["prefix_counterfactuals"], strict=True):
                    if actual["linear_macs"] != projected["linear_macs"]:
                        raise ValueError("actual skipped-pass MACs disagree with prior operation trace")
                probes.append(probe)
            cases.append(dict(role=row["role"], variant=row["variant"], seed=row["seed"],
                              model_sha256=row["model_sha256"], generated_probes=probes))
    check_bindings(inputs)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA was initialized during qualification")
    return dict(schema="luma.chromaseed.recurrent-stream.v1", inputs=inputs,
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, reference_torch=torch.__version__),
                exports=36, generated_feature_cases=72, cases=cases,
                blas_threads=1, gpu_context_initialized=False,
                image_rows_read=0, target_rows_read=0, quality_metric=None, latency_seconds=None,
                limitations=["All inputs are generated features, not photographs or reference targets.",
                             "AS exports use the unit head; separate scoped tests cover all three HR heads with synthetic weights.",
                             "No live HR export, held target, model selection or threshold calibration is used.",
                             "Dense linear counts exclude anchor, normalization, activation, attention arithmetic and image processing.",
                             "Numerical equality and omitted updates are not quality or measured latency improvements.",
                             "Default execution remains four passes; the large forcing threshold is only a functional probe."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    path = OUT / "qualification.json"
    if args.command == "run" and path.exists():
        raise FileExistsError("qualification already exists; use verify")
    if args.command == "verify":
        saved = read_json(path)
        check_bindings(saved["inputs"])
    result = prepare()
    if args.command == "run":
        OUT.mkdir(parents=True, exist_ok=True)
        result["created_utc"] = datetime.now(timezone.utc).isoformat()
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        saved.pop("created_utc")
        if saved != result:
            raise ValueError("streaming qualification did not reproduce")
    print(json.dumps(dict(command=args.command, sha256=digest(path), exports=result["exports"],
                          generated_feature_cases=result["generated_feature_cases"],
                          all_passes_bitwise_equal=True, cuda_initialized=False,
                          quality_metric=None, latency_seconds=None)))


if __name__ == "__main__":
    main()
