"""Earlier and lower-rate WIDE training with exact original and paired canaries."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
from chromaseed_gate_stability import affine_features, grid, summaries
from chromaseed_gated import flatten, unpack
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import context, save_npz
from chromaseed_neural_prefix_numpy import predict as base_predict
from chromaseed_patch8_numpy import choose, transform_tokens
from chromaseed_refine_train import setup
from chromaseed_widen import SEEDS, capacity, predict
from chromaseed_widen_audit import exact
from chromaseed_widen_early_fit import FIRST_RATES, fit
from chromaseed_widen_run import OUT as WIDE_OUT
from chromaseed_widen_run import ROOT, check_map, load_data
from chromaseed_widen_run import RUN as WIDE
from chromaseed_widen_run import bank_path as wide_bank
from skin_local_search_train import CACHE_HASH, metrics, roles, sha, weights_for, write_json

CAPS = ("m31", "m61", "m111", "m832")
TIMES = (0, 32, 128, 512, 2048)
RUN = ROOT / "experiments/runs/chromaseed_widen_early_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_widen_early_v1"
PARENT = "d86795ebc9659fdefe869298fa8a44106555e5ac90a5fa153105d408be3b4876"


def bank_path(role, variant, first_rate, fold=None):
    return (
        RUN
        / ("final" if fold is None else "inner")
        / role
        / variant
        / f"r{FIRST_RATES.index(first_rate)}"
        / ("bank" if fold is None else f"fold{fold}")
    )


def freeze():
    assert sha(WIDE_OUT / "verification.json") == PARENT
    old = js(WIDE_OUT / "verification.json")
    sources = {**old["sources"], **old["postprocess_sources"]}
    inputs = {**old["inputs"], **old["artifact_sha256"]}
    inputs[(WIDE_OUT / "verification.json").relative_to(ROOT).as_posix()] = PARENT
    for p in (
        "scripts/chromaseed_widen_early_fit.py",
        "scripts/chromaseed_widen_early_run.py",
        "tests/test_chromaseed_widen_early.py",
        "docs/research/chromaseed_widen_early_v1_protocol.md",
    ):
        sources[p] = sha(ROOT / p)
    check_map({**sources, **inputs})
    assert all(
        os.environ.get(k) == "1"
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    )
    value = dict(
        sources=sources,
        input_sha256=inputs,
        parent_verification_sha256=PARENT,
        cache_sha256=CACHE_HASH,
        capacities=list(CAPS),
        first_rates=list(FIRST_RATES),
        control_rate=0.001,
        checkpoints=list(TIMES),
        original_horizon=8192,
        inner_banks=108,
        inner_trajectories=648,
        inner_checkpoint_payloads=3240,
        inner_prediction_vectors=612000,
        original_checkpoint_controls=432,
        paired_canaries=1080,
        candidates=222,
        policies=15,
        final_records=90,
        final_stress_vectors=1186020,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        numpy=np.__version__,
        previous_turn_classification="verified progress",
    )
    p = RUN / "source_lock.json"
    if p.exists():
        assert js(p) == value, "WE sources changed after freeze"
    else:
        write_json(p, value)
    return sha(p)


def train_bank(data, source, role, variant, first_rate, fold=None, steps=2048):
    ix, query, warm, parents = context(data, role, fold)
    path = bank_path(role, variant, first_rate, fold)
    selector = None if fold is not None else sha(RUN / "selections.json")
    checkpoints = TIMES if fold is not None else (0, steps)
    if (path / "receipt.json").exists():
        old = js(path / "receipt.json")
        assert (
            old["source_lock_sha256"] == source
            and old["selection_sha256"] == selector
            and old["warm_parent_sha256"] == parents
            and old["steps"] == steps
            and old["checkpoints"] == list(checkpoints)
        )
        for name, digest in old["files"].items():
            assert sha(path / name) == digest
        print("WE REUSE", role, variant, first_rate, fold, flush=True)
        return
    print("WE START", role, variant, first_rate, fold, flush=True)

    def progress(item):
        if item["step"] == steps:
            print(
                "WE TRAIN",
                role,
                variant,
                first_rate,
                fold,
                steps,
                round(item["seconds"], 2),
                flush=True,
            )

    models, info = fit(
        data["color"][ix],
        data["tokens"][ix],
        data["target"][ix],
        weights_for(data["patient"][ix], data["site"][ix]),
        warm,
        variant,
        first_rate,
        steps,
        checkpoints,
        "cuda",
        "cuda_graph",
        progress,
    )
    files = {}

    def save(name, payload):
        save_npz(path / name, payload)
        files[name] = sha(path / name)

    save("warm_models.npz", flatten({str(i): m for i, m in enumerate(warm)}))
    for step, ms in models.items():
        save(f"models_{step}.npz", flatten({str(i): m for i, m in enumerate(ms)}))
        if fold is not None:
            output = np.stack([predict(m, data["color"][query], data["tokens"][query]) for m in ms])
            save(f"oof_{step}.npz", dict(row_indices=query, predictions=output))
    save(
        "rows.npz",
        dict(fit_rows=ix, query_rows=query if fold is not None else np.array([], np.int64)),
    )
    original_controls = paired_canaries = 0
    if fold is not None:
        if first_rate == 0.0001:
            for step in (512, 2048):
                original = nz(wide_bank(role, variant, fold) / f"models_{step}.npz")
                for slot in range(6):
                    exact(models[step][slot], unpack(original, str(slot)))
                    original_controls += 1
                old = nz(wide_bank(role, variant, fold) / f"oof_{step}.npz")
                new = nz(path / f"oof_{step}.npz")
                np.testing.assert_array_equal(new["row_indices"], old["row_indices"])
                np.testing.assert_array_equal(new["predictions"], old["predictions"])
        else:
            for step in checkpoints:
                control = nz(bank_path(role, variant, 0.0001, fold) / f"models_{step}.npz")
                for slot in (1, 3, 5):
                    exact(models[step][slot], unpack(control, str(slot)))
                    paired_canaries += 1
    info.update(
        source_lock_sha256=source,
        selection_sha256=selector,
        role=role,
        fold=fold,
        warm_parent_sha256=parents,
        files=files,
        original_controls_bitwise=original_controls,
        paired_canaries_bitwise=paired_canaries,
    )
    write_json(path / "receipt.json", info)
    print("WE SAVED", role, variant, first_rate, fold, flush=True)


def oof(role, variant, rate, step):
    first = 0.0001 if rate == 0.001 else rate
    paths = [
        wide_bank(role, variant, f) if step == 8192 else bank_path(role, variant, first, f)
        for f in range(3)
    ]
    parts = [nz(p / f"oof_{step}.npz") for p in paths]
    rows = np.concatenate([p["row_indices"] for p in parts])
    order = np.argsort(rows)
    values = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
    slots = [2 * i + (rate == 0.001) for i in range(3)]
    return rows[order], values[slots]


def select(data, source):
    previous = js(WIDE / "selections.json")
    value = dict(source_lock_sha256=source, roles={})
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        ix = np.flatnonzero(mask)
        candidates = []
        for variant in CAPS:
            for rate in (*FIRST_RATES, 0.001):
                for step in (*TIMES[1:], 8192) if rate in (0.0001, 0.001) else TIMES[1:]:
                    rows, predictions = oof(role, variant, rate, step)
                    np.testing.assert_array_equal(rows, ix)
                    mm = [
                        metrics(p, data["target"][ix], data["patient"][ix], data["site"][ix])
                        for p in predictions
                    ]
                    cap = capacity(
                        unpack(nz(bank_path(role, variant, 0.0001, 0) / "models_0.npz"), "0")
                    )
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=rate,
                            **cap,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
            for old in [
                c for c in previous["roles"][role]["candidates"] if c["variant"] == variant
            ]:
                matching = next(
                    c
                    for c in candidates
                    if c["variant"] == variant and c["step"] == old["step"] and c["lr"] == old["lr"]
                )
                assert matching == old, "unchanged WIDE candidate score differs"
        candidates.extend(
            [
                previous["roles"][role]["policies"]["tiny"],
                next(c for c in previous["roles"][role]["candidates"] if c["variant"] == "np"),
            ]
        )
        assert len(candidates) == 74
        policies = {v: choose([c for c in candidates if c["variant"] == v]) for v in CAPS}
        policies["overall"] = choose(candidates)
        value["roles"][role] = dict(candidates=candidates, policies=policies)
    p = RUN / "selections.json"
    if p.exists():
        assert js(p) == value
    else:
        write_json(p, value)
    print("WE SELECTION FROZEN", sha(p), flush=True)
    return value


def selected_sources(data, source, selection):
    previous = js(WIDE / "selections.json")
    entries = []
    for role, entry in selection["roles"].items():
        for variant in CAPS:
            c = entry["policies"][variant]
            first = 0.0001 if c["lr"] == 0.001 else c["lr"]
            imported = c["lr"] in (0.0001, 0.001) and c["step"] in (512, 2048, 8192)
            if imported:
                path = wide_bank(role, variant) / f"models_{c['step']}.npz"
            else:
                train_bank(data, source, role, variant, first, None, c["step"])
                path = bank_path(role, variant, first) / f"models_{c['step']}.npz"
            for si, seed in enumerate(SEEDS):
                slot = 2 * si + (c["lr"] == 0.001)
                entries.append(
                    dict(
                        role=role,
                        variant=variant,
                        kind="new",
                        name=f"new_{variant}_s{seed}",
                        seed=seed,
                        step=c["step"],
                        lr=c["lr"],
                        source_path=path.relative_to(ROOT).as_posix(),
                        source_slot=str(slot),
                        source_sha256=sha(path),
                        imported_from_wide=imported,
                        unchanged_setting=c == previous["roles"][role]["policies"][variant],
                    )
                )
        for variant in (*CAPS, "tiny", "np"):
            if variant == "np":
                for seed in SEEDS:
                    p = WIDE / "models" / role / f"np_s{seed}.npz"
                    entries.append(
                        dict(
                            role=role,
                            variant=variant,
                            kind="np",
                            name=f"np_s{seed}",
                            seed=seed,
                            step=0,
                            lr=None,
                            source_path=p.relative_to(ROOT).as_posix(),
                            source_slot=None,
                            source_sha256=sha(p),
                            imported_from_wide=True,
                            unchanged_setting=True,
                        )
                    )
            else:
                c = previous["roles"][role]["policies"][variant]
                for seed in SEEDS:
                    name = f"{variant}_r{(0.0001, 0.001).index(c['lr'])}_t{c['step']}_s{seed}"
                    p = WIDE / "models" / role / f"{name}.npz"
                    entries.append(
                        dict(
                            role=role,
                            variant=variant,
                            kind="tiny" if variant == "tiny" else "wide",
                            name=f"wide_{variant}_s{seed}",
                            seed=seed,
                            step=c["step"],
                            lr=c["lr"],
                            source_path=p.relative_to(ROOT).as_posix(),
                            source_slot=None,
                            source_sha256=sha(p),
                            imported_from_wide=True,
                            unchanged_setting=True,
                        )
                    )
    assert len(entries) == 90
    p = RUN / "final_sources.json"
    if p.exists():
        assert js(p) == entries
    else:
        write_json(p, entries)
    return entries


def evaluate(data, source, selection, entries):
    records = []
    for entry in entries:
        role = entry["role"]
        query = np.flatnonzero(roles(data["patient"], data["device"])[role][1])
        path = ROOT / entry["source_path"]
        assert sha(path) == entry["source_sha256"]
        raw = nz(path)
        model = raw if entry["source_slot"] is None else unpack(raw, entry["source_slot"])
        p = RUN / "models" / role / f"{entry['name']}.npz"
        save_npz(p, model)
        out = []
        for setting in grid():
            xx = affine_features(data["color"][query], setting["dose"], setting["anchor"])
            tt = transform_tokens(data["tokens"][query], setting["dose"], setting["anchor"])
            out.append(base_predict(model, xx) if entry["kind"] == "np" else predict(model, xx, tt))
        out = np.stack(out)
        pp = RUN / "evaluated" / role / f"{entry['name']}.npz"
        save_npz(pp, dict(row_indices=query, predictions=out))
        trans, doses = summaries(
            out, data["target"][query], data["patient"][query], data["device"][query], None
        )
        cap = dict(parameters=643, numeric_bytes=2886) if entry["kind"] == "np" else capacity(model)
        policies = []
        if entry["kind"] == "new":
            policies.append(entry["variant"])
        overall = selection["roles"][role]["policies"]["overall"]
        if (
            entry["kind"] != "wide"
            and entry["variant"] == overall["variant"]
            and entry["step"] == overall["step"]
            and entry["lr"] == overall["lr"]
        ):
            policies.append("overall")
        records.append(
            dict(
                **entry,
                **cap,
                policies=policies,
                archive_bytes=p.stat().st_size,
                model_sha256=sha(p),
                prediction_sha256=sha(pp),
                metrics=metrics(
                    out[0],
                    data["target"][query],
                    data["patient"][query],
                    data["site"][query],
                    data["device"][query],
                ),
                transforms=trans,
                doses=doses,
            )
        )
        if entry["seed"] == 43:
            print("WE EVALUATED", role, entry["kind"], entry["variant"], flush=True)
    assert len(records) == 90 and sum("overall" in r["policies"] for r in records) == 9
    value = dict(
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        final_sources_sha256=sha(RUN / "final_sources.json"),
        records=records,
    )
    p = RUN / "results.json"
    if p.exists():
        assert js(p) == value
    else:
        write_json(p, value)


def main():
    assert not (OUT / "verification.json").exists(), "sealed WE; report verifier only"
    setup("cuda")
    start = time.perf_counter()
    data = load_data()
    source = freeze()
    write_json(RUN / "job.json", dict(status="running", pid=os.getpid(), source_lock_sha256=source))
    for role in roles(data["patient"], data["device"]):
        for variant in CAPS:
            for fold in range(3):
                for first in (0.0001, 0.00001, 0.00003):
                    train_bank(data, source, role, variant, first, fold)
    selection = select(data, source)
    entries = selected_sources(data, source, selection)
    evaluate(data, source, selection, entries)
    write_json(
        RUN / "job.json",
        dict(
            status="complete",
            pid=os.getpid(),
            source_lock_sha256=source,
            results_sha256=sha(RUN / "results.json"),
            seconds=time.perf_counter() - start,
        ),
    )
    print("WE COMPLETE", sha(RUN / "results.json"), flush=True)


if __name__ == "__main__":
    main()
