"""Independent WE sampling/lineage, canaries, candidate reconstruction and consumers."""

from __future__ import annotations

import hashlib
import time

import numpy as np
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics, verify_normalizers
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen import Predictor
from chromaseed_widen_audit import PARAMS, direct, exact, independent_gpu, local_transform
from chromaseed_widen_early_run import (
    OUT,
    ROOT,
    RUN,
    WIDE,
    bank_path,
    check_map,
    load_data,
    wide_bank,
)
from skin_local_search_train import folds_for, roles, sha, write_json

CAPS = ("m31", "m61", "m111", "m832")
FIRST = (0.00001, 0.00003, 0.0001)
SEEDS = (17, 29, 43)
TIMES = (0, 32, 128, 512, 2048)


def inspect_bank(data, source, role, variant, rate, fold, artifacts, steps=2048):
    mask, held = roles(data["patient"], data["device"])[role]
    ix, query = np.flatnonzero(mask), np.flatnonzero(held)
    if fold is not None:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
    path = bank_path(role, variant, rate, fold)
    info = js(path / "receipt.json")
    assert info["source_lock_sha256"] == source and info["selection_sha256"] == (
        None if fold is not None else sha(RUN / "selections.json")
    )
    assert (
        info["first_rate"] == rate
        and info["variant"] == variant
        and info["steps"] == steps
        and info["schedule_horizon"] == 8192
        and info["trajectory_count"] == 6
    )
    assert info["slots"] == [dict(seed=s, lr=r) for s in SEEDS for r in (rate, 0.001)]
    assert info["checkpoints"] == list(TIMES if fold is not None else (0, steps)) and info[
        "original_rows"
    ] == len(ix)
    assert info["engine"] == "cuda_graph" and info["device"] == "cuda"
    check_map(info["warm_parent_sha256"])
    for name, digest in info["files"].items():
        assert sha(path / name) == digest
        artifacts[(path / name).relative_to(ROOT).as_posix()] = digest
    artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
    rr = nz(path / "rows.npz")
    np.testing.assert_array_equal(rr["fit_rows"], ix)
    np.testing.assert_array_equal(
        rr["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    oldpath = wide_bank(role, variant, fold)
    oldrows = nz(oldpath / "rows.npz")
    np.testing.assert_array_equal(oldrows["fit_rows"], ix)
    np.testing.assert_array_equal(oldrows["query_rows"], rr["query_rows"])
    oldwarm = nz(oldpath / "warm_models.npz")
    warm = nz(path / "warm_models.npz")
    exact(warm, oldwarm)
    token = data["tokens"][ix]
    assert hashlib.sha256(token.tobytes()).hexdigest() == info["fit_tokens_sha256"]
    flat = token.astype(float).reshape(-1, 18)
    tm, ts = flat.mean(0).astype(np.float32), np.maximum(flat.std(0), 1e-6).astype(np.float32)
    np.testing.assert_array_equal(np.array(info["token_mean"], np.float32), tm)
    np.testing.assert_array_equal(np.array(info["token_std"], np.float32), ts)
    weights = balanced(data["patient"][ix], data["site"][ix])
    weights /= weights.sum()
    sampled = np.stack(
        [np.random.default_rng(s + 9001).choice(len(ix), (steps, 64), p=weights) for s in SEEDS]
    )
    assert hashlib.sha256(sampled.tobytes()).hexdigest() == info["sampling_sha256"]
    factor = np.float32(0.1 + 0.45 * (1 + np.cos(np.pi * (steps - 1) / 8191)))
    np.testing.assert_array_equal(
        np.array(info["final_learning_rates"], np.float32),
        np.tile(np.array([rate, 0.001], np.float32), 3) * factor,
    )
    initial = nz(path / "models_0.npz")
    exact(initial, nz(oldpath / "models_0.npz"))
    for si in range(3):
        verify_normalizers(model_from(warm, str(si)), data["color"][ix], data["target"][ix])
    return ix, query, tm, ts, info


def audit_inner(data, source, selection, artifacts):
    counts = dict(
        inner_banks=0,
        inner_models=0,
        inner_vectors=0,
        original_controls=0,
        paired_canaries=0,
        candidates=0,
        choices=0,
    )
    maxlab = maxgpu = 0.0
    old_selection = js(WIDE / "selections.json")
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        fullrows = np.flatnonzero(mask)
        candidates = []
        for variant in CAPS:
            by_rate = {}
            for first in FIRST:
                folds = {step: [] for step in TIMES}
                allqueries = []
                for fold in range(3):
                    ix, query, tm, ts, info = inspect_bank(
                        data, source, role, variant, first, fold, artifacts
                    )
                    allqueries.append(query)
                    path = bank_path(role, variant, first, fold)
                    originals = canaries = 0
                    for step in TIMES:
                        packed = nz(path / f"models_{step}.npz")
                        saved = nz(path / f"oof_{step}.npz")
                        np.testing.assert_array_equal(saved["row_indices"], query)
                        ms = [model_from(packed, str(i)) for i in range(6)]
                        if first == 0.0001 and step in (512, 2048):
                            old = nz(wide_bank(role, variant, fold) / f"models_{step}.npz")
                            for slot, m in enumerate(ms):
                                exact(m, model_from(old, str(slot)))
                                originals += 1
                            exact(saved, nz(wide_bank(role, variant, fold) / f"oof_{step}.npz"))
                        if first != 0.0001:
                            original = nz(
                                bank_path(role, variant, 0.0001, fold) / f"models_{step}.npz"
                            )
                            for slot in (1, 3, 5):
                                exact(ms[slot], model_from(original, str(slot)))
                                canaries += 1
                        predictions = []
                        for slot, m in enumerate(ms):
                            Predictor(m)
                            verify_normalizers(m, data["color"][ix], data["target"][ix])
                            np.testing.assert_array_equal(m["t_mean"], tm)
                            np.testing.assert_array_equal(m["t_std"], ts)
                            pred = direct(m, data["color"][query], data["tokens"][query])
                            maxlab = max(
                                maxlab, float(np.max(abs(pred - saved["predictions"][slot])))
                            )
                            np.testing.assert_allclose(
                                pred, saved["predictions"][slot], rtol=0, atol=2e-8
                            )
                            predictions.append(pred)
                            counts["inner_models"] += 1
                            counts["inner_vectors"] += len(query)
                        predictions = np.stack(predictions)
                        folds[step].append(predictions)
                        if step == 2048:
                            maxgpu = max(
                                maxgpu,
                                independent_gpu(
                                    ms,
                                    data["color"][query[:3]],
                                    data["tokens"][query[:3]],
                                    predictions[:, :3],
                                ),
                            )
                    assert (
                        info["original_controls_bitwise"] == originals
                        and info["paired_canaries_bitwise"] == canaries
                    )
                    counts["original_controls"] += originals
                    counts["paired_canaries"] += canaries
                    counts["inner_banks"] += 1
                rows = np.concatenate(allqueries)
                order = np.argsort(rows)
                np.testing.assert_array_equal(rows[order], fullrows)
                by_rate[first] = {
                    step: np.concatenate(folds[step], axis=1)[:, order] for step in TIMES
                }
            for rate in (*FIRST, 0.001):
                first = 0.0001 if rate == 0.001 else rate
                times = (*TIMES[1:], 8192) if rate in (0.0001, 0.001) else TIMES[1:]
                for step in times:
                    if step == 8192:
                        parts = [nz(wide_bank(role, variant, f) / "oof_8192.npz") for f in range(3)]
                        rr = np.concatenate([p["row_indices"] for p in parts])
                        order = np.argsort(rr)
                        np.testing.assert_array_equal(rr[order], fullrows)
                        values = np.concatenate([p["predictions"] for p in parts], axis=1)[:, order]
                    else:
                        values = by_rate[first][step]
                    mm = [
                        error_summary(
                            values[2 * si + (rate == 0.001)],
                            data["target"][fullrows],
                            data["patient"][fullrows],
                            data["site"][fullrows],
                        )[0]
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=rate,
                            parameters=PARAMS[variant],
                            numeric_bytes=4 * PARAMS[variant] + 458,
                            clean=float(np.mean([m["person_mean"] for m in mm])),
                            p90=float(np.mean([m["p90"] for m in mm])),
                            seed_metrics=mm,
                        )
                    )
            print("WE AUDIT INNER", role, variant, flush=True)
        old = old_selection["roles"][role]
        # These two controls are unchanged, previously audited WIDE candidates.
        candidates.extend(
            [old["policies"]["tiny"], next(c for c in old["candidates"] if c["variant"] == "np")]
        )
        for c in old["candidates"]:
            if c["variant"] in CAPS:
                new = next(
                    x
                    for x in candidates
                    if x["variant"] == c["variant"]
                    and x["step"] == c["step"]
                    and x["lr"] == c["lr"]
                )
                close(new, c)

        def rank(c):
            return c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0

        policies = {v: min([c for c in candidates if c["variant"] == v], key=rank) for v in CAPS}
        policies["overall"] = min(candidates, key=rank)
        close(candidates, selection["roles"][role]["candidates"])
        close(policies, selection["roles"][role]["policies"])
        counts["candidates"] += len(candidates)
        counts["choices"] += len(policies)
    assert counts == dict(
        inner_banks=108,
        inner_models=3240,
        inner_vectors=612000,
        original_controls=432,
        paired_canaries=1080,
        candidates=222,
        choices=15,
    ), counts
    return counts, maxlab, maxgpu


def audit_final(data, source, selection, result, artifacts):
    counts = dict(
        final_banks=0,
        final_models=0,
        actual_stress_vectors=0,
        unchanged_selected_aliases=0,
        exact_old_stress_controls=0,
    )
    maximum = gpu_max = 0.0
    previous = js(WIDE / "selections.json")
    entries = js(RUN / "final_sources.json")
    assert len(entries) == len(result["records"]) == 90
    seen = set()
    for rec, entry in zip(result["records"], entries, strict=True):
        for key, value in entry.items():
            assert rec[key] == value
        role, variant, kind = rec["role"], rec["variant"], rec["kind"]
        mask, held = roles(data["patient"], data["device"])[role]
        ix, query = np.flatnonzero(mask), np.flatnonzero(held)
        seed = rec["seed"]
        si = SEEDS.index(seed)
        old = None if variant == "np" else previous["roles"][role]["policies"][variant]
        if kind == "new":
            chosen = selection["roles"][role]["policies"][variant]
            assert rec["step"] == chosen["step"] and rec["lr"] == chosen["lr"]
            first = 0.0001 if chosen["lr"] == 0.001 else chosen["lr"]
            imported = chosen["lr"] in (0.0001, 0.001) and chosen["step"] in (512, 2048, 8192)
            expected = (
                wide_bank(role, variant) if imported else bank_path(role, variant, first)
            ) / f"models_{chosen['step']}.npz"
            assert rec["imported_from_wide"] == imported and rec["source_slot"] == str(
                2 * si + (chosen["lr"] == 0.001)
            )
            assert rec["unchanged_setting"] == (
                chosen["step"] == old["step"] and chosen["lr"] == old["lr"]
            )
            if not imported and (role, variant) not in seen:
                fi, fq, _, _, _ = inspect_bank(
                    data, source, role, variant, first, None, artifacts, chosen["step"]
                )
                np.testing.assert_array_equal(fi, ix)
                np.testing.assert_array_equal(fq, query)
                packed = nz(expected)
                ms = [model_from(packed, str(i)) for i in range(6)]
                ref = np.stack(
                    [direct(m, data["color"][query[:3]], data["tokens"][query[:3]]) for m in ms]
                )
                gpu_max = max(
                    gpu_max,
                    independent_gpu(ms, data["color"][query[:3]], data["tokens"][query[:3]], ref),
                )
                seen.add((role, variant))
                counts["final_banks"] += 1
        elif kind == "np":
            assert variant == "np" and rec["step"] == 0 and rec["lr"] is None
            expected = WIDE / "models" / role / f"np_s{seed}.npz"
        else:
            assert (kind == "wide" and variant in CAPS) or (kind == "tiny" and variant == "tiny")
            assert rec["step"] == old["step"] and rec["lr"] == old["lr"]
            expected = (
                WIDE
                / "models"
                / role
                / f"{variant}_r{int(old['lr'] == 0.001)}_t{old['step']}_s{seed}.npz"
            )
        assert (ROOT / rec["source_path"]) == expected and sha(expected) == rec["source_sha256"]
        p, pp = (
            RUN / "models" / role / f"{rec['name']}.npz",
            RUN / "evaluated" / role / f"{rec['name']}.npz",
        )
        assert (
            sha(p) == rec["model_sha256"]
            and sha(pp) == rec["prediction_sha256"]
            and p.stat().st_size == rec["archive_bytes"]
        )
        for path in (p, pp):
            artifacts[path.relative_to(ROOT).as_posix()] = sha(path)
        m, saved = nz(p), nz(pp)
        parent = nz(expected)
        parent = parent if rec["source_slot"] is None else model_from(parent, rec["source_slot"])
        exact(m, parent)
        verify_normalizers(m, data["color"][ix], data["target"][ix])
        if kind == "np":
            consumer = BasePredictor(m)
            assert rec["parameters"] == 643 and rec["numeric_bytes"] == 2886
        else:
            consumer = Predictor(m)
            assert (
                rec["parameters"] == PARAMS[variant]
                and rec["numeric_bytes"] == 4 * PARAMS[variant] + 458
            )
            token = data["tokens"][ix].astype(float).reshape(-1, 18)
            np.testing.assert_array_equal(m["t_mean"], token.mean(0).astype(np.float32))
            np.testing.assert_array_equal(
                m["t_std"], np.maximum(token.std(0), 1e-6).astype(np.float32)
            )
        np.testing.assert_array_equal(saved["row_indices"], query)
        x, t = data["color"][query], data["tokens"][query]
        reference = direct(m, x, t)
        np.testing.assert_allclose(reference, saved["predictions"][0], rtol=0, atol=2e-8)
        maximum = max(maximum, float(np.max(abs(reference - saved["predictions"][0]))))
        actual = []
        for dose, anchor in SETTINGS:
            xx, tt = transformed(x, dose, anchor), local_transform(t, dose, anchor)
            output = (
                np.stack([consumer(row) for row in xx])
                if kind == "np"
                else np.stack([consumer(row, patch) for row, patch in zip(xx, tt, strict=True)])
            )
            actual.append(output)
        actual = np.stack(actual)
        assert actual.shape == (33, len(query), 3)
        maximum = max(maximum, float(np.max(abs(actual - saved["predictions"]))))
        np.testing.assert_allclose(actual, saved["predictions"], rtol=0, atol=2e-8)
        close(full_metrics(actual[0], data, query), rec["metrics"])
        summaries(
            rec, actual, data["target"][query], data["patient"][query], data["device"][query], None
        )
        policies = [variant] if kind == "new" else []
        overall = selection["roles"][role]["policies"]["overall"]
        if (
            kind != "wide"
            and variant == overall["variant"]
            and rec["step"] == overall["step"]
            and rec["lr"] == overall["lr"]
        ):
            policies.append("overall")
        assert rec["policies"] == policies
        if kind != "new" or rec["unchanged_setting"]:
            name = (
                f"np_s{seed}"
                if variant == "np"
                else f"{variant}_r{int(old['lr'] == 0.001)}_t{old['step']}_s{seed}"
            )
            old_saved = nz(WIDE / "evaluated" / role / f"{name}.npz")
            np.testing.assert_array_equal(old_saved["row_indices"], query)
            np.testing.assert_array_equal(old_saved["predictions"], saved["predictions"])
            counts["exact_old_stress_controls"] += 1
            if kind == "new":
                exact(m, nz(WIDE / "models" / role / f"{name}.npz"))
                counts["unchanged_selected_aliases"] += 1
        counts["final_models"] += 1
        counts["actual_stress_vectors"] += len(query) * 33
        if seed == 43:
            print("WE AUDIT FINAL", role, kind, variant, flush=True)
    assert counts["final_models"] == 90 and counts["actual_stress_vectors"] == 1186020
    assert counts["exact_old_stress_controls"] == 54 + counts["unchanged_selected_aliases"]
    assert 0 <= counts["final_banks"] <= 12
    return counts, maximum, gpu_max


def main():
    assert not (OUT / "verification.json").exists(), "sealed WE; report verifier only"
    start = time.perf_counter()
    setup("cuda")
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    source = sha(RUN / "source_lock.json")
    check_map({**lock["sources"], **lock["input_sha256"]})
    assert result["source_lock_sha256"] == selection["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json") and result[
        "final_sources_sha256"
    ] == sha(RUN / "final_sources.json")
    assert js(RUN / "job.json")["status"] == "complete"
    data = load_data()
    artifacts = {}
    inner, im, ig = audit_inner(data, source, selection, artifacts)
    final, fm, fg = audit_final(data, source, selection, result, artifacts)
    for name in (
        "source_lock.json",
        "selections.json",
        "final_sources.json",
        "results.json",
        "job.json",
    ):
        p = RUN / name
        artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    check_map(artifacts)
    deps = {
        "scripts/chromaseed_widen_early_audit.py": sha(
            ROOT / "scripts/chromaseed_widen_early_audit.py"
        )
    }
    value = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        counts={**inner, **final},
        maximum_native_lab=max(im, fm),
        maximum_cuda_lab=max(ig, fg),
        artifact_sha256=artifacts,
        dependencies=deps,
        seconds=time.perf_counter() - start,
    )
    write_json(OUT / "audit.json", value)
    print("WE AUDIT PASSED", value["counts"], "maxLab", value["maximum_native_lab"], flush=True)


if __name__ == "__main__":
    main()
