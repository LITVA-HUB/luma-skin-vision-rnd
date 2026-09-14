"""Independent capacity-study audit; never trains or changes frozen weights."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import torch
from chromaseed_affine_audit import summaries
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS
from chromaseed_kernel_audit import js, nz
from chromaseed_local_denoise_audit import close, full_metrics, verify_normalizers
from chromaseed_long_training_run import ND, NP
from chromaseed_neural_prefix_audit import inspect_export
from chromaseed_neural_prefix_numpy import Predictor as BasePredictor
from chromaseed_perceptual_audit import balanced
from chromaseed_refine_audit import error_summary
from chromaseed_refine_train import setup
from chromaseed_widen import Predictor
from chromaseed_widen_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from skin_local_search_train import folds_for, roles, sha, write_json

SPECS = {
    "tiny": (8, 16),
    "m31": (64, 128),
    "m61": (64, 256),
    "m111": (128, 256),
    "m832": (256, 1024),
}
PARAMS = {"tiny": 1179, "m31": 30915, "m61": 60611, "m111": 110979, "m832": 832259}
SEEDS = (17, 29, 43)
RATES = (0.0001, 0.001)
TIMES = (0, 512, 2048, 8192)
SLOTS = [dict(seed=s, lr=r) for s in SEEDS for r in RATES]


def direct(m, x, t):
    """Non-BLAS explicit contraction; independent from deployed matmul path."""
    chunks = []
    for start in range(0, len(x), 32):
        xx = ((np.asarray(x[start : start + 32], np.float32) - m["x_mean"]) / m["x_std"]).astype(
            float
        )
        hidden = np.einsum("nd,dh->nh", xx, m["w0"].astype(float), optimize=False) + m["b0"]
        if "variant" in m:
            tt = (
                (np.asarray(t[start : start + 32], np.float32) - m["t_mean"]) / m["t_std"]
            ).astype(float)
            local = np.maximum(
                np.einsum("ntd,de->nte", tt, m["u"].astype(float), optimize=False) + m["e"], 0
            )
            mean = local.mean(1)
            spread = np.sqrt(np.mean((local - mean[:, None, :]) ** 2, axis=1) + 1e-6)
            pool = np.concatenate((mean, spread, np.max(local, axis=1)), axis=1)
            hidden = hidden + np.einsum("nd,dh->nh", pool, m["g"].astype(float), optimize=False)
        answer = (
            np.einsum("nh,hc->nc", np.maximum(hidden, 0), m["v0"].astype(float), optimize=False)
            + m["c0"]
        )
        chunks.append(answer * m["y_std"] + m["y_mean"])
    return np.concatenate(chunks)


def local_transform(tokens, dose, anchor):
    v = np.asarray(tokens, np.float32).astype(float)
    for j in range(18):
        v[:, :, j] += dose * ((anchor[j % 3] if j < 12 else 0) - v[:, :, j])
    return v.astype(np.float32)


def exact(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def independent_gpu(ms, x, t, expected):
    m = ms[0]
    xx = (x - m["x_mean"]) / m["x_std"]
    tt = (t - m["t_mean"]) / m["t_std"]

    def stack(key):
        return torch.tensor(np.stack([v[key] for v in ms]), device="cuda")

    with torch.no_grad():
        xx = torch.tensor(np.repeat(xx[None], 6, axis=0), device="cuda")
        tt = torch.tensor(np.repeat(tt.reshape(-1, 18)[None], 6, axis=0), device="cuda")
        local = torch.relu(torch.bmm(tt, stack("u")) + stack("e")[:, None, :]).reshape(
            6, len(x), 64, -1
        )
        mean = local.mean(2)
        pooled = torch.cat(
            (mean, ((local - mean[:, :, None, :]).square().mean(2) + 1e-6).sqrt(), local.amax(2)),
            dim=-1,
        )
        hidden = torch.relu(
            (torch.bmm(xx, stack("w0")) + stack("b0")[:, None, :]) + torch.bmm(pooled, stack("g"))
        )
        actual = (torch.bmm(hidden, stack("v0")) + stack("c0")[:, None, :]).cpu().numpy() * m[
            "y_std"
        ] + m["y_mean"]
    drift = float(np.max(abs(actual - expected)))
    assert drift <= 0.002, drift
    return drift


def inspect_bank(data, source, role, variant, fold, artifacts):
    mask, held = roles(data["patient"], data["device"])[role]
    ix, query = np.flatnonzero(mask), np.flatnonzero(held)
    if fold is not None:
        assignment = folds_for(data["patient"][ix], data["device"][ix])
        query, ix = ix[assignment == fold], ix[assignment != fold]
    path = bank_path(role, variant, fold)
    rec = js(path / "receipt.json")
    assert rec["source_lock_sha256"] == source and rec["selection_sha256"] == (
        None if fold is not None else sha(RUN / "selections.json")
    )
    assert rec["slots"] == SLOTS and rec["trajectory_count"] == 6 and rec["variant"] == variant
    assert rec["schedule_horizon"] == rec["steps"] == 8192 and rec["checkpoints"] == list(TIMES)
    assert rec["engine"] == "cuda_graph" and rec["device"] == "cuda"
    assert rec["original_rows"] == len(ix) and all(
        np.isfinite(v["minibatch_loss"]).all() for v in rec["trace"]
    )
    check_map(rec["warm_parent_sha256"])
    for name, digest in rec["files"].items():
        assert sha(path / name) == digest
        artifacts[(path / name).relative_to(ROOT).as_posix()] = digest
    artifacts[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
    rr = nz(path / "rows.npz")
    np.testing.assert_array_equal(rr["fit_rows"], ix)
    np.testing.assert_array_equal(
        rr["query_rows"], query if fold is not None else np.array([], np.int64)
    )
    assert not set(data["patient"][ix]) & set(data["patient"][query])
    token = data["tokens"][ix]
    assert hashlib.sha256(token.tobytes()).hexdigest() == rec["fit_tokens_sha256"]
    flat = token.astype(float).reshape(-1, 18)
    tm, ts = flat.mean(0).astype(np.float32), np.maximum(flat.std(0), 1e-6).astype(np.float32)
    np.testing.assert_array_equal(np.array(rec["token_mean"], np.float32), tm)
    np.testing.assert_array_equal(np.array(rec["token_std"], np.float32), ts)
    p = balanced(data["patient"][ix], data["site"][ix])
    p /= p.sum()
    sampled = np.stack(
        [np.random.default_rng(s + 9001).choice(len(ix), (8192, 64), p=p) for s in SEEDS]
    )
    assert hashlib.sha256(sampled.tobytes()).hexdigest() == rec["sampling_sha256"]
    np.testing.assert_array_equal(
        np.array(rec["final_learning_rates"], np.float32),
        np.array([s["lr"] for s in SLOTS], np.float32) * np.float32(0.1),
    )
    prior = js(NP / "selections.json")["roles"][role]["blind4"]
    j = prior["policies"]["quality"]["prefix"]
    warm = nz(path / "warm_models.npz")
    initial = nz(path / "models_0.npz")
    e, h = SPECS[variant]
    for si, seed in enumerate(SEEDS):
        w = model_from(warm, str(si))
        verify_normalizers(w, data["color"][ix], data["target"][ix])
        if fold is None:
            exact(w, nz(NP / "selected" / role / f"blind4_j{j}_s{seed}.npz"))
        else:
            oldpath = ND / "inner" / role / "blind4" / f"fold{fold}"
            oldrows = nz(oldpath / "rows.npz")
            np.testing.assert_array_equal(oldrows["fit_rows"], ix)
            np.testing.assert_array_equal(oldrows["query_rows"], query)
            raw = model_from(
                nz(oldpath / f"models_{prior['step']}.npz"), str(2 * si + prior["lr_index"])
            )
            inspect_export(raw, w, j)
        for offset in range(2):
            m = model_from(initial, str(2 * si + offset))
            Predictor(m)
            assert int(m["warm_prefix"]) == j and int(m["warm_original_k"]) == 4
            np.testing.assert_array_equal(m["w0"][:, :16], w["w0"])
            np.testing.assert_array_equal(
                m["w0"][:, 16:],
                np.random.default_rng(seed + 880003)
                .uniform(-1 / 6, 1 / 6, (36, h - 16))
                .astype(np.float32),
            )
            np.testing.assert_array_equal(m["b0"][:16], w["b0"])
            np.testing.assert_array_equal(m["v0"][:16], w["v0"])
            np.testing.assert_array_equal(m["c0"], w["c0"])
            assert (
                not np.any(m["b0"][16:])
                and not np.any(m["v0"][16:])
                and not np.any(m["g"])
                and not np.any(m["e"])
            )
            np.testing.assert_array_equal(
                m["u"],
                np.random.default_rng(seed + 770003)
                .uniform(-1 / np.sqrt(18), 1 / np.sqrt(18), (18, e))
                .astype(np.float32),
            )
            np.testing.assert_array_equal(m["t_mean"], tm)
            np.testing.assert_array_equal(m["t_std"], ts)
    return ix, query, warm, tm, ts


def audit_inner(data, source, selection, artifacts):
    counts = dict(inner_banks=0, inner_models=0, inner_vectors=0, candidates=0, choices=0)
    maximum = gpu_max = 0.0
    for role, (mask, _) in roles(data["patient"], data["device"]).items():
        allrows = np.flatnonzero(mask)
        candidates = []
        baseline = None
        for variant in SPECS:
            pieces = {s: [] for s in TIMES}
            query_parts = []
            for fold in range(3):
                fitrows, query, _, tm, ts = inspect_bank(
                    data, source, role, variant, fold, artifacts
                )
                query_parts.append(query)
                path = bank_path(role, variant, fold)
                for step in TIMES:
                    flat = nz(path / f"models_{step}.npz")
                    saved = nz(path / f"oof_{step}.npz")
                    np.testing.assert_array_equal(saved["row_indices"], query)
                    predictions = []
                    ms = []
                    for slot in range(6):
                        m = model_from(flat, str(slot))
                        ms.append(m)
                        Predictor(m)
                        verify_normalizers(m, data["color"][fitrows], data["target"][fitrows])
                        np.testing.assert_array_equal(m["t_mean"], tm)
                        np.testing.assert_array_equal(m["t_std"], ts)
                        output = direct(m, data["color"][query], data["tokens"][query])
                        maximum = max(
                            maximum, float(np.max(abs(output - saved["predictions"][slot])))
                        )
                        np.testing.assert_allclose(
                            output, saved["predictions"][slot], rtol=0, atol=2e-8
                        )
                        predictions.append(output)
                        counts["inner_models"] += 1
                        counts["inner_vectors"] += len(query)
                    pieces[step].append(np.stack(predictions))
                    if step == 8192:
                        gpu_max = max(
                            gpu_max,
                            independent_gpu(
                                ms,
                                data["color"][query[:3]],
                                data["tokens"][query[:3]],
                                np.stack(predictions)[:, :3],
                            ),
                        )
                counts["inner_banks"] += 1
            rows = np.concatenate(query_parts)
            order = np.argsort(rows)
            np.testing.assert_array_equal(rows[order], allrows)
            values = {s: np.concatenate(pieces[s], axis=1)[:, order] for s in TIMES}
            if baseline is None:
                baseline = values[0].copy()
            else:
                np.testing.assert_allclose(values[0], baseline, rtol=0, atol=2e-8)
            for step in TIMES[1:]:
                for li, lr in enumerate(RATES):
                    mm = [
                        error_summary(
                            values[step][2 * si + li],
                            data["target"][allrows],
                            data["patient"][allrows],
                            data["site"][allrows],
                        )[0]
                        for si in range(3)
                    ]
                    candidates.append(
                        dict(
                            variant=variant,
                            step=step,
                            lr=lr,
                            parameters=PARAMS[variant],
                            numeric_bytes=4 * PARAMS[variant] + 458,
                            clean=float(np.mean([v["person_mean"] for v in mm])),
                            p90=float(np.mean([v["p90"] for v in mm])),
                            seed_metrics=mm,
                        )
                    )
            print("WIDE AUDIT INNER", role, variant, flush=True)
        mm = [
            error_summary(
                baseline[2 * si],
                data["target"][allrows],
                data["patient"][allrows],
                data["site"][allrows],
            )[0]
            for si in range(3)
        ]
        candidates.append(
            dict(
                variant="np",
                step=0,
                lr=None,
                parameters=643,
                numeric_bytes=2886,
                clean=float(np.mean([v["person_mean"] for v in mm])),
                p90=float(np.mean([v["p90"] for v in mm])),
                seed_metrics=mm,
            )
        )
        expected = selection["roles"][role]
        close(candidates, expected["candidates"])

        def rank(c):
            return c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0

        policies = {v: min([c for c in candidates if c["variant"] == v], key=rank) for v in SPECS}
        policies["overall"] = min(candidates, key=rank)
        close(policies, expected["policies"])
        counts["candidates"] += len(candidates)
        counts["choices"] += len(policies)
    assert counts == dict(
        inner_banks=45, inner_models=1080, inner_vectors=204000, candidates=93, choices=18
    ), counts
    return counts, maximum, gpu_max


def audit_final(data, source, selection, result, artifacts):
    counts = dict(
        final_banks=0, final_models=0, clean_vectors=0, stress_models=0, actual_stress_vectors=0
    )
    maximum = gpu_max = 0.0
    for role in roles(data["patient"], data["device"]):
        banks = {}
        for variant in SPECS:
            ix, query, warm, tm, ts = inspect_bank(data, source, role, variant, None, artifacts)
            banks[variant] = dict(ix=ix, query=query, warm=warm, tm=tm, ts=ts)
            packed = nz(bank_path(role, variant) / "models_8192.npz")
            ms = [model_from(packed, str(i)) for i in range(6)]
            output = np.stack(
                [direct(m, data["color"][query[:3]], data["tokens"][query[:3]]) for m in ms]
            )
            gpu_max = max(
                gpu_max,
                independent_gpu(ms, data["color"][query[:3]], data["tokens"][query[:3]], output),
            )
            counts["final_banks"] += 1
        for rec in [r for r in result["records"] if r["role"] == role]:
            variant, name = rec["variant"], rec["name"]
            p, pp = RUN / "models" / role / f"{name}.npz", RUN / "evaluated" / role / f"{name}.npz"
            assert (
                sha(p) == rec["model_sha256"]
                and sha(pp) == rec["prediction_sha256"]
                and p.stat().st_size == rec["archive_bytes"]
            )
            for file in (p, pp):
                artifacts[file.relative_to(ROOT).as_posix()] = sha(file)
            m, saved = nz(p), nz(pp)
            ix, query = banks["tiny"]["ix"], banks["tiny"]["query"]
            np.testing.assert_array_equal(saved["row_indices"], query)
            x, t = data["color"][query], data["tokens"][query]
            if variant == "np":
                assert rec["step"] == 0 and rec["lr"] is None and not rec["baseline_alias"]
                exact(m, model_from(banks["tiny"]["warm"], str(SEEDS.index(rec["seed"]))))
                expected_policies = (
                    ["overall"]
                    if selection["roles"][role]["policies"]["overall"]["variant"] == "np"
                    else []
                )
                consumer = BasePredictor(m)
                assert rec["parameters"] == 643 and rec["numeric_bytes"] == 2886
            else:
                slot = 2 * SEEDS.index(rec["seed"]) + RATES.index(rec["lr"])
                original = model_from(
                    nz(bank_path(role, variant) / f"models_{rec['step']}.npz"), str(slot)
                )
                exact(m, original)
                consumer = Predictor(m)
                verify_normalizers(m, data["color"][ix], data["target"][ix])
                np.testing.assert_array_equal(m["t_mean"], banks[variant]["tm"])
                np.testing.assert_array_equal(m["t_std"], banks[variant]["ts"])
                assert (
                    rec["parameters"] == PARAMS[variant]
                    and rec["numeric_bytes"] == 4 * PARAMS[variant] + 458
                )
                expected_policies = [
                    k
                    for k, c in selection["roles"][role]["policies"].items()
                    if c["variant"] == variant and c["step"] == rec["step"] and c["lr"] == rec["lr"]
                ]
                assert (
                    rec["baseline_alias"] == (rec["step"] == 0)
                    and rec["permutation_max_lab"] <= 2e-8
                )
                permuted = direct(m, x[:3], t[:3, ::-1].copy())
                np.testing.assert_allclose(permuted, direct(m, x[:3], t[:3]), rtol=0, atol=2e-8)
            assert rec["policies"] == expected_policies
            selected = variant == "np" or variant in expected_policies
            assert rec["stress_evaluated"] == selected
            assert saved["predictions"].shape == ((33 if selected else 1), len(query), 3)
            reference = direct(m, x, t)
            maximum = max(maximum, float(np.max(abs(reference - saved["predictions"][0]))))
            np.testing.assert_allclose(reference, saved["predictions"][0], rtol=0, atol=2e-8)
            close(full_metrics(reference, data, query), rec["metrics"])
            counts["clean_vectors"] += len(query)
            counts["final_models"] += 1
            if selected:
                actual = []
                for dose, anchor in SETTINGS:
                    xx = transformed(x, dose, anchor)
                    tt = local_transform(t, dose, anchor)
                    if variant == "np":
                        pred = np.stack([consumer(row) for row in xx])
                    else:
                        pred = np.stack(
                            [consumer(row, patch) for row, patch in zip(xx, tt, strict=True)]
                        )
                    actual.append(pred)
                actual = np.stack(actual)
                maximum = max(maximum, float(np.max(abs(actual - saved["predictions"]))))
                np.testing.assert_allclose(actual, saved["predictions"], rtol=0, atol=2e-8)
                summaries(
                    rec,
                    actual,
                    data["target"][query],
                    data["patient"][query],
                    data["device"][query],
                    None,
                )
                counts["actual_stress_vectors"] += actual.shape[0] * actual.shape[1]
                counts["stress_models"] += 1
        print("WIDE AUDIT FINAL", role, flush=True)
    assert counts == dict(
        final_banks=15,
        final_models=369,
        clean_vectors=147354,
        stress_models=54,
        actual_stress_vectors=711612,
    ), counts
    return counts, maximum, gpu_max


def main():
    assert not (OUT / "verification.json").exists(), "sealed audit; read-only verifier only"
    start = time.perf_counter()
    setup("cuda")
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    source = sha(RUN / "source_lock.json")
    check_map({**lock["sources"], **lock["input_sha256"]})
    assert result["source_lock_sha256"] == selection["source_lock_sha256"] == source
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    assert js(RUN / "job.json")["status"] == "complete"
    artifacts = {}
    data = load_data()
    inner, im, ig = audit_inner(data, source, selection, artifacts)
    final, fm, fg = audit_final(data, source, selection, result, artifacts)
    for name in ("source_lock.json", "selections.json", "results.json", "job.json"):
        p = RUN / name
        artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    check_map(artifacts)
    deps = {"scripts/chromaseed_widen_audit.py": sha(ROOT / "scripts/chromaseed_widen_audit.py")}
    value = dict(
        passed=True,
        source_lock_sha256=source,
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        counts={**inner, **final},
        max_consumer_native_lab=max(im, fm),
        max_cuda_native_lab=max(ig, fg),
        artifact_sha256=artifacts,
        dependencies=deps,
        seconds=time.perf_counter() - start,
        limitations="Reused original TRAIN roles; P8 audit still unfinished; no independent phone-face evidence",
    )
    write_json(OUT / "audit.json", value)
    print(
        "WIDE AUDIT PASSED", value["counts"], "maxLab", value["max_consumer_native_lab"], flush=True
    )


if __name__ == "__main__":
    main()
