"""Independent checks of every saved NS model, choice and actual consumer."""

from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter
from pathlib import Path

import numpy as np
from chromaseed_affine_audit import exact, summaries
from chromaseed_feature_groups_audit import compare
from chromaseed_gate_stability_audit import transformed
from chromaseed_gated_audit import model_from
from chromaseed_gaussian_audit import SETTINGS, actual_consumer, direct, row_hash, scoring
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, folds_for, roles, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_shrinkage_v1"
NR = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_shrinkage_v1"
BASES = (
    "random",
    "adam_e1",
    "adam_e4",
    "adam_e16",
    "tagi_full3_e1",
    "tagi_full3_e4",
    "tagi_full3_e16",
)
GROUPS, FAMILIES, SEEDS, ALPHAS = (
    ("raw36", "mean3"),
    ("norm", "perceptual"),
    (17, 29, 43),
    (0.1, 1.0, 10.0, 100.0, 1000.0),
)


def ident(family, basis, group, seed, ai):
    return f"{family}_{basis}_{group}_s{seed}_a{ai}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert not (OUT / "verification.json").exists(), "sealed audit is read-only"
    started = time.perf_counter()
    lock, selection, result = (
        js(RUN / f) for f in ("source_lock.json", "selections.json", "results.json")
    )
    assert sha(args.cache) == CACHE_HASH
    for p, h in {**lock["sources"], **lock["input_sha256"]}.items():
        assert sha(ROOT / p) == h, p
    assert (
        result["source_lock_sha256"]
        == selection["source_lock_sha256"]
        == sha(RUN / "source_lock.json")
    )
    assert result["selection_sha256"] == sha(RUN / "selections.json")
    data = nz(args.cache)
    checks, maxima = (
        Counter(),
        dict(qr_fit=0.0, qr_query=0.0, oof=0.0, direct=0.0, consumer=0.0, qr_stress=0.0),
    )
    bound, errors = {}, {}
    heads = {
        ident(f, b, g, s, i)
        for f, b, g, s, i in itertools.product(FAMILIES, BASES, GROUPS, SEEDS, range(5))
    }
    for role, (fit_mask, held_mask) in roles(data["patient"], data["device"]).items():
        outer, held = np.flatnonzero(fit_mask), np.flatnonzero(held_mask)
        folds = folds_for(data["patient"][outer], data["device"][outer])
        merged = {}
        for fold in (*range(3), None):
            stage, sub = ("final", "bank") if fold is None else ("inner", f"fold{fold}")
            fit = outer if fold is None else outer[folds != fold]
            query = None if fold is None else outer[folds == fold]
            path, previous = RUN / stage / role / sub, NR / stage / role / sub
            rec, old = js(path / "receipt.json"), js(previous / "receipt.json")
            assert rec["source_lock_sha256"] == sha(RUN / "source_lock.json")
            assert rec["selection_sha256"] == (
                None if fold is not None else sha(RUN / "selections.json")
            )
            assert rec["fit_rows_sha256"] == old["fit_rows_sha256"] == row_hash(fit)
            assert rec["query_rows_sha256"] == (None if query is None else row_hash(query))
            for name, h in rec["files"].items():
                assert sha(path / name) == h
                bound[(path / name).relative_to(ROOT).as_posix()] = h
            bound[(path / "receipt.json").relative_to(ROOT).as_posix()] = sha(path / "receipt.json")
            references = {n for n, m in old["models"].items() if m["family"] not in FAMILIES}
            assert len(references) == 43 and set(rec["models"]) == heads | references
            models, qrs, originals, bases = (
                nz(p)
                for p in (
                    path / "models.npz",
                    path / "qr_models.npz",
                    previous / "models.npz",
                    previous / "bases.npz",
                )
            )
            oof = None if query is None else nz(path / "oof.npz")
            if query is not None:
                np.testing.assert_array_equal(oof["row_indices"], query)
                assert not set(data["patient"][fit]) & set(data["patient"][query])
            for name, meta in rec["models"].items():
                model = model_from(models, name)
                assert sum(v.nbytes for v in model.values()) == meta["numeric_bytes"]
                checks["bank_models"] += 1
                if name in old["models"]:
                    exact(model, model_from(originals, name))
                    checks["exact_NR_models"] += 1
                if name in heads:
                    base = model_from(bases, meta["base_name"])
                    for field in set(base) - {"w2", "b2"}:
                        np.testing.assert_array_equal(model[field], base[field])
                    assert meta["alpha"] == ALPHAS[meta["alpha_index"]]
                    qr = model_from(qrs, name)
                    df = float(
                        np.max(
                            abs(direct(qr, data["color"][fit]) - direct(model, data["color"][fit]))
                        )
                    )
                    assert df <= 0.001
                    maxima["qr_fit"] = max(maxima["qr_fit"], df)
                    checks["saved_QR_heads"] += 1
                    if query is not None:
                        dq = float(
                            np.max(
                                abs(
                                    direct(qr, data["color"][query])
                                    - direct(model, data["color"][query])
                                )
                            )
                        )
                        assert dq <= 0.001
                        maxima["qr_query"] = max(maxima["qr_query"], dq)
                if query is not None:
                    output = direct(model, data["color"][query])
                    diff = float(np.max(abs(output - oof["pred__" + name])))
                    assert diff <= 2e-8
                    maxima["oof"] = max(maxima["oof"], diff)
                    if name not in merged:
                        merged[name] = np.empty((len(outer), 3))
                    merged[name][folds == fold] = output
                    checks["oof_rows"] += len(query)
            checks["banks"] += 1
        for group, family in itertools.product(GROUPS, FAMILIES):
            entry = selection["roles"][role][group][family]
            selected = []
            for basis in BASES:
                candidates = []
                for ai, alpha in enumerate(ALPHAS):
                    ss = [
                        scoring(
                            merged[ident(family, basis, group, seed, ai)],
                            data["target"][outer],
                            data["patient"][outer],
                        )
                        for seed in SEEDS
                    ]
                    item = dict(
                        group=group,
                        family=family,
                        basis=basis,
                        alpha=alpha,
                        alpha_index=ai,
                        clean=float(np.mean([s["clean"] for s in ss])),
                        p90=float(np.mean([s["p90"] for s in ss])),
                        seed_scores=ss,
                    )
                    candidates.append(item)
                    checks["candidate_scores"] += 1
                compare(entry["bases"][basis]["candidates"], candidates)
                winner = min(candidates, key=lambda c: (c["clean"], c["p90"], c["alpha_index"]))
                compare(entry["bases"][basis]["selected"], winner)
                selected.append(winner)
                checks["choices"] += 1
            policy = min(
                selected,
                key=lambda c: (
                    c["clean"],
                    c["p90"],
                    0 if c["basis"] == "random" else int(c["basis"].split("_e")[-1]),
                    BASES.index(c["basis"]),
                ),
            )
            compare(entry["policy"], policy)
            checks["policies"] += 1
        refs = {
            n: scoring(p, data["target"][outer], data["patient"][outer])
            for n, p in merged.items()
            if n in references
        }
        compare(selection["references"][role], refs)
        x, y, person, site, camera = (
            data[k][held] for k in ("color", "target", "patient", "site", "device")
        )
        xx = [transformed(x, t, a) for t, a in SETTINGS]
        final_models = nz(RUN / "final" / role / "bank" / "models.npz")
        final_qrs = nz(RUN / "final" / role / "bank" / "qr_models.npz")
        expected = set(references)
        for family, basis, group, seed in itertools.product(FAMILIES, BASES, GROUPS, SEEDS):
            ai = selection["roles"][role][group][family]["bases"][basis]["selected"]["alpha_index"]
            expected.add(ident(family, basis, group, seed, ai))
        records = [r for r in result["records"] if r["role"] == role]
        assert len(records) == 127 and {r["name"] for r in records} == expected
        for rec in records:
            name = rec["name"]
            mp, pp = (
                RUN / "selected" / role / f"{name}.npz",
                RUN / "evaluated" / role / f"{name}.npz",
            )
            assert sha(mp) == rec["model_sha256"] and sha(pp) == rec["prediction_sha256"]
            bound[mp.relative_to(ROOT).as_posix()] = sha(mp)
            bound[pp.relative_to(ROOT).as_posix()] = sha(pp)
            model, saved = nz(mp), nz(pp)
            exact(model, model_from(final_models, name))
            np.testing.assert_array_equal(saved["row_indices"], held)
            assert saved["prediction"].shape == (33, len(held), 3)
            output = np.stack([direct(model, z) for z in xx])
            call = actual_consumer(model)
            actual = np.array([[call(row) for row in z] for z in xx])
            for key, pred in (("direct", output), ("consumer", actual)):
                drift = float(np.max(abs(pred - saved["prediction"])))
                maxima[key] = max(maxima[key], drift)
                assert drift <= 2e-8
            if rec["family"] in FAMILIES:
                qr = model_from(final_qrs, name)
                drift = float(
                    np.max(abs(np.stack([direct(qr, z) for z in xx]) - saved["prediction"]))
                )
                assert drift <= 0.001
                maxima["qr_stress"] = max(maxima["qr_stress"], drift)
                policy = selection["roles"][role][rec["group"]][rec["family"]]["policy"]
                assert rec["policy_selected"] == (policy["basis"] == rec["basis"])
            clean, per_person = error_summary(output[0], y, person, site)
            compare(rec["metrics"], clean)
            summaries(rec, output, y, person, camera, None)
            errors[(role, rec["group"], rec["family"], rec["basis"], rec["seed"])] = per_person
            checks["selected_models"] += 1
            checks["final_rows"] += 33 * len(held)
            checks["transform_cases"] += 33
            checks["dose_cases"] += 4
        print(f"AUDIT {role}: all banks/choices/127 consumers x33 transforms passed", flush=True)
    expected = dict(
        banks=12,
        bank_models=5556,
        exact_NR_models=3540,
        saved_QR_heads=5040,
        oof_rows=787100,
        candidate_scores=420,
        choices=84,
        policies=12,
        selected_models=381,
        final_rows=5020818,
        transform_cases=12573,
        dose_cases=1524,
    )
    assert dict(checks) == expected, (dict(checks), expected)
    old_result, old_sel = js(NR / "results.json"), js(NR / "selections.json")
    old_errors = {}
    for rec in old_result["records"]:
        path = NR / "evaluated" / rec["role"] / f"{rec['name']}.npz"
        assert sha(path) == rec["prediction_sha256"]
        bound[path.relative_to(ROOT).as_posix()] = sha(path)
        saved = nz(path)
        rows = saved["row_indices"]
        _, pp = error_summary(
            saved["prediction"][0], data["target"][rows], data["patient"][rows], data["site"][rows]
        )
        old_errors[(rec["role"], rec["group"], rec["family"], rec["basis"], rec["seed"])] = pp

    def average(table, role, group, family, basis):
        return np.mean([table[(role, group, family, basis, s)] for s in SEEDS], axis=0)

    paired = []
    rng = np.random.default_rng(771031)
    for role in selection["roles"]:
        n = len(average(errors, role, "raw36", "norm", "random"))
        draws = rng.integers(0, n, size=(20000, n))
        for group, family, basis in itertools.product(GROUPS, FAMILIES, BASES):
            cur = average(errors, role, group, family, basis)
            for kind, prev in (
                ("head_vs_NR", average(old_errors, role, group, family, basis)),
                ("head_vs_FG", average(errors, role, group, "fg_norm_static", "reference")),
            ):
                d = cur - prev
                paired.append(
                    dict(
                        role=role,
                        group=group,
                        family=family,
                        basis=basis,
                        kind=kind,
                        mean_difference=float(d.mean()),
                        people_improved=int(np.sum(d < 0)),
                        people=n,
                        descriptive_95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist(),
                    )
                )
        for group, family in itertools.product(GROUPS, FAMILIES):
            b = selection["roles"][role][group][family]["policy"]["basis"]
            ob = old_sel["roles"][role][group][family]["policy"]["basis"]
            cur = average(errors, role, group, family, b)
            for kind, prev in (
                ("policy_vs_NR", average(old_errors, role, group, family, ob)),
                ("policy_vs_FG", average(errors, role, group, "fg_norm_static", "reference")),
            ):
                d = cur - prev
                paired.append(
                    dict(
                        role=role,
                        group=group,
                        family=family,
                        basis=b,
                        kind=kind,
                        mean_difference=float(d.mean()),
                        people_improved=int(np.sum(d < 0)),
                        people=n,
                        descriptive_95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist(),
                    )
                )
    deps = (
        "scripts/chromaseed_affine_audit.py",
        "scripts/chromaseed_feature_groups_audit.py",
        "scripts/chromaseed_gate_stability_audit.py",
        "scripts/chromaseed_gated_audit.py",
        "scripts/chromaseed_gaussian_audit.py",
        "scripts/chromaseed_refine_audit.py",
        "scripts/chromaseed_neural_readout_reference.py",
        "scripts/chromaseed_perceptual_reference.py",
    )
    write_json(
        OUT / "audit.json",
        dict(
            passed=True,
            source_lock_sha256=sha(RUN / "source_lock.json"),
            selection_sha256=sha(RUN / "selections.json"),
            results_sha256=sha(RUN / "results.json"),
            checks=dict(checks),
            maxima=maxima,
            paired=paired,
            audit_source_sha256=sha(Path(__file__)),
            dependencies={p: sha(ROOT / p) for p in deps},
            artifact_sha256=bound,
            wall_seconds=time.perf_counter() - started,
            qr_scope="5040 independent augmented-QR fits executed in frozen primary; saved QR/primary payloads independently re-evaluated here; no hidden network retraining",
        ),
    )
    print(
        dict(
            passed=True,
            checks=dict(checks),
            maxima=maxima,
            wall_seconds=time.perf_counter() - started,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
