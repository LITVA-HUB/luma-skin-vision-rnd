"""Seal TG once and verify the complete canonical evidence chain read-only later."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from chromaseed_gate_stability_audit import close
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_gaussian_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_gaussian_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gaussian-2026-09-13.md"
SOURCE = "c5f2d9f4a4a9111b4da1a5b238c0d48ecd6c851a808efb272d66e5c92cebca62"
SELECTION = "22a8c9543614ced61fb1809e95fe3ec31c12710446db0b1261080d89b47bab9b"
RESULT = "2ae1278362c022d86a31102b88db86af03cec236baec193e32274ce36c437e8f"
FG_RECEIPT = "459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908"
METHODS = ("adam", "tagi_diag", "tagi_full3")


def matched(records, row):
    return [r for r in records if all(r[k] == row[k] for k in ("role", "method", "group"))]


def aggregate_checks(summary, result, runtime, selection, audit):
    for filename, field, n in (
        ("selected.csv", "rows", 27),
        ("learning_curves.csv", "curves", 81),
        ("all_doses.csv", "doses", 108),
        ("inner_candidates.csv", "candidates", 216),
        ("paired.csv", "paired", 30),
        ("variance_checkpoints.csv", "variance_checkpoints", 2160),
    ):
        with (OUT / filename).open(encoding="utf-8-sig", newline="") as f:
            actual = list(csv.DictReader(f))
        assert len(actual) == len(summary[field]) == n
        for a, b in zip(actual, summary[field], strict=True):
            assert a == {k: "" if v is None else str(v) for k, v in b.items()}
    assert len({(r["role"], r["method"], r["group"]) for r in summary["rows"]}) == 27
    for row in summary["rows"]:
        rr, tr, fr = (
            matched(records, row)
            for records in (
                result["records"],
                runtime["records"],
                runtime["standalone_fit_records"],
            )
        )
        assert row["seeds"] == len(rr) == len(tr) == (1 if row["method"] == "constant" else 3)
        for field in ("person_mean", "image_mean", "p90"):
            close(row[field], np.mean([r["metrics"][field] for r in rr]), 1e-12)
        close(
            row["stress4"],
            np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr]),
            1e-12,
        )
        close(row["median_us"], np.median([r["numpy_only"]["median_us"] for r in tr]), 1e-12)
        close(row["maximum_seed_p95_us"], max(r["numpy_only"]["p95_us"] for r in tr), 1e-12)
        for field in ("numeric_bytes", "archive_bytes"):
            for op in (min, max):
                assert row[field + "_" + op.__name__] == op(r[field] for r in rr)
        for op in (min, max):
            assert row["cached_array_bytes_" + op.__name__] == op(
                r["numpy_only"]["cached_array_bytes"] for r in tr
            )
        for field in ("parameter", "epoch"):
            assert all(row[field] == r[field] for r in rr)
        if row["method"] in METHODS:
            chosen = selection["roles"][row["role"]][row["method"]][row["group"]]["selected"]
            for field in ("parameter", "epoch"):
                assert row[field] == chosen[field]
            close(row["inner_clean"], chosen["clean"], 1e-12)
            close(row["inner_p90"], chosen["p90"], 1e-12)
        else:
            assert row["inner_clean"] is row["inner_p90"] is None
        if fr:
            assert len(fr) == 1 and fr[0]["seed"] == 17
            assert fr[0]["parameter"] == row["parameter"] and fr[0]["epoch"] == row["epoch"]
            assert row["training_parameter_state_bytes"] == fr[0]["training_parameter_state_bytes"]
            close(row["full_fit_ms"], fr[0]["median_seconds"] * 1000, 1e-12)
        else:
            assert row["full_fit_ms"] is row["training_parameter_state_bytes"] is None
    for row in summary["curves"]:
        rr = [r for r in matched(result["curve_records"], row) if r["epoch"] == row["epoch"]]
        assert row["seeds"] == len(rr) == (1 if row["method"] == "constant" else 3)
        assert all(r["parameter"] == row["parameter"] for r in rr)
        assert row["selected_for_stress"] == all(r["selected_for_stress"] for r in rr)
        for field in ("person_mean", "p90"):
            close(row[field], np.mean([r["metrics"][field] for r in rr]), 1e-12)
    for row in summary["doses"]:
        index = (1 / 255, 4 / 255, 16 / 255, 64 / 255).index(row["dose"])
        close(
            row["worst_person_error"],
            np.mean(
                [
                    r["doses"][index]["worst_error"]["person_mean"]
                    for r in matched(result["records"], row)
                ]
            ),
            1e-12,
        )
    for row in summary["candidates"]:
        entry = selection["roles"][row["role"]][row["method"]][row["group"]]
        candidate = next(
            c
            for c in entry["candidates"]
            if c["parameter"] == row["parameter"] and c["epoch"] == row["epoch"]
        )
        assert {k: v for k, v in row.items() if k not in ("role", "selected")} == {
            k: v for k, v in candidate.items() if k != "seed_scores"
        }
        assert row["selected"] == (
            row["parameter"] == entry["selected"]["parameter"]
            and row["epoch"] == entry["selected"]["epoch"]
        )
    assert sum(r["selected"] for r in summary["candidates"]) == 18
    for row, p in zip(summary["paired"], audit["paired"], strict=True):
        assert {
            k: v for k, v in row.items() if k not in ("descriptive_95_low", "descriptive_95_high")
        } == {k: v for k, v in p.items() if k != "fixed_prediction_person_bootstrap_95"}
        assert [row["descriptive_95_low"], row["descriptive_95_high"]] == p[
            "fixed_prediction_person_bootstrap_95"
        ]
        selected = matched(summary["rows"], row)[0]
        control = matched(summary["rows"], dict(row, method=row["control"]))[0]
        close(row["mean_difference"], selected["person_mean"] - control["person_mean"], 1e-12)
    flattened = [
        dict(**{k: v for k, v in d.items() if k != "checkpoints"}, **c)
        for d in audit["variance_diagnostics"]
        for c in d["checkpoints"]
    ]
    assert flattened == summary["variance_checkpoints"]
    for row in summary["variance_diagnostics"]:
        dd = [d for d in audit["variance_diagnostics"] if d["method"] == row["method"]]
        last = [d["checkpoints"][-1] for d in dd]
        assert row["trajectories"] == len(dd) == 180
        assert row["trajectories_with_floor"] == sum(
            c["floored_variance_updates"] > 0 for c in last
        )
        for k in ("floored_variance_updates", "negative_variance_updates"):
            assert row[k] == sum(c[k] for c in last)
        for k in ("minimum_pre_floor_variance", "minimum_variance"):
            assert row[k] == (None if row["method"] == "adam" else min(c[k] for c in last))
    counts = dict(
        new_vs_FG=sum(p["control"] == "fg_norm_static" for p in audit["paired"]),
        adverse_vs_FG=sum(
            p["control"] == "fg_norm_static" and p["mean_difference"] > 0 for p in audit["paired"]
        ),
        gaussian_vs_Adam=sum(p["control"] == "adam" for p in audit["paired"]),
        adverse_vs_Adam=sum(
            p["control"] == "adam" and p["mean_difference"] > 0 for p in audit["paired"]
        ),
        selected_epoch64=sum(r["epoch"] == 64 for r in summary["rows"]),
    )
    assert (
        counts
        == summary["comparison_counts"]
        == dict(
            new_vs_FG=18,
            adverse_vs_FG=16,
            gaussian_vs_Adam=12,
            adverse_vs_Adam=7,
            selected_epoch64=17,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == SOURCE and sha(RUN / "selections.json") == SELECTION
    assert sha(RUN / "results.json") == RESULT
    lock, selection, result = (
        read(RUN / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    assert check_map(ROOT, lock["sources"]) == 79 and check_map(ROOT, lock["input_sha256"]) == 42
    pp = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1/verification.json"
    assert sha(pp) == FG_RECEIPT
    prior = read(pp)
    check_map(ROOT, prior["postprocess_sources"])
    parent = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_feature_groups_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent and sha(pp) == FG_RECEIPT
    source_checks = [
        *prior["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=79,
            unique_source_paths_checked=79,
        ),
    ]
    assert sum(s["source_entries_checked"] for s in source_checks) == 693
    assert sum(s["unique_source_paths_checked"] for s in source_checks) == 691
    assert prior["diagnostic_input_bindings_checked"] + 42 == 847
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for obj in (result, audit, runtime, summary):
        assert obj["source_lock_sha256"] == SOURCE and obj["selection_sha256"] == SELECTION
    for obj in (audit, runtime, summary):
        assert obj["results_sha256"] == RESULT
    for obj, field, file in (
        (audit, "audit_source_sha256", "audit"),
        (runtime, "runtime_source_sha256", "runtime"),
        (summary, "report_source_sha256", "report"),
    ):
        assert obj[field] == sha(ROOT / f"scripts/chromaseed_gaussian_{file}.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    expected = dict(
        banks=12,
        stored_models=2244,
        new_checkpoints=2160,
        exact_FG_models=84,
        trajectories=540,
        trajectory_checkpoints=2160,
        example_updates=13708800,
        oof_rows=379100,
        candidate_scores=216,
        choices=18,
        clean_models=237,
        clean_rows=94642,
        selected_models=75,
        selected_rows=988350,
        transform_cases=2475,
        dose_cases=300,
        scalar_refit_trajectories=54,
        scalar_refit_checkpoints=216,
        scalar_refit_clean_rows=86256,
        scalar_refit_stress_rows=711612,
    )
    assert audit["checks"] == expected
    for key in ("oof_prediction", "direct_prediction", "consumer_prediction"):
        assert audit["maxima"][key] <= 2e-8
    assert audit["maxima"]["refit_parameter_absolute"] == 0
    assert audit["maxima"]["refit_prediction"] <= 0.001
    assert len(runtime["records"]) == 75 and len(runtime["standalone_fit_records"]) == 24
    assert runtime["complete_fits_including_warmups"] == 72 and runtime["hardware"]["threads"] == 1
    assert runtime["new_network_complete_fits"] == 54 and runtime["FG_complete_fits"] == 18
    assert len({(r["role"], r["name"]) for r in runtime["records"]}) == 75
    assert (
        len({(r["role"], r["method"], r["group"]) for r in runtime["standalone_fit_records"]}) == 24
    )
    for r in runtime["records"]:
        assert r["numpy_only"]["query_count"] == r["numpy_only"]["verification_rows"] * 3
        assert r["numpy_only"]["max_lab_drift"] <= 2e-8
    for r in runtime["standalone_fit_records"]:
        assert len(r["seconds"]) == 2 and min(r["seconds"]) > 0
        assert r["complete_fits_including_warmup"] == r["exact_payload_repeats"] == 3
        assert r["max_parameter_absolute"] == 0 and r["max_lab_drift"] <= 0.001
        assert r["n_fit_rows"] == dict(mixed=734, slr_to_ipod=323, ipod_to_slr=643)[r["role"]]
        close(r["median_seconds"], np.median(r["seconds"]), 1e-12)
    aggregate_checks(summary, result, runtime, selection, audit)
    files = {}
    for rec in result["curve_records"]:
        for area, field in (("models", "model_sha256"), ("curves", "clean_prediction_sha256")):
            p = RUN / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            files[p.relative_to(RUN).as_posix()] = rec[field]
    for rec in result["records"]:
        p = RUN / "evaluated" / rec["role"] / f"{rec['name']}.npz"
        assert sha(p) == rec["prediction_sha256"]
        files[p.relative_to(RUN).as_posix()] = rec["prediction_sha256"]
    assert len(files) == 549
    for area, n in (("models", 237), ("curves", 237), ("evaluated", 75)):
        assert len(list((RUN / area).glob("*/*.npz"))) == n
    receipts = sorted(RUN.glob("inner/*/fold*/receipt.json")) + sorted(
        RUN.glob("final/*/bank/receipt.json")
    )
    bank_files = {}
    for p in receipts:
        rec = read(p)
        assert rec["source_lock_sha256"] == SOURCE
        assert len(rec["models"]) == (223 if "inner" in p.parts else 79)
        assert len(rec["trajectories"]) == (54 if "inner" in p.parts else 18)
        check_map(p.parent, rec["files"])
        bank_files.update(
            {(p.parent / n).relative_to(RUN).as_posix(): v for n, v in rec["files"].items()}
        )
    assert len(receipts) == 12 and len(bank_files) == 21
    assert read(RUN / "inner_complete.json") == dict(
        source_lock_sha256=SOURCE, selection_sha256=None, banks=9, records=2007, trajectories=486
    )
    assert read(RUN / "final_complete.json") == dict(
        source_lock_sha256=SOURCE, selection_sha256=SELECTION, banks=3, records=237, trajectories=54
    )
    assert read(RUN / "workflow.json")["exit_code"] == 0
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(gaussian|feature_groups|crossfit|hybrid|projection|affine)_(train|audit|runtime)\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    scripts = sorted(ROOT.glob("scripts/chromaseed_gaussian*.py"))
    tests = [ROOT / "tests/test_chromaseed_gaussian.py"]
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    plan = ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-gaussian.md"
    extra = [
        ROOT / "docs/research/chromaseed_gaussian_next_decision.md",
        ROOT / "docs/architecture/chromaseed_gaussian_model_card.md",
    ]
    links = 0
    for p in [*OUT.glob("*.md"), *extra, plan, SHORTCUT]:
        for target in re.findall(r"\]\(([^\n)]+)\)", p.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-z]+://", target):
                continue
            dest = (p.parent / target).resolve()
            assert dest == (OUT / "verification.json").resolve() or dest.exists(), (p, target)
            links += 1
    vp = OUT / "verification.json"
    if vp.exists():
        previous = read(vp)
        assert previous["passed"] and previous["source_lock_sha256"] == SOURCE
        for field in (
            "artifact_sha256",
            "postprocess_sources",
            "tested_sources",
            "verifier_dependency_sha256",
        ):
            check_map(ROOT, previous[field])
        check_map(RUN, previous["run_marker_and_receipt_sha256"])
        assert previous["shortcut_sha256"] == sha(SHORTCUT)
        assert read(RUN / "progress.json")["verification_sha256"] == sha(vp)
        print(dict(passed=True, mode="read-only existing receipt", verification_sha256=sha(vp)))
        return
    test_output = command([sys.executable, "-m", "pytest", *map(str, tests), "-q"])
    assert re.search(r"18 passed", test_output), test_output
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"] + extra
    markers = [
        RUN / p
        for p in (
            "inner_complete.json",
            "final_complete.json",
            "workflow.json",
            "source_lock.json",
            "selections.json",
            "results.json",
        )
    ]
    primary = {
        "chromaseed_gaussian.py",
        "chromaseed_gaussian_numpy.py",
        "chromaseed_gaussian_reference.py",
        "chromaseed_gaussian_train.py",
    }
    tested = [
        *tests,
        *(ROOT / "scripts" / p for p in sorted(primary)),
        ROOT / "docs/research/chromaseed_gaussian_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: FG feature-group series verified, parent freshly checked read-only before TG",
        source_locks_verified=source_checks,
        source_entries_checked_total=693,
        unique_source_paths_per_run_checked_total=691,
        parent_artifact_bindings_checked=prior["parent_artifact_bindings_checked"],
        diagnostic_input_bindings_checked=847,
        current_input_bindings_checked=42,
        parent_FG_verification_sha256=FG_RECEIPT,
        parent_FG_fresh_readonly_output=parent,
        selected_numeric_file_hashes_checked=files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=18,
        pytest_output=test_output,
        pytest_evidence="18 numerical tests passed before primary fitting (2.41 s); final identical suite captured here. Frozen primary/ref/consumer/tests/protocol unchanged. Independent full scalar refits are additional evidence.",
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID26824/session6175, independent audit99472 and runtime47215 all returned exit0 through their original handles; report returned exit0 directly. Fresh process scan empty.",
        complete_timing_fits_including_warmups=72,
        exact_timing_payloads=72,
        local_markdown_links_checked=links,
        figures_visually_inspected=["gaussian_learning_curves.png", "gaussian_cost.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p) for p in scripts if p.name not in primary
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in (
                "scripts/chromaseed_feature_groups_verify.py",
                "scripts/chromaseed_gated_verify.py",
                "scripts/chromaseed_gate_stability_audit.py",
            )
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "torch", "pytest", "ruff", "matplotlib")
        },
        mutable_plan_path=plan.relative_to(ROOT).as_posix(),
        goal_classification="progress; full compact/fast/high-quality goal active; matched analytical readout follow-up planned, not implemented/launched",
        evidence_limits="Smaller/faster response but slower training;16/18 comparisons worse than exact FG norm_static and7/12 Gaussian worse than Adam. Full3 removes observed diag variance-floor case, not universal quality win. People and roles reused/confounded, no ordinary-phone/end-to-end validation. Numerical audit is not independent product-quality evidence.",
    )
    write_json(vp, receipt)
    progress = read(RUN / "progress.json")
    progress.update(
        status="series_complete_verified",
        exit_code=0,
        updated_unix=time.time(),
        verification_sha256=sha(vp),
        active_study_processes=0,
        broader_goal="active",
    )
    write_json(RUN / "progress.json", progress)
    print(
        dict(
            passed=True,
            mode="initial receipt",
            tests=18,
            clean_rows=94642,
            stress_rows=988350,
            verification_sha256=sha(vp),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
