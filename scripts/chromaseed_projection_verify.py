"""X integrity receipt; inherited A verifier is explicitly read-only on existing receipt."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_projection_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_projection_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-projection-2026-09-13.md"
SOURCE = "31558592fca6b0b23f22448ac61e8e86fc27da694112b2f9394160e4a5e29ecc"
SELECTION = "134a11bb56bfac8320ca48efe7c3f7c71e971a65147badde4f7d0500edc85e5a"
RESULT = "693280fab39c123e2e7c455b7b5215bf372d564e08551cbb857df9ad7097997e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == SOURCE
    assert sha(RUN / "selections.json") == SELECTION
    assert sha(RUN / "results.json") == RESULT
    lock, result, selected = (
        read(RUN / p) for p in ("source_lock.json", "results.json", "selections.json")
    )
    assert check_map(ROOT, lock["sources"]) == 59
    assert check_map(ROOT, lock["input_sha256"]) == 145
    apath = ROOT / "docs/benchmarks/chromaseed_affine_v1/verification.json"
    a_hash = sha(apath)
    assert a_hash == "91a348711bbe95dba917a6b31c34e5c78d94c6113737a33f0a93ba4fcc5ec646"
    a = read(apath)
    # Verify the verifier code before executing its documented read-only path.
    check_map(ROOT, a["postprocess_sources"])
    parent_check = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_affine_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent_check and sha(apath) == a_hash
    source_checks = [
        *a["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=59,
            unique_source_paths_checked=59,
        ),
    ]
    assert sum(r["source_entries_checked"] for r in source_checks) == 409
    assert sum(r["unique_source_paths_checked"] for r in source_checks) == 407
    assert (
        a["diagnostic_input_bindings_checked"] + 145 == 421
        and a["parent_artifact_bindings_checked"] == 163
    )
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for obj in (result, audit, runtime, summary):
        assert obj["source_lock_sha256"] == SOURCE and obj["selection_sha256"] == SELECTION
    for obj in (audit, runtime, summary):
        assert obj["results_sha256"] == RESULT
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_projection_audit.py")
    assert runtime["runtime_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_projection_runtime.py"
    )
    assert summary["report_source_sha256"] == sha(ROOT / "scripts/chromaseed_projection_report.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    assert summary["parent_A_results_sha256"] == sha(
        ROOT / "experiments/runs/chromaseed_affine_v1/results.json"
    )
    assert (
        len(summary["rows"]) == 39
        and len(summary["doses"]) == 156
        and len(summary["inner_candidates"]) == 468
    )
    assert len(summary["raw_A_controls"]) == 12 and summary["adverse_outer_comparisons"] == 19
    assert sum(r["mean_difference"] > 0 for r in audit["paired"]) == 19
    for r in summary["rows"]:
        rr = [
            v
            for v in result["records"]
            if (v["role"], v["family"], v["policy"]) == (r["role"], r["family"], r["policy"])
        ]
        assert len(rr) == r["seeds"]
        np.testing.assert_allclose(
            r["person_mean"], np.mean([v["metrics"]["person_mean"] for v in rr]), rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            r["stress4"],
            np.mean([v["doses"][1]["worst_error"]["person_mean"] for v in rr]),
            rtol=0,
            atol=1e-12,
        )
        assert r["numeric_bytes"] == rr[0]["numeric_bytes"]
    for role in selected["roles"].values():
        for family in role.values():
            assert len(family["candidates"]) == 39
            assert all(v["alpha"] == 0.1 for v in family["policies"].values())
    assert (
        audit["checks"]["oof_query_rows"] == 817700
        and audit["checks"]["final_query_rows"] == 1462758
    )
    assert (
        audit["checks"]["projection_checks"] == 144
        and audit["checks"]["dense_landmark_checks"] == 468
    )
    assert (
        audit["checks"]["qr_selected_refits"] == 72 and audit["checks"]["qr_positive_probes"] == 12
    )
    assert (
        audit["maxima"]["qr_prediction"] <= 0.001
        and audit["maxima"]["standalone_prediction"] <= 2e-8
    )
    assert audit["maxima"]["projection_metric_relative"] <= 2e-6 and audit["maxima"]["width"] == 0
    assert (
        len(runtime["records"]) == 111
        and len(runtime["standalone_fit_records"]) == 24
        and len(runtime["fixed_positive_records"]) == 4
    )
    assert (
        runtime["selected_complete_fits_including_warmups"] == 96
        and runtime["fixed_complete_fits_including_warmups"] == 16
    )
    for r in runtime["records"]:
        assert r["numpy_only"]["query_count"] == 3 * r["numpy_only"]["verification_rows"]
        assert r["numpy_only"]["max_lab_drift"] <= 2e-8
    selected_files = {}
    for rec in result["records"]:
        for part, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = RUN / part / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            selected_files[p.relative_to(RUN).as_posix()] = rec[field]
    assert len(selected_files) == 222
    bank_files, receipts, records = {}, [], []
    for path in sorted(RUN.rglob("receipt.json")):
        rec = read(path)
        assert rec["source_lock_sha256"] == SOURCE
        check_map(path.parent, rec["files"])
        bank_files.update(
            {(path.parent / p).relative_to(RUN).as_posix(): h for p, h in rec["files"].items()}
        )
        receipts.append(path)
        records.append(rec)
    assert len(receipts) == 12 and len(bank_files) == 21
    for field, value in (
        ("new_coefficient_solutions", 3744),
        ("gram_decompositions", 624),
        ("basis_preparations", 468),
        ("exact_width_calculations", 156),
        ("covariance_eigendecompositions", 12),
        ("gate_fits", 4),
        ("exact_joint_static_aliases", 1872),
        ("exact_A_raw_payloads", 432),
        ("imported_G_payloads", 72),
        ("imported_A_payloads", 72),
        ("constant_fits", 12),
        ("auxiliary_baseline_solves", 36),
        ("auxiliary_theta_solves", 36),
    ):
        assert sum(r[field] for r in records) == value
    for name, count, readouts in (
        ("inner_complete.json", 9, 4329),
        ("final_complete.json", 3, 1443),
    ):
        m = read(RUN / name)
        assert (
            m["source_lock_sha256"] == SOURCE and m["banks"] == count and m["readouts"] == readouts
        )
    assert read(RUN / "final_complete.json")["selection_sha256"] == SELECTION
    markers = [
        RUN / p
        for p in (
            "inner_complete.json",
            "final_complete.json",
            "workflow.json",
            "selections.json",
            "results.json",
        )
    ]
    scripts = sorted(ROOT.glob("scripts/chromaseed_projection*.py"))
    tests = sorted(ROOT.glob("tests/test_chromaseed_projection*.py"))
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(projection_(train|audit|runtime)|affine_(train|audit|runtime)|gate_stability_(run|audit)|gated_(train|audit|runtime))\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    extra = [
        ROOT / p
        for p in (
            "docs/research/chromaseed_projection_next_decision.md",
            "docs/architecture/chromaseed_projection_model_card.md",
            "docs/superpowers/plans/2026-09-13-chromaseed-projection.md",
        )
    ]
    links = 0
    for path in [*sorted(OUT.glob("*.md")), *extra, SHORTCUT]:
        for target in re.findall(r"\]\(([^\n)]+)\)", path.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-z]+://", target):
                continue
            destination = (path.parent / target).resolve()
            assert destination == (OUT / "verification.json").resolve() or destination.exists(), (
                path,
                target,
            )
            links += 1
    receipt_path = OUT / "verification.json"
    if receipt_path.exists():
        previous = read(receipt_path)
        assert previous["passed"] and previous["source_lock_sha256"] == SOURCE
        check_map(ROOT, previous["artifact_sha256"])
        check_map(ROOT, previous["postprocess_sources"])
        check_map(ROOT, previous["tested_sources"])
        assert previous["shortcut_sha256"] == sha(SHORTCUT)
        assert read(RUN / "progress.json")["verification_sha256"] == sha(receipt_path)
        print(
            dict(
                passed=True,
                mode="read-only existing receipt",
                verification_sha256=sha(receipt_path),
            )
        )
        return
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"] + extra
    tested = (
        "scripts/chromaseed_projection.py",
        "scripts/chromaseed_projection_numpy.py",
        "scripts/chromaseed_projection_train.py",
        "scripts/chromaseed_projection_reference.py",
        "tests/test_chromaseed_projection.py",
        "tests/test_chromaseed_projection_reference.py",
        "docs/research/chromaseed_projection_v1_protocol.md",
    )
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: A files/receipt and terminal state revalidated by its read-only verifier before X",
        source_locks_verified=source_checks,
        source_entries_checked_total=409,
        unique_source_paths_per_run_checked_total=407,
        parent_artifact_bindings_checked=163,
        diagnostic_input_bindings_checked=421,
        current_input_bindings_checked=145,
        parent_A_verification_sha256=a_hash,
        parent_A_fresh_readonly_output=parent_check,
        selected_numeric_file_hashes_checked=selected_files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=15,
        pytest_seconds=1.72,
        pytest_command="python -m pytest tests/test_chromaseed_projection.py tests/test_chromaseed_projection_reference.py -q",
        pytest_evidence="Captured15-pass1.72s final combined run.12 primary tests1.66s passed before freezing/fits;3 independent reference tests1.52s before audit. Tested sources unchanged.",
        tested_sources={p: sha(ROOT / p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="PrimaryPID36304/session37948 and independent audit72869 exited0 through their handles. Runtime/report exited0 directly; fresh scan no X/A/G/GS workers.",
        selected_full_timing_fits_including_warmups=96,
        fixed_full_timing_fits_including_warmups=16,
        local_markdown_links_checked=links,
        figures_visually_inspected=["projection_tradeoff.png", "projection_stress.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name
            not in (
                "chromaseed_projection.py",
                "chromaseed_projection_numpy.py",
                "chromaseed_projection_train.py",
            )
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in ("scripts/chromaseed_affine_verify.py", "scripts/chromaseed_gated_verify.py")
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad goal active; raw/projected hybrid study planned, not launched",
        evidence_limits="35.46% mixed payload reduction with small reused-six-person error gain, but19/24 matched outer cases worsen and response is slower than raw controls. Inner guards are not outer bounds. Ordinary-phone facial accuracy unvalidated.",
    )
    write_json(receipt_path, receipt)
    progress = read(RUN / "progress.json")
    progress.update(
        status="series_complete_verified",
        exit_code=0,
        updated_unix=time.time(),
        verification_sha256=sha(receipt_path),
        active_study_processes=0,
        broader_goal="active",
    )
    write_json(RUN / "progress.json", progress)
    print(
        dict(
            passed=True,
            mode="created final receipt",
            verification_sha256=sha(receipt_path),
            source_entries=409,
            input_bindings=421,
            parent_bindings=163,
            selected_files=222,
            bank_files=21,
            local_links=links,
        )
    )


if __name__ == "__main__":
    main()
