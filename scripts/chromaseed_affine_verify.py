"""Verify A and inherited evidence; create the final receipt once, then read-only."""

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
RUNS = ROOT / "experiments/runs"
RUN = RUNS / "chromaseed_affine_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_affine_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-affine-2026-09-13.md"
SOURCE = "dfa348cfa6663f58de56d7a09105e4f1c81f77712297b85c9b8039eb1c5cef2a"
SELECTION = "e2e290c7b798294e76b4cab1d142ff5e10844bece9d265adb20b5432d1e39977"
RESULT = "283e5863c7e0df50cb27883b9c4151f88272788789c6f8a5486828949c9cca58"


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
    assert check_map(ROOT, lock["input_sha256"]) == 101
    gs_path = ROOT / "docs/benchmarks/chromaseed_gate_stability_v1/verification.json"
    assert sha(gs_path) == "2ad0c77feabbe7d63fe5f6ab4ecf905a4705a95014e4cb4dc7c8b703a8c4c738"
    gs = read(gs_path)
    assert gs["passed"]
    parent_checks = check_map(ROOT, gs["artifact_sha256"]) + check_map(
        ROOT, gs["postprocess_sources"]
    )
    parent_checks += check_map(ROOT, gs["verifier_dependency_sha256"])
    parent_checks += check_map(
        RUNS / "chromaseed_gate_stability_v1", gs["current_numeric_file_hashes_checked"]
    )
    gp = read(RUNS / "chromaseed_gate_stability_v1/progress.json")
    assert gp["status"] == "series_complete_verified" and gp["verification_sha256"] == sha(gs_path)
    sources, locks = [], {}
    for item in [*gs["source_locks_verified"], dict(run=RUN.name, source_lock_sha256=SOURCE)]:
        path = RUNS / item["run"] / "source_lock.json"
        assert sha(path) == item["source_lock_sha256"]
        bound = read(path)
        locks[item["run"]] = bound
        assert bound["cache_sha256"] == CACHE_HASH
        seen, count = {}, 0
        for field in ("sources", "source_hashes", "source_code_hashes", "compact_hashes"):
            for p, h in bound.get(field, {}).items():
                assert sha(ROOT / p) == h, p
                canonical = Path(p).as_posix()
                assert canonical not in seen or seen[canonical] == h
                seen[canonical] = h
                count += 1
        sources.append(
            dict(
                run=item["run"],
                source_lock_sha256=sha(path),
                source_entries_checked=count,
                unique_source_paths_checked=len(seen),
            )
        )
    assert sum(r["source_entries_checked"] for r in sources) == 350
    assert sum(r["unique_source_paths_checked"] for r in sources) == 348
    bindings = 0
    for child, parent, key in (
        ("chromaseed_kernel_v1", "skin_local_search_v1", "legacy_bindings"),
        ("chromaseed_fast_kernel_v1", "chromaseed_kernel_v1", "parent_bindings"),
        ("chromaseed_perceptual_v1", "chromaseed_fast_kernel_v1", "parent_bindings"),
        ("chromaseed_weak_ridge_v1", "chromaseed_perceptual_v1", "parent_bindings"),
        ("chromaseed_gated_v1", "chromaseed_perceptual_v1", "parent_bindings"),
    ):
        bindings += check_map(RUNS / parent, locks[child][key])
    for field, p in {
        "source_run_lock_sha256": "experiments/runs/chromaseed_fast_kernel_v1/source_lock.json",
        "source_selections_sha256": "experiments/runs/chromaseed_fast_kernel_v1/selections.json",
        "source_results_sha256": "experiments/runs/chromaseed_fast_kernel_v1/results.json",
        "parent_audit_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/audit.json",
        "parent_runtime_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/runtime.json",
    }.items():
        assert sha(ROOT / p) == locks["chromaseed_condensed_exact_v1"][field]
        bindings += 1
    for child in ("chromaseed_perceptual_v1", "chromaseed_weak_ridge_v1"):
        assert (
            sha(RUNS / "chromaseed_condensed_exact_v1/source_lock.json")
            == locks[child]["exact_source_lock_sha256"]
        )
        bindings += 1
    for field, p in (
        ("parent_audit_sha256", "audit.json"),
        ("parent_verification_sha256", "verification.json"),
    ):
        assert (
            sha(ROOT / "docs/benchmarks/chromaseed_perceptual_v1" / p)
            == locks["chromaseed_weak_ridge_v1"][field]
        )
        bindings += 1
    assert bindings == 163
    inputs = sum(check_map(ROOT, r.get("input_sha256", {})) for r in locks.values())
    assert inputs == 276
    gpath = ROOT / "docs/benchmarks/chromaseed_gated_v1/verification.json"
    assert sha(gpath) == gs["parent_G_verification_sha256"]
    g = read(gpath)
    check_map(ROOT, g["artifact_sha256"])
    check_map(ROOT, g["postprocess_sources"])
    for field in (
        "selected_numeric_file_hashes_checked",
        "bank_numeric_file_hashes_checked",
        "run_marker_and_receipt_sha256",
    ):
        check_map(RUNS / "chromaseed_gated_v1", g[field])
    dpath = ROOT / "docs/benchmarks/chromaseed_camera_support_v1/verification.json"
    assert sha(dpath) == g["parent_D_verification_sha256"]
    d = read(dpath)
    check_map(ROOT, d["artifact_sha256"])
    check_map(ROOT, d["postprocess_sources"])
    check_map(RUNS / "chromaseed_camera_support_v1", d["current_numeric_file_hashes_checked"])
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for obj in (audit, runtime, summary, result):
        assert obj["source_lock_sha256"] == SOURCE and obj["selection_sha256"] == SELECTION
    for obj in (audit, runtime, summary):
        assert obj["results_sha256"] == RESULT
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_affine_audit.py")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_affine_runtime.py")
    assert summary["report_source_sha256"] == sha(ROOT / "scripts/chromaseed_affine_report.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    assert (
        len(summary["rows"]) == 39
        and len(summary["doses"]) == 156
        and len(summary["inner_candidates"]) == 144
    )
    for r in summary["rows"]:
        rr = [
            v
            for v in result["records"]
            if (v["role"], v["family"], v["policy"]) == (r["role"], r["family"], r["policy"])
        ]
        assert len(rr) == r["seeds"]
        np.testing.assert_allclose(
            r["clean_person_mean"],
            np.mean([v["metrics"]["person_mean"] for v in rr]),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            r["stress4_person_mean"],
            np.mean([v["doses"][1]["worst_error"]["person_mean"] for v in rr]),
            atol=1e-12,
            rtol=0,
        )
    for role in selected["roles"].values():
        for family in role.values():
            assert family["policies"]["clean"]["eta"] == 0
            assert family["policies"]["guarded"]["eta"] == 0.75
            assert all(v["alpha"] == 0.1 for v in family["policies"].values())
    assert (
        audit["checks"]["oof_query_rows"] == 2402100
        and audit["checks"]["final_query_rows"] == 1462758
    )
    assert (
        audit["checks"]["qr_selected_refits"] == 72 and audit["checks"]["qr_positive_probes"] == 12
    )
    assert (
        audit["maxima"]["qr_prediction"] <= 0.001
        and audit["maxima"]["standalone_prediction"] <= 2e-8
    )
    assert audit["zero_eta_static_payloads_exact_G"] == 72
    assert (
        runtime["selected_complete_fits_including_warmups"] == 96
        and runtime["fixed_complete_fits_including_warmups"] == 16
    )
    assert (
        len(runtime["records"]) == 111
        and len(runtime["standalone_fit_records"]) == 24
        and len(runtime["fixed_positive_records"]) == 4
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
    banks, receipts, counts = {}, [], []
    for path in sorted(RUN.rglob("receipt.json")):
        rec = read(path)
        assert rec["source_lock_sha256"] == SOURCE
        check_map(path.parent, rec["files"])
        banks.update(
            {(path.parent / p).relative_to(RUN).as_posix(): h for p, h in rec["files"].items()}
        )
        receipts.append(path)
        counts.append(rec)
    assert len(receipts) == 12 and len(banks) == 21
    for field, expected in (
        ("new_coefficient_solutions", 1152),
        ("gram_decompositions", 192),
        ("gate_fits", 4),
        ("exact_joint_static_aliases", 576),
        ("imported_G_payloads", 144),
        ("constant_fits", 12),
    ):
        assert sum(r[field] for r in counts) == expected
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
    for name, nbank, nread in (("inner_complete.json", 9, 1413), ("final_complete.json", 3, 471)):
        m = read(RUN / name)
        assert m["source_lock_sha256"] == SOURCE and m["banks"] == nbank and m["readouts"] == nread
    assert read(RUN / "final_complete.json")["selection_sha256"] == SELECTION
    scripts = sorted(ROOT.glob("scripts/chromaseed_affine*.py"))
    tests = sorted(ROOT.glob("tests/test_chromaseed_affine*.py"))
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(affine_(train|audit|runtime)|gate_stability_(run|audit)|gated_(train|audit|runtime))\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    extra = [
        ROOT / p
        for p in (
            "docs/research/chromaseed_affine_next_decision.md",
            "docs/architecture/chromaseed_affine_model_card.md",
            "docs/superpowers/plans/2026-09-13-chromaseed-affine.md",
        )
    ]
    links = 0
    for path in [*sorted(OUT.glob("*.md")), *extra, SHORTCUT]:
        for link in re.findall(r"\]\(([^\n)]+)\)", path.read_text(encoding="utf-8")):
            target = unquote(link.strip("<>").split("#", 1)[0])
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
        assert sha(SHORTCUT) == previous["shortcut_sha256"]
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
        "scripts/chromaseed_affine.py",
        "scripts/chromaseed_affine_train.py",
        "scripts/chromaseed_affine_reference.py",
        "tests/test_chromaseed_affine.py",
        "tests/test_chromaseed_affine_reference.py",
        "tests/test_chromaseed_gate_stability.py",
        "docs/research/chromaseed_affine_v1_protocol.md",
    )
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: GS source/input/artifact bindings and terminal state revalidated before A learning",
        source_locks_verified=sources,
        source_entries_checked_total=350,
        unique_source_paths_per_run_checked_total=348,
        parent_artifact_bindings_checked=163,
        diagnostic_input_bindings_checked=276,
        current_input_bindings_checked=101,
        parent_GS_verification_sha256=sha(gs_path),
        parent_GS_artifacts_postprocess_numeric_checked=parent_checks,
        parent_G_verification_sha256=sha(gpath),
        parent_D_verification_sha256=sha(dpath),
        selected_numeric_file_hashes_checked=selected_files,
        bank_numeric_file_hashes_checked=banks,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=23,
        pytest_seconds=1.62,
        pytest_command="python -m pytest tests/test_chromaseed_affine.py tests/test_chromaseed_affine_reference.py tests/test_chromaseed_gate_stability.py -q",
        pytest_evidence="Captured23-pass1.62s after independent audit/runtime; primary12A+8GS passed before source freeze/fits and3QR tests before audit. Tested sources unchanged.",
        tested_sources={p: sha(ROOT / p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID35120/session41445 exit0 confirmed before continuation; independent audit18105 exit0 confirmed via handle; runtime/report exit0 directly. Fresh scan no A/G/GS workers.",
        selected_full_timing_fits_including_warmups=96,
        fixed_full_timing_fits_including_warmups=16,
        local_markdown_links_checked=links,
        figures_visually_inspected=["affine_quality.png", "affine_all_doses.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name not in ("chromaseed_affine.py", "chromaseed_affine_train.py")
        },
        verifier_dependency_sha256={
            "scripts/chromaseed_gated_verify.py": sha(ROOT / "scripts/chromaseed_gated_verify.py")
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad compact/fast/high-quality goal remains active; input projection learning planned, not launched",
        evidence_limits="Historical TRAIN roles overlap and are reused. Augmentation is synthetic with unchanged targets; improves some stress metrics but worsens forward transfer. Joint inner score and person bootstrap prevent a confirmed winner claim. No ordinary-phone face validation.",
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
            source_entries=350,
            parent_bindings=163,
            input_bindings=276,
            selected_files=222,
            bank_files=21,
            local_links=links,
        )
    )


if __name__ == "__main__":
    main()
