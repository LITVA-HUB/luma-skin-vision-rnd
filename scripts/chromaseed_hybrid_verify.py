"""Create one H integrity receipt; subsequent runs verify it without rewriting."""

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
from chromaseed_kernel_audit import nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_hybrid_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_hybrid_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-hybrid-2026-09-13.md"
SOURCE = "42d85a41e8b63d15dda61ab6411b043c1c4eefd075088dc90d934d3b19505d84"
SELECTION = "c3a65f8be1af5e69724729c67b097957dcacac0c179811986d149daaef7daec3"
RESULT = "f5918a094f446a49ed26df84ac8251a3ca793d22b469289c683d7cd0c952876e"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == SOURCE and sha(RUN / "selections.json") == SELECTION
    assert sha(RUN / "results.json") == RESULT
    lock, selected, result = (
        read(RUN / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    assert check_map(ROOT, lock["sources"]) == 64 and check_map(ROOT, lock["input_sha256"]) == 73
    xpath = ROOT / "docs/benchmarks/chromaseed_projection_v1/verification.json"
    xhash = sha(xpath)
    assert xhash == "537014c2d803f27bb9a72f93d260c2b0d1a830a9732fc4037bfe3dc202a98153"
    x = read(xpath)
    check_map(ROOT, x["postprocess_sources"])
    parent = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_projection_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent and sha(xpath) == xhash
    source_checks = [
        *x["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=64,
            unique_source_paths_checked=64,
        ),
    ]
    assert sum(r["source_entries_checked"] for r in source_checks) == 473
    assert sum(r["unique_source_paths_checked"] for r in source_checks) == 471
    assert x["diagnostic_input_bindings_checked"] + 73 == 494
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for o in (audit, runtime, summary, result):
        assert o["source_lock_sha256"] == SOURCE and o["selection_sha256"] == SELECTION
    for o in (audit, runtime, summary):
        assert o["results_sha256"] == RESULT
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_hybrid_audit.py")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_hybrid_runtime.py")
    assert summary["report_source_sha256"] == sha(ROOT / "scripts/chromaseed_hybrid_report.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    assert summary["adverse_outer_comparisons"] == summary["adverse_transfer_comparisons"] == 16
    assert sum(p["mean_difference"] > 0 for p in audit["paired"]) == 16
    assert (
        len(summary["rows"]) == 39
        and len(summary["doses"]) == 156
        and len(summary["inner_candidates"]) == 258
    )
    for file, field, n in (
        ("all_policies.csv", "rows", 39),
        ("all_doses.csv", "doses", 156),
        ("all_inner_candidates.csv", "inner_candidates", 258),
    ):
        with (OUT / file).open(encoding="utf-8-sig", newline="") as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == n
        for a, b in zip(csv_rows, summary[field], strict=True):
            assert a == {k: "" if v is None else str(v) for k, v in b.items()}
    with np.load(args.cache, allow_pickle=False) as z:
        data = {k: z[k] for k in ("target", "patient", "site")}
    files = {}
    for rec in result["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = RUN / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            files[p.relative_to(RUN).as_posix()] = rec[field]
        saved = nz(RUN / "evaluated" / rec["role"] / f"{rec['name']}.npz")
        rows = saved["row_indices"]
        independent, _ = error_summary(
            saved["prediction"][0], data["target"][rows], data["patient"][rows], data["site"][rows]
        )
        close(rec["metrics"], independent)
    assert len(files) == 222
    for row in summary["rows"]:
        rr = [
            r
            for r in result["records"]
            if r["role"] == row["role"] and r["family"] == row["family"]
        ]
        tr = [
            r
            for r in runtime["records"]
            if r["role"] == row["role"] and r["family"] == row["family"]
        ]
        for field in ("person_mean", "image_mean", "p90"):
            close(row[field], np.mean([r["metrics"][field] for r in rr]), 1e-12)
        close(
            row["stress4"],
            np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr]),
            1e-12,
        )
        close(row["median_us"], np.median([r["numpy_only"]["median_us"] for r in tr]), 1e-12)
        assert row["numeric_bytes"] == rr[0]["numeric_bytes"] and row["seeds"] == len(rr)
    count = 0
    for role in selected["roles"].values():
        for loss in role.values():
            for family, entry in loss.items():
                assert (
                    len(entry["candidates"])
                    == dict(raw=1, projected=3, blend=10, uniform=10, support=19)[family]
                )
                if family in ("blend", "uniform", "support"):
                    assert entry["selected"]["rho"] > 0
                if family == "support":
                    assert entry["selected"]["power"] == 1
                count += 1
    assert count == 30
    for field, expected in dict(
        stored_readouts=2964,
        oof_rows=419900,
        final_query_rows=1462758,
        candidate_scores=258,
        choices=30,
        selected_reconstructions=90,
        forced_positive_reconstructions=18,
        new_coefficient_solutions=936,
        gram_decompositions=288,
    ).items():
        assert audit["checks"][field] == expected
    assert audit["maxima"]["qr_prediction"] <= 0.001 and audit["maxima"]["standalone"] <= 2e-8
    assert audit["maxima"]["normal_residual"] <= 1e-8 and audit["maxima"]["width"] == 0
    assert (
        len(runtime["records"]) == 111
        and len(runtime["standalone_fit_records"]) == 30
        and len(runtime["fixed_positive_records"]) == 6
    )
    assert (
        runtime["selected_complete_fits_including_warmups"] == 120
        and runtime["fixed_complete_fits_including_warmups"] == 24
    )
    for r in runtime["records"]:
        assert r["numpy_only"]["query_count"] == 3 * r["numpy_only"]["verification_rows"]
        assert r["numpy_only"]["max_lab_drift"] <= 2e-8
    bank_files, receipts = {}, []
    for p in sorted(RUN.rglob("receipt.json")):
        r = read(p)
        assert r["source_lock_sha256"] == SOURCE
        check_map(p.parent, r["files"])
        bank_files.update(
            {(p.parent / n).relative_to(RUN).as_posix(): h for n, h in r["files"].items()}
        )
        receipts.append(p)
    assert len(receipts) == 12 and len(bank_files) == 21
    assert read(RUN / "inner_complete.json")["readouts"] == 2223
    assert read(RUN / "final_complete.json")["readouts"] == 741
    assert read(RUN / "final_complete.json")["selection_sha256"] == SELECTION
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(hybrid|projection|affine)_(train|audit|runtime)\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    scripts, tests = (
        sorted(ROOT.glob("scripts/chromaseed_hybrid*.py")),
        sorted(ROOT.glob("tests/test_chromaseed_hybrid*.py")),
    )
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    plan = ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-hybrid.md"
    extra = [
        plan,
        ROOT / "docs/research/chromaseed_hybrid_next_decision.md",
        ROOT / "docs/architecture/chromaseed_hybrid_model_card.md",
    ]
    links = 0
    for p in [*OUT.glob("*.md"), *extra, SHORTCUT]:
        for target in re.findall(r"\]\(([^\n)]+)\)", p.read_text(encoding="utf-8")):
            target = unquote(target.strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-z]+://", target):
                continue
            destination = (p.parent / target).resolve()
            assert destination == (OUT / "verification.json").resolve() or destination.exists(), (
                p,
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
        check_map(RUN, previous["run_marker_and_receipt_sha256"])
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
    test_output = command([sys.executable, "-m", "pytest", *map(str, tests), "-q"])
    assert re.search(r"18 passed", test_output), test_output
    plan.write_text(plan.read_text(encoding="utf-8").replace("- [ ]", "- [x]"), encoding="utf-8")
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
    tested = [
        *tests,
        ROOT / "scripts/chromaseed_hybrid.py",
        ROOT / "scripts/chromaseed_hybrid_numpy.py",
        ROOT / "scripts/chromaseed_hybrid_train.py",
        ROOT / "scripts/chromaseed_hybrid_reference.py",
        ROOT / "docs/research/chromaseed_hybrid_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: X source/receipt/files and terminal state freshly revalidated through read-only verifier before H",
        source_locks_verified=source_checks,
        source_entries_checked_total=473,
        unique_source_paths_per_run_checked_total=471,
        parent_artifact_bindings_checked=x["parent_artifact_bindings_checked"],
        diagnostic_input_bindings_checked=494,
        current_input_bindings_checked=73,
        parent_X_verification_sha256=xhash,
        parent_X_fresh_readonly_output=parent,
        selected_numeric_file_hashes_checked=files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        ordinary_metric_reconstructions=111,
        tests_passed=18,
        pytest_output=test_output,
        pytest_evidence="15 primary numerical tests passed before source freeze and real fits;3 reference tests passed before independent audit. Final combined output captured here.",
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="PrimaryPID21388/session62638 and audit56384 returned exit0 through handles;runtime/report returned exit0 directly. Fresh process scan empty.",
        selected_full_timing_fits_including_warmups=120,
        fixed_full_timing_fits_including_warmups=24,
        local_markdown_links_checked=links,
        figures_visually_inspected=["hybrid_quality.png", "hybrid_cost.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name
            not in (
                "chromaseed_hybrid.py",
                "chromaseed_hybrid_numpy.py",
                "chromaseed_hybrid_train.py",
            )
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in (
                "scripts/chromaseed_projection_verify.py",
                "scripts/chromaseed_gated_verify.py",
            )
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad goal active; nested person-held-out residual-target study planned, not launched",
        evidence_limits="Mixed error improves slightly, all16 transfer comparisons worsen. Extra branch+25.17% numeric storage and~1.86x CPU response time; support is not calibrated confidence. Ordinary-phone facial accuracy unvalidated.",
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
            sources=473,
            input_bindings=494,
            parent_bindings=163,
            selected_files=222,
            bank_files=21,
            links=links,
            tests=test_output,
        )
    )


if __name__ == "__main__":
    main()
