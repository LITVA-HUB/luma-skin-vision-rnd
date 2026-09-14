"""Seal C once and verify existing C/H/X/A receipts without rewriting them."""

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
RUN = ROOT / "experiments/runs/chromaseed_crossfit_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_crossfit_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-crossfit-2026-09-13.md"
SOURCE = "bf6b8d289bab901007d925588a90ab8337fe231291c87ce96ecb01ae0b2b808e"
SETTINGS = "6e8fe0c8e4a2035ebc271c5164c2bde09fb46fe36d1300b3c6026946e06e1041"
RESULT = "5ce1a44e759334d8f76222870b22873584e10484deb29f295b9cc563c87b391b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == SOURCE and sha(RUN / "frozen_settings.json") == SETTINGS
    assert sha(RUN / "results.json") == RESULT
    lock, fixed, result = (
        read(RUN / p) for p in ("source_lock.json", "frozen_settings.json", "results.json")
    )
    assert check_map(ROOT, lock["sources"]) == 68 and check_map(ROOT, lock["input_sha256"]) == 229
    hp = ROOT / "docs/benchmarks/chromaseed_hybrid_v1/verification.json"
    hhash = sha(hp)
    assert hhash == "53883c8debdd447f0237640384439df2a886743b21152e7473cb7d959e242daa"
    h = read(hp)
    check_map(ROOT, h["postprocess_sources"])
    parent = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_hybrid_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent and sha(hp) == hhash
    source_checks = [
        *h["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=68,
            unique_source_paths_checked=68,
        ),
    ]
    assert sum(v["source_entries_checked"] for v in source_checks) == 541
    assert sum(v["unique_source_paths_checked"] for v in source_checks) == 539
    assert h["diagnostic_input_bindings_checked"] + 229 == 723
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for o in (audit, runtime, summary, result):
        assert o["source_lock_sha256"] == SOURCE and o["settings_sha256"] == SETTINGS
    for o in (audit, runtime, summary):
        assert o["results_sha256"] == RESULT
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_crossfit_audit.py")
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_crossfit_runtime.py")
    assert summary["report_source_sha256"] == sha(ROOT / "scripts/chromaseed_crossfit_report.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    assert summary["comparison_counts"] == dict(
        new_vs_raw_adverse=18,
        transfer_vs_raw_adverse=16,
        new_vs_H_adverse=15,
        excluded_vs_included_adverse=6,
    )
    assert len(audit["paired"]) == 60
    assert (
        sum(p["mean_difference"] > 0 for p in audit["paired"] if p["control"].endswith("_raw"))
        == 18
    )
    assert (
        sum(
            p["mean_difference"] > 0
            for p in audit["paired"]
            if p["control"].startswith("in_matched")
        )
        == 6
    )
    for file, field, n in (
        ("all_policies.csv", "rows", 63),
        ("all_doses.csv", "doses", 252),
        ("supervision.csv", "supervision", 18),
    ):
        with (OUT / file).open(encoding="utf-8-sig", newline="") as f:
            contents = list(csv.DictReader(f))
        assert len(contents) == len(summary[field]) == n
        for a, b in zip(contents, summary[field], strict=True):
            assert a == {k: "" if v is None else str(v) for k, v in b.items()}
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
        fr = [
            r
            for r in runtime["full_fit_records"]
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
        assert (
            row["numeric_bytes"] == rr[0]["numeric_bytes"]
            and row["cached_array_bytes"] == tr[0]["numpy_only"]["cached_array_bytes"]
        )
        assert row["seeds"] == len(rr)
        if fr:
            close(row["full_fit_ms"], 1000 * fr[0]["median_seconds"], 1e-12)
        else:
            assert row["full_fit_ms"] is None
    prior = read(ROOT / "experiments/runs/chromaseed_hybrid_v1/selections.json")
    assert fixed["new_C_hyperparameter_selection"] is False
    count = 0
    for role, losses in fixed["settings"].items():
        for loss, kinds in losses.items():
            for kind, s in kinds.items():
                old = prior["roles"][role][loss][kind]["selected"]
                assert s == {k: old[k] for k in s}
                count += 1
    assert count == 12
    expected = dict(
        teacher_subsets=42,
        teacher_models=252,
        teacher_prediction_rows=156504,
        routed_rows=20400,
        matched_row_memberships=1700,
        student_raw_refits=18,
        head_refits=72,
        head_refit_query_rows=258768,
        exact_H_models=111,
        exact_H_prediction_rows=1462758,
        final_models=183,
        final_query_rows=2411574,
        dose_summaries=732,
        analytic_solutions=342,
        gram_decompositions=207,
        basis_preparations=144,
        widths=48,
        covariances=3,
        gates=19,
        auxiliary_baseline_solves=135,
        auxiliary_theta_solves=135,
    )
    for k, v in expected.items():
        assert audit["checks"][k] == v
    assert audit["maxima"]["teacher_qr"] <= 0.001 and audit["maxima"]["head_qr"] <= 0.001
    assert audit["maxima"]["final_portable"] <= 2e-8 and audit["maxima"]["normal_residual"] <= 1e-8
    assert audit["maxima"]["width"] == 0
    assert len(runtime["records"]) == 183 and len(runtime["full_fit_records"]) == 54
    assert (
        runtime["C_complete_fits_including_warmups"] == 96
        and runtime["H_complete_fits_including_warmups"] == 120
    )
    for r in runtime["records"]:
        assert r["numpy_only"]["query_count"] == r["numpy_only"]["verification_rows"] * 3
        assert r["numpy_only"]["max_lab_drift"] <= 2e-8
    selected_files = {}
    for rec in result["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = RUN / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            selected_files[p.relative_to(RUN).as_posix()] = rec[field]
    assert len(selected_files) == 366
    bank_files, receipts = {}, sorted(RUN.glob("banks/*/receipt.json"))
    for p in receipts:
        rec = read(p)
        assert rec["source_lock_sha256"] == SOURCE and rec["settings_sha256"] == SETTINGS
        check_map(p.parent, rec["files"])
        bank_files.update(
            {(p.parent / n).relative_to(RUN).as_posix(): v for n, v in rec["files"].items()}
        )
    assert len(receipts) == 3 and len(bank_files) == 9
    assert read(RUN / "fit_complete.json")["settings_sha256"] == SETTINGS
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(crossfit|hybrid|projection|affine)_(train|audit|runtime)\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    scripts, tests = (
        sorted(ROOT.glob("scripts/chromaseed_crossfit*.py")),
        sorted(ROOT.glob("tests/test_chromaseed_crossfit*.py")),
    )
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    plan = ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-crossfit.md"
    extra = [
        plan,
        ROOT / "docs/research/chromaseed_crossfit_next_decision.md",
        ROOT / "docs/architecture/chromaseed_crossfit_model_card.md",
    ]
    links = 0
    for p in [*OUT.glob("*.md"), *extra, SHORTCUT]:
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
        for field in ("artifact_sha256", "postprocess_sources", "tested_sources"):
            check_map(ROOT, previous[field])
        check_map(RUN, previous["run_marker_and_receipt_sha256"])
        assert previous["shortcut_sha256"] == sha(SHORTCUT)
        assert read(RUN / "progress.json")["verification_sha256"] == sha(vp)
        print(dict(passed=True, mode="read-only existing receipt", verification_sha256=sha(vp)))
        return
    test_output = command([sys.executable, "-m", "pytest", *map(str, tests), "-q"])
    assert re.search(r"11 passed", test_output), test_output
    plan.write_text(plan.read_text(encoding="utf-8").replace("- [ ]", "- [x]"), encoding="utf-8")
    artifacts = [p for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"] + extra
    markers = [
        RUN / p
        for p in (
            "fit_complete.json",
            "workflow.json",
            "source_lock.json",
            "frozen_settings.json",
            "results.json",
        )
    ]
    tested = [
        *tests,
        ROOT / "scripts/chromaseed_crossfit.py",
        ROOT / "scripts/chromaseed_crossfit_train.py",
        ROOT / "scripts/chromaseed_crossfit_reference.py",
        ROOT / "docs/research/chromaseed_crossfit_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        settings_sha256=SETTINGS,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: H source/files/receipt and terminal state freshly revalidated through its read-only verifier before C",
        source_locks_verified=source_checks,
        source_entries_checked_total=541,
        unique_source_paths_per_run_checked_total=539,
        parent_artifact_bindings_checked=h["parent_artifact_bindings_checked"],
        diagnostic_input_bindings_checked=723,
        current_input_bindings_checked=229,
        parent_H_verification_sha256=hhash,
        parent_H_fresh_readonly_output=parent,
        selected_numeric_file_hashes_checked=selected_files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=11,
        pytest_output=test_output,
        pytest_evidence="Nine numerical tests passed before freezing/real fits, including perturbation of excluded person's data;two independent teacher/head tests before audit. Metadata duplicate-key bug fixed before primary freeze. Final combined output captured here.",
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="PrimaryPID30544/session47425, audit29731 and runtime21263 returned exit0 through their handles;report returned exit0 directly. Fresh process scan empty.",
        C_full_timing_fits_including_warmups=96,
        H_full_timing_fits_including_warmups=120,
        local_markdown_links_checked=links,
        figures_visually_inspected=["crossfit_quality.png", "crossfit_training_cost.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name not in ("chromaseed_crossfit.py", "chromaseed_crossfit_train.py")
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in ("scripts/chromaseed_hybrid_verify.py", "scripts/chromaseed_gated_verify.py")
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad goal active; inventory and registered feature-group ablation planned, not launched",
        evidence_limits="Fixed-H excluded-person residual targets do not improve robustly.18/24 new-vs-raw comparisons worsen, including all16 transfer cases;15/24 worse than matched H correction. Training about14x H with unchanged numeric size/inference. Ordinary-phone facial quality unvalidated.",
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
            mode="created final receipt",
            verification_sha256=sha(vp),
            sources=541,
            input_bindings=723,
            parent_bindings=163,
            selected_files=366,
            bank_files=9,
            links=links,
            tests=test_output,
        )
    )


if __name__ == "__main__":
    main()
