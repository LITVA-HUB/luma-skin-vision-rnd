"""Seal FG once; subsequent verification and the C/H/X/A chain are read-only."""

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
RUN = ROOT / "experiments/runs/chromaseed_feature_groups_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_feature_groups_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-feature-groups-2026-09-13.md"
SOURCE = "f6a8ac340650ae5d62004f59916ba720b702e6dc38f1fedf6928f5797b53f4cf"
SELECTION = "595f9e44cf31f0d354147d09e1293600ee2cde602727578d3db0525b500ff031"
RESULT = "7c0007b1e35e675901227340483cf156e0c80703d0d44076c520531235934250"
C_RECEIPT = "80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd"


def aggregate_checks(summary, result, runtime, selection, audit):
    for file, field, n in (
        ("all_groups.csv", "rows", 99),
        ("all_doses.csv", "doses", 396),
        ("all_policies.csv", "policies", 24),
        ("paired.csv", "paired", 84),
    ):
        with (OUT / file).open(encoding="utf-8-sig", newline="") as f:
            actual = list(csv.DictReader(f))
        assert len(actual) == len(summary[field]) == n
        for a, b in zip(actual, summary[field], strict=True):
            assert a == {k: "" if v is None else str(v) for k, v in b.items()}
    assert len({(r["role"], r["family"], r["group"]) for r in summary["rows"]}) == 99
    for row in summary["rows"]:
        key = (row["role"], row["family"], row["group"])
        rr = [r for r in result["records"] if (r["role"], r["family"], r["group"]) == key]
        tr = [r for r in runtime["records"] if (r["role"], r["family"], r["group"]) == key]
        fr = [
            r
            for r in runtime["standalone_fit_records"]
            if (r["role"], r["family"], r["group"]) == key
        ]
        assert row["seeds"] == len(rr) == len(tr) == (1 if row["family"] == "constant" else 3)
        for field in ("person_mean", "image_mean", "p90"):
            close(row[field], np.mean([r["metrics"][field] for r in rr]), 1e-12)
        close(
            row["stress4"],
            np.mean([r["doses"][1]["worst_error"]["person_mean"] for r in rr]),
            1e-12,
        )
        close(row["median_us"], np.median([r["numpy_only"]["median_us"] for r in tr]), 1e-12)
        close(row["maximum_seed_p95_us"], max(r["numpy_only"]["p95_us"] for r in tr), 1e-12)
        for field in ("numeric_bytes", "archive_bytes", "actual_centers"):
            assert row[field + "_min"] == min(r[field] for r in rr)
            assert row[field + "_max"] == max(r[field] for r in rr)
        for op in (min, max):
            assert row["cached_array_bytes_" + op.__name__] == op(
                r["numpy_only"]["cached_array_bytes"] for r in tr
            )
        for field in ("alpha", "input_dimensions", "kernel_dimensions", "active_gate"):
            assert all(row[field] == r[field] for r in rr)
        if fr:
            assert len(fr) == 1 and fr[0]["seed"] == 17 and fr[0]["alpha"] == row["alpha"]
            close(row["full_fit_ms"], fr[0]["median_seconds"] * 1000, 1e-12)
            s = selection["roles"][key[0]][key[1]]["groups"][key[2]]["selected"]
            close(row["inner_clean"], s["clean"], 1e-12)
            close(row["inner_p90"], s["p90"], 1e-12)
            assert row["alpha"] == s["alpha"]
        else:
            assert row["full_fit_ms"] is row["inner_clean"] is row["inner_p90"] is None
    for row in summary["doses"]:
        rr = [
            r for r in result["records"] if all(r[k] == row[k] for k in ("role", "family", "group"))
        ]
        index = (1 / 255, 4 / 255, 16 / 255, 64 / 255).index(row["dose"])
        close(
            row["worst_person_error"],
            np.mean([r["doses"][index]["worst_error"]["person_mean"] for r in rr]),
            1e-12,
        )
    for row in summary["policies"]:
        s = selection["roles"][row["role"]][row["family"]]["policies"][row["policy"]]
        for k in ("group", "alpha"):
            assert row[k] == s[k]
        for k in ("clean", "p90", "numeric_bytes"):
            assert row["inner_" + k] == s[k]
        selected = next(
            r for r in summary["rows"] if all(r[k] == row[k] for k in ("role", "family", "group"))
        )
        raw = next(
            r
            for r in summary["rows"]
            if r["role"] == row["role"] and r["family"] == row["family"] and r["group"] == "raw36"
        )
        for k in (
            "person_mean",
            "p90",
            "stress4",
            "numeric_bytes_min",
            "numeric_bytes_max",
            "median_us",
            "full_fit_ms",
        ):
            assert row[k] == selected[k]
        close(row["difference_vs_raw"], selected["person_mean"] - raw["person_mean"], 1e-12)
    for row, original in zip(summary["paired"], audit["paired"], strict=True):
        for k in (
            "role",
            "family",
            "group",
            "kind",
            "people",
            "improved_people",
            "mean_difference",
        ):
            assert row[k] == original[k]
        assert [row["descriptive_95_low"], row["descriptive_95_high"]] == original[
            "fixed_prediction_person_bootstrap_95"
        ]
        pair = [
            next(
                r
                for r in summary["rows"]
                if r["role"] == row["role"] and r["family"] == row["family"] and r["group"] == g
            )
            for g in (row["group"], "raw36")
        ]
        close(row["mean_difference"], pair[0]["person_mean"] - pair[1]["person_mean"], 1e-12)
    new = [p for p in audit["paired"] if p["kind"] == "new_subset"]
    counts = dict(
        new_comparisons=len(new),
        new_adverse=sum(p["mean_difference"] > 0 for p in new),
        mixed_adverse=sum(p["role"] == "mixed" and p["mean_difference"] > 0 for p in new),
        transfer_adverse=sum(p["role"] != "mixed" and p["mean_difference"] > 0 for p in new),
        historical_X_comparisons=sum(p["kind"] == "historical_X" for p in audit["paired"]),
        policy_adverse=sum(p["difference_vs_raw"] > 0 for p in summary["policies"]),
        compact_policy_adverse=sum(
            p["policy"] == "compact" and p["difference_vs_raw"] > 0 for p in summary["policies"]
        ),
    )
    assert (
        counts
        == summary["comparison_counts"]
        == dict(
            new_comparisons=72,
            new_adverse=36,
            mixed_adverse=24,
            transfer_adverse=12,
            historical_X_comparisons=12,
            policy_adverse=13,
            compact_policy_adverse=8,
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
    assert check_map(ROOT, lock["sources"]) == 73 and check_map(ROOT, lock["input_sha256"]) == 82
    cp = ROOT / "docs/benchmarks/chromaseed_crossfit_v1/verification.json"
    assert sha(cp) == C_RECEIPT
    prior = read(cp)
    check_map(ROOT, prior["postprocess_sources"])
    parent = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_crossfit_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent and sha(cp) == C_RECEIPT
    source_checks = [
        *prior["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=73,
            unique_source_paths_checked=73,
        ),
    ]
    assert sum(s["source_entries_checked"] for s in source_checks) == 614
    assert sum(s["unique_source_paths_checked"] for s in source_checks) == 612
    assert prior["diagnostic_input_bindings_checked"] + 82 == 805
    audit, runtime, summary = (
        read(OUT / p) for p in ("audit.json", "runtime.json", "summary.json")
    )
    assert audit["passed"]
    for obj in (result, audit, runtime, summary):
        assert obj["source_lock_sha256"] == SOURCE and obj["selection_sha256"] == SELECTION
    for obj in (audit, runtime, summary):
        assert obj["results_sha256"] == RESULT
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_feature_groups_audit.py")
    assert runtime["runtime_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_feature_groups_runtime.py"
    )
    assert summary["report_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_feature_groups_report.py"
    )
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["runtime_sha256"] == sha(OUT / "runtime.json")
    assert (
        audit["correct_final_row_predictions"]
        == summary["final_query_rows"]
        == 97 * 1198 * 33
        == 3834798
    )
    erratum = "docs/research/chromaseed_feature_groups_count_erratum.md"
    assert audit["protocol_count_erratum"] == summary["protocol_count_erratum"] == erratum
    assert erratum in audit["dependencies"]
    expected = dict(
        banks=12,
        stored_readouts=3468,
        exact_A_payloads=432,
        imported_X_payloads=432,
        constant_fits=12,
        aliases=1008,
        width_checks=96,
        dense_pivot_paths=288,
        pivot_path_differences=0,
        rank_differences=0,
        gate_reference_checks=28,
        oof_query_rows=491300,
        candidate_scores=288,
        group_choices=96,
        policy_choices=24,
        selected_models=291,
        final_query_rows=3834798,
        final_transform_cases=9603,
        dose_cases=1164,
        qr_refits=288,
        qr_query_rows=3795264,
        new_coefficient_solutions=2016,
        reduced_coefficient_solutions=1728,
        gram_decompositions=336,
        basis_preparations=252,
        exact_width_calculations=84,
        gate_fits=28,
        auxiliary_baseline_solves=36,
        auxiliary_theta_solves=36,
        target_metric_preparations=12,
    )
    assert audit["checks"] == expected
    for key in ("oof_prediction", "direct_prediction", "consumer_prediction"):
        assert audit["maxima"][key] <= 2e-8
    assert audit["maxima"]["qr_prediction"] <= 0.001 and audit["maxima"]["normal_residual"] <= 1e-8
    assert audit["maxima"]["gate_parameter"] <= 1e-5
    assert len(runtime["records"]) == 291 and len(runtime["standalone_fit_records"]) == 96
    assert runtime["complete_fits_including_warmups"] == 384 and runtime["hardware"]["threads"] == 1
    assert len({(r["role"], r["name"]) for r in runtime["records"]}) == 291
    assert (
        len({(r["role"], r["family"], r["group"]) for r in runtime["standalone_fit_records"]}) == 96
    )
    for r in runtime["records"]:
        assert r["numpy_only"]["query_count"] == r["numpy_only"]["verification_rows"] * 3
        assert r["numpy_only"]["max_lab_drift"] <= 2e-8
    for r in runtime["standalone_fit_records"]:
        assert len(r["seconds"]) == 3 and min(r["seconds"]) > 0
        assert r["n_fit_rows"] == dict(mixed=734, slr_to_ipod=323, ipod_to_slr=643)[r["role"]]
        close(r["median_seconds"], np.median(r["seconds"]), 1e-12)
    aggregate_checks(summary, result, runtime, selection, audit)
    selected_files = {}
    for rec in result["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = RUN / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            selected_files[p.relative_to(RUN).as_posix()] = rec[field]
    assert len(selected_files) == 582
    assert (
        len(list((RUN / "selected").glob("*/*.npz")))
        == len(list((RUN / "evaluated").glob("*/*.npz")))
        == 291
    )
    bank_files, receipts = (
        {},
        sorted(RUN.glob("inner/*/fold*/receipt.json"))
        + sorted(RUN.glob("final/*/bank/receipt.json")),
    )
    for p in receipts:
        rec = read(p)
        assert rec["source_lock_sha256"] == SOURCE and len(rec["models"]) == 289
        check_map(p.parent, rec["files"])
        bank_files.update(
            {(p.parent / n).relative_to(RUN).as_posix(): v for n, v in rec["files"].items()}
        )
    assert len(receipts) == 12 and len(bank_files) == 21
    assert read(RUN / "inner_complete.json") == dict(
        source_lock_sha256=SOURCE, banks=9, readouts=2601, selection_sha256=None
    )
    assert read(RUN / "final_complete.json") == dict(
        source_lock_sha256=SOURCE, banks=3, readouts=867, selection_sha256=SELECTION
    )
    assert read(RUN / "workflow.json")["exit_code"] == 0
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(feature_groups|crossfit|hybrid|projection|affine)_(train|audit|runtime)\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    scripts, tests = (
        sorted(ROOT.glob("scripts/chromaseed_feature_groups*.py")),
        sorted(ROOT.glob("tests/test_chromaseed_feature_groups*.py")),
    )
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    plan = ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-feature-groups.md"
    extra = [
        ROOT / erratum,
        ROOT / "docs/research/chromaseed_feature_groups_next_decision.md",
        ROOT / "docs/architecture/chromaseed_feature_groups_model_card.md",
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
    assert re.search(r"14 passed", test_output), test_output
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
        "chromaseed_feature_groups.py",
        "chromaseed_feature_groups_numpy.py",
        "chromaseed_feature_groups_train.py",
    }
    tested = [
        *tests,
        *(ROOT / "scripts" / p for p in sorted(primary)),
        ROOT / "docs/research/chromaseed_feature_groups_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: primary-checked forum/data search completed and C freshly verified read-only before FG",
        source_locks_verified=source_checks,
        source_entries_checked_total=614,
        unique_source_paths_per_run_checked_total=612,
        parent_artifact_bindings_checked=prior["parent_artifact_bindings_checked"],
        diagnostic_input_bindings_checked=805,
        current_input_bindings_checked=82,
        parent_C_verification_sha256=C_RECEIPT,
        parent_C_fresh_readonly_output=parent,
        selected_numeric_file_hashes_checked=selected_files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=14,
        pytest_output=test_output,
        pytest_evidence="14 numerical tests passed before the primary run (1.53 s); final same suite output captured here. Frozen core/consumer/train/test/protocol unchanged after primary source lock.",
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID39764/session44863, independent audit67225 and runtime9942 all returned exit0 through their original handles; report returned exit0 directly. Fresh process scan empty.",
        complete_timing_fits_including_warmups=384,
        local_markdown_links_checked=links,
        figures_visually_inspected=["feature_groups_quality.png", "feature_groups_cost.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p) for p in scripts if p.name not in primary
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in (
                "scripts/chromaseed_crossfit_verify.py",
                "scripts/chromaseed_gated_verify.py",
                "scripts/chromaseed_gate_stability_audit.py",
            )
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "pytest", "ruff", "matplotlib")
        },
        protocol_count_erratum=erratum,
        correct_final_row_predictions=3834798,
        goal_classification="progress; full compact/fast/high-quality goal active; matched TAGI-versus-backprop study planned, not implemented/launched",
        mutable_plan_path=plan.relative_to(ROOT).as_posix(),
        evidence_limits="Three RGB features reduce bytes and improve both reused camera-transfer means but worsen mixed. All24 new mixed comparisons adverse;36/72 adverse overall.13/24 frozen policies worse than raw. Camera/person composition confounded, no ordinary-phone/end-to-end validation. Numerical checks are not independent product-quality evidence.",
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
            tests=14,
            final_query_rows=3834798,
            verification_sha256=sha(vp),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
