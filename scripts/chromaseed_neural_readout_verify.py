"""Seal NR once; validate existing NR/TG/FG/C/H/X/A receipts without rewriting."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from chromaseed_gate_stability_audit import close
from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_neural_readout_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_neural_readout_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-neural-readout-2026-09-13.md"
SOURCE = "70246d9935e52d5be0dcebf758ba3e63b30be4965a9154a19ba180f5cf1782c1"
SELECTION = "258ba273abff66fbac6f442736f9af0ec3312b204f5cbc9e9310d6ecf4154954"
RESULT = "b13dbb2cfde6a8e85dbe4d35f4d5254cd384c916c5b5d2a323c1d0e1ecf67522"
TG_RECEIPT = "19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098"


def matched(records, row):
    return [r for r in records if all(r[k] == row[k] for k in ("role", "group", "family", "basis"))]


def aggregate_checks(summary, result, runtime, selection, audit):
    for file, field, n in (
        ("all_models.csv", "rows", 129),
        ("all_doses.csv", "doses", 516),
        ("inner_candidates.csv", "candidates", 252),
        ("policies.csv", "policies", 12),
        ("paired.csv", "paired", 168),
    ):
        with (OUT / file).open(encoding="utf-8-sig", newline="") as f:
            actual = list(csv.DictReader(f))
        assert len(actual) == len(summary[field]) == n
        for a, b in zip(actual, summary[field], strict=True):
            assert a == {k: "" if v is None else str(v) for k, v in b.items()}
    keys = {(r["role"], r["group"], r["family"], r["basis"]) for r in summary["rows"]}
    assert len(keys) == 129
    for row in summary["rows"]:
        rr, tr, fr = (
            matched(records, row)
            for records in (
                result["records"],
                runtime["records"],
                runtime["standalone_fit_records"],
            )
        )
        assert row["seeds"] == len(rr) == len(tr) == (1 if row["family"] == "constant" else 3)
        assert row["policy_selected"] == all(r["policy_selected"] for r in rr)
        assert all(r["alpha"] == row["alpha"] for r in rr)
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
        if row["family"] in ("norm", "perceptual"):
            selected = selection["roles"][row["role"]][row["group"]][row["family"]]["bases"][
                row["basis"]
            ]["selected"]
            assert row["alpha"] == selected["alpha"]
            close(row["inner_clean"], selected["clean"], 1e-12)
            close(row["inner_p90"], selected["p90"], 1e-12)
        else:
            assert row["inner_clean"] is row["inner_p90"] is None
        if fr:
            assert len(fr) == 1 and fr[0]["seed"] == 17 and fr[0]["alpha"] == row["alpha"]
            close(row["full_fit_ms"], fr[0]["median_seconds"] * 1000, 1e-12)
            assert (
                row["representation_training_parameter_state_bytes"]
                == fr[0]["representation_training_parameter_state_bytes"]
            )
        else:
            assert (
                row["full_fit_ms"] is row["representation_training_parameter_state_bytes"] is None
            )
        for field in ("temporary_design_bytes", "temporary_metric_bytes", "temporary_system_bytes"):
            want = (
                None
                if not fr or fr[0]["head_training_array_bytes"] is None
                else fr[0]["head_training_array_bytes"][field]
            )
            assert row["head_" + field] == want
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
        entry = selection["roles"][row["role"]][row["group"]][row["family"]]["bases"][row["basis"]]
        c = entry["candidates"][row["alpha_index"]]
        assert {k: v for k, v in row.items() if k not in ("role", "selected")} == {
            k: v for k, v in c.items() if k != "seed_scores"
        }
        assert row["selected"] == (row["alpha_index"] == entry["selected"]["alpha_index"])
    assert sum(r["selected"] for r in summary["candidates"]) == 84
    for row in summary["policies"]:
        c = selection["roles"][row["role"]][row["group"]][row["family"]]["policy"]
        for field in ("alpha", "basis"):
            assert row[field] == c[field]
        close(row["inner_clean"], c["clean"], 1e-12)
        close(row["inner_p90"], c["p90"], 1e-12)
        selected = matched(summary["rows"], row)[0]
        baseline = matched(summary["rows"], dict(row, family="fg_norm_static", basis="reference"))[
            0
        ]
        for field in ("person_mean", "p90", "stress4", "full_fit_ms", "median_us"):
            assert row[field] == selected[field]
        assert (
            row["numeric_bytes"] == selected["numeric_bytes_min"]
            and row["FG_person_mean"] == baseline["person_mean"]
        )
        close(row["difference_vs_FG"], selected["person_mean"] - baseline["person_mean"], 1e-12)
    for row, original in zip(summary["paired"], audit["paired"], strict=True):
        assert {
            k: v for k, v in row.items() if k not in ("descriptive_95_low", "descriptive_95_high")
        } == {k: v for k, v in original.items() if k != "fixed_prediction_person_bootstrap_95"}
        assert [row["descriptive_95_low"], row["descriptive_95_high"]] == original[
            "fixed_prediction_person_bootstrap_95"
        ]
        current = matched(summary["rows"], row)[0]
        control = matched(
            summary["rows"],
            dict(
                row,
                family=row["control"],
                basis="reference" if row["control"] == "fg_norm_static" else row["basis"],
            ),
        )[0]
        close(row["mean_difference"], current["person_mean"] - control["person_mean"], 1e-12)
    counts = {
        kind: dict(
            comparisons=sum(p["kind"] == kind for p in audit["paired"]),
            adverse=sum(p["kind"] == kind and p["mean_difference"] > 0 for p in audit["paired"]),
        )
        for kind in ("head_vs_FG", "head_vs_unchanged", "policy_vs_FG")
    }
    assert (
        counts
        == summary["comparison_counts"]
        == dict(
            head_vs_FG=dict(comparisons=84, adverse=79),
            head_vs_unchanged=dict(comparisons=72, adverse=18),
            policy_vs_FG=dict(comparisons=12, adverse=12),
        )
    )
    assert (
        summary["selected_alpha_counts"]
        == dict(
            Counter(
                str(r["alpha"]) for r in summary["rows"] if r["family"] in ("norm", "perceptual")
            )
        )
        == {"10.0": 77, "1.0": 7}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert args.cache.name == "train.npz" and sha(args.cache) == CACHE_HASH
    assert (
        sha(RUN / "source_lock.json") == SOURCE
        and sha(RUN / "selections.json") == SELECTION
        and sha(RUN / "results.json") == RESULT
    )
    lock, selection, result = (
        read(RUN / p) for p in ("source_lock.json", "selections.json", "results.json")
    )
    assert check_map(ROOT, lock["sources"]) == 84 and check_map(ROOT, lock["input_sha256"]) == 44
    pp = ROOT / "docs/benchmarks/chromaseed_gaussian_v1/verification.json"
    assert sha(pp) == TG_RECEIPT
    prior = read(pp)
    check_map(ROOT, prior["postprocess_sources"])
    parent = command(
        [
            sys.executable,
            str(ROOT / "scripts/chromaseed_gaussian_verify.py"),
            "--cache",
            str(args.cache),
        ]
    )
    assert "read-only existing receipt" in parent and sha(pp) == TG_RECEIPT
    source_checks = [
        *prior["source_locks_verified"],
        dict(
            run=RUN.name,
            source_lock_sha256=SOURCE,
            source_entries_checked=84,
            unique_source_paths_checked=84,
        ),
    ]
    assert sum(s["source_entries_checked"] for s in source_checks) == 777
    assert sum(s["unique_source_paths_checked"] for s in source_checks) == 775
    assert prior["diagnostic_input_bindings_checked"] + 44 == 891
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
        assert obj[field] == sha(ROOT / f"scripts/chromaseed_neural_readout_{file}.py")
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(
        OUT / "audit.json"
    ) and summary["runtime_sha256"] == sha(OUT / "runtime.json")
    expected = dict(
        banks=12,
        representation_trajectories=144,
        learned_checkpoints=432,
        random_bases=72,
        basis_payloads=504,
        exact_reconstructed_bases=503,
        exact_TG_bases=369,
        example_updates=979200,
        qr_head_refits=3024,
        qr_fit_rows=1285200,
        qr_query_rows=730296,
        stored_models=3540,
        unchanged_models=432,
        exact_FG_models=84,
        oof_rows=501500,
        candidate_scores=252,
        choices=84,
        policies=12,
        selected_models=381,
        final_rows=5020818,
        transform_cases=12573,
        dose_cases=1524,
        refit_stress_rows=4744080,
    )
    assert audit["checks"] == expected
    assert len(audit["non_bitwise_representations"]) == 1
    assert audit["non_bitwise_representations"][0] == dict(
        role="slr_to_ipod",
        stage="inner",
        fold=1,
        name="tagi_full3_e16_raw36_s29",
        differences=dict(w1=7.450580596923828e-09),
    )
    assert audit["maxima"]["representation_parameter"] == 7.450580596923828e-09
    for k in ("oof_prediction", "direct_prediction", "consumer_prediction"):
        assert audit["maxima"][k] <= 2e-8
    for k in ("qr_prediction", "qr_stress_prediction", "folded_prediction"):
        assert audit["maxima"][k] <= 0.001
    assert audit["maxima"]["normal_residual"] <= 1e-8
    assert len(runtime["records"]) == 381 and len(runtime["standalone_fit_records"]) == 126
    assert runtime["complete_fits_including_warmups"] == 378 and runtime["hardware"]["threads"] == 1
    assert (
        runtime["head_complete_fits"] == 252
        and runtime["unchanged_complete_fits"] == 108
        and runtime["FG_complete_fits"] == 18
    )
    assert len({(r["role"], r["name"]) for r in runtime["records"]}) == 381
    assert len({(r["role"], r["name"]) for r in runtime["standalone_fit_records"]}) == 126
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
    selected_files = {}
    for rec in result["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            p = RUN / area / rec["role"] / f"{rec['name']}.npz"
            assert sha(p) == rec[field]
            selected_files[p.relative_to(RUN).as_posix()] = rec[field]
    assert len(selected_files) == 762
    assert (
        len(list((RUN / "selected").glob("*/*.npz")))
        == len(list((RUN / "evaluated").glob("*/*.npz")))
        == 381
    )
    assert (RUN / "selected/mixed/norm_random_raw36_s17_a2.npz").exists()
    receipts = sorted(RUN.glob("inner/*/fold*/receipt.json")) + sorted(
        RUN.glob("final/*/bank/receipt.json")
    )
    bank_files = {}
    for p in receipts:
        r = read(p)
        assert (
            r["source_lock_sha256"] == SOURCE
            and len(r["models"]) == 295
            and len(r["bases"]) == 42
            and len(r["trajectories"]) == 12
        )
        check_map(p.parent, r["files"])
        bank_files.update(
            {(p.parent / n).relative_to(RUN).as_posix(): v for n, v in r["files"].items()}
        )
    assert len(receipts) == 12 and len(bank_files) == 33
    assert read(RUN / "inner_complete.json") == dict(
        source_lock_sha256=SOURCE,
        selection_sha256=None,
        banks=9,
        records=2655,
        trajectories=108,
        bases=378,
        head_solves=2268,
    )
    assert read(RUN / "final_complete.json") == dict(
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        banks=3,
        records=885,
        trajectories=36,
        bases=126,
        head_solves=756,
    )
    assert read(RUN / "workflow.json")["exit_code"] == 0
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(neural_readout|gaussian|feature_groups|crossfit|hybrid|projection|affine)_(train|audit|runtime)\\.py' } | Select-Object ProcessId,Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    scripts, tests = (
        sorted(ROOT.glob("scripts/chromaseed_neural_readout*.py")),
        [ROOT / "tests/test_chromaseed_neural_readout.py"],
    )
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    plan = ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-neural-readout.md"
    extra = [
        ROOT / p
        for p in (
            "docs/research/chromaseed_neural_readout_next_decision.md",
            "docs/architecture/chromaseed_neural_readout_model_card.md",
            "docs/research/chromaseed_gaussian_consumer_erratum.md",
        )
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
    assert re.search(r"24 passed", test_output), test_output
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
        "chromaseed_neural_readout.py",
        "chromaseed_neural_readout_reference.py",
        "chromaseed_neural_readout_train.py",
    }
    tested = [
        *tests,
        *(ROOT / "scripts" / p for p in sorted(primary)),
        ROOT / "docs/research/chromaseed_neural_readout_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=SOURCE,
        selection_sha256=SELECTION,
        results_sha256=RESULT,
        cache_sha256=CACHE_HASH,
        previous_goal_turn_classification="progress: TG sealed and freshly verified read-only before NR",
        source_locks_verified=source_checks,
        source_entries_checked_total=777,
        unique_source_paths_per_run_checked_total=775,
        parent_artifact_bindings_checked=prior["parent_artifact_bindings_checked"],
        diagnostic_input_bindings_checked=891,
        current_input_bindings_checked=44,
        parent_TG_verification_sha256=TG_RECEIPT,
        parent_TG_fresh_readonly_output=parent,
        selected_numeric_file_hashes_checked=selected_files,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p) for p in [*receipts, *markers]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=24,
        pytest_output=test_output,
        pytest_evidence="24 numerical tests passed before primary(1.47 s); final same suite captured here. Frozen primary/core/ref/tests/protocol unchanged. Independent144 trajectory and3024 QR refits are additional evidence.",
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID18004/session82004 exit0; initial audit79371 exit1 only at unregistered exact-count assumption after numerical comparisons; corrected full audit11895 exit0; runtime51100 exit0. All original handles polled terminal. Fresh process scan empty.",
        audit_correction="Exact reconstructed count is measured under original2e-6atol/rtol contract, not required504 bitwise matches.503 exact; one w1 difference7.45e-9. No numeric tolerance/primary model change.",
        complete_timing_fits_including_warmups=378,
        exact_timing_payloads=378,
        local_markdown_links_checked=links,
        figures_visually_inspected=["neural_readout_quality.png", "neural_readout_cost.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p) for p in scripts if p.name not in primary
        },
        verifier_dependency_sha256={
            p: sha(ROOT / p)
            for p in (
                "scripts/chromaseed_gaussian_verify.py",
                "scripts/chromaseed_gated_verify.py",
                "scripts/chromaseed_gate_stability_audit.py",
            )
        },
        shortcut_sha256=sha(SHORTCUT),
        mutable_plan_path=plan.relative_to(ROOT).as_posix(),
        environment={
            k: importlib.metadata.version(k)
            for k in ("numpy", "scipy", "pytest", "ruff", "matplotlib")
        },
        goal_classification="progress; full compact/fast/high-quality goal active; stronger-readout-regularization inventory/follow-up planned,not implemented/launched",
        evidence_limits="Analytic heads improve54/72 unchanged-network comparisons but79/84 lose to FG and all12 policies lose. Random bases reduce full construction time at worse error. Reused/confounded people/roles; no ordinary-phone face/end-to-end validation.",
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
            tests=24,
            final_rows=5020818,
            verification_sha256=sha(vp),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
