"""Final G integrity receipt; no fitting, evaluation, or changes to frozen primary sources."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote

from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs"
RUN = RUNS / "chromaseed_gated_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_gated_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-gated-2026-09-13.md"
G_LOCK = "7af66943a24be2f3d41c0f8dd5ad1261ad0126471ea0c9f97d48e9069608dd2f"
G_SELECTION = "da046187e9fa2da00c9605d1e709b28452b0cb0128349ae92b35c83cfbb44418"
G_RESULTS = "0f1da381da0ff5c882aaf29c6091fc56c179a264cd60f8c1ae2838b256a18cf5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_map(base, bindings):
    for path, expected in bindings.items():
        assert sha(base / path) == expected, str(base / path)
    return len(bindings)


def command(args, cwd=ROOT):
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    assert sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == G_LOCK
    assert sha(RUN / "selections.json") == G_SELECTION
    assert sha(RUN / "results.json") == G_RESULTS
    lock = read(RUN / "source_lock.json")
    check_map(ROOT, lock["input_sha256"])
    d_path = ROOT / "docs/benchmarks/chromaseed_camera_support_v1/verification.json"
    prior = read(d_path)
    assert prior["passed"]
    sources = []
    locks = {}
    for item in [*prior["source_locks_verified"], dict(run=RUN.name, source_lock_sha256=G_LOCK)]:
        path = RUNS / item["run"] / "source_lock.json"
        assert sha(path) == item["source_lock_sha256"]
        current = read(path)
        locks[item["run"]] = current
        assert current["cache_sha256"] == CACHE_HASH
        seen, count = {}, 0
        for key in ("sources", "source_hashes", "source_code_hashes", "compact_hashes"):
            for file, value in current.get(key, {}).items():
                canonical = Path(file).as_posix()
                assert sha(ROOT / file) == value, file
                assert canonical not in seen or seen[canonical] == value
                seen[canonical] = value
                count += 1
        assert count > 0
        sources.append(
            dict(
                run=item["run"],
                source_lock_sha256=sha(path),
                source_entries_checked=count,
                unique_source_paths_checked=len(seen),
            )
        )
    assert sum(r["source_entries_checked"] for r in sources) == 246
    assert sum(r["unique_source_paths_checked"] for r in sources) == 244
    parents = 0
    for child, parent, key in [
        ("chromaseed_kernel_v1", "skin_local_search_v1", "legacy_bindings"),
        ("chromaseed_fast_kernel_v1", "chromaseed_kernel_v1", "parent_bindings"),
        ("chromaseed_perceptual_v1", "chromaseed_fast_kernel_v1", "parent_bindings"),
        ("chromaseed_weak_ridge_v1", "chromaseed_perceptual_v1", "parent_bindings"),
        ("chromaseed_gated_v1", "chromaseed_perceptual_v1", "parent_bindings"),
    ]:
        parents += check_map(RUNS / parent, locks[child][key])
    ke = locks["chromaseed_condensed_exact_v1"]
    for field, path in {
        "source_run_lock_sha256": "experiments/runs/chromaseed_fast_kernel_v1/source_lock.json",
        "source_selections_sha256": "experiments/runs/chromaseed_fast_kernel_v1/selections.json",
        "source_results_sha256": "experiments/runs/chromaseed_fast_kernel_v1/results.json",
        "parent_audit_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/audit.json",
        "parent_runtime_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/runtime.json",
    }.items():
        assert sha(ROOT / path) == ke[field], path
        parents += 1
    for name in ("chromaseed_perceptual_v1", "chromaseed_weak_ridge_v1"):
        assert (
            sha(RUNS / "chromaseed_condensed_exact_v1/source_lock.json")
            == locks[name]["exact_source_lock_sha256"]
        )
        parents += 1
    for field, name in [
        ("parent_audit_sha256", "audit.json"),
        ("parent_verification_sha256", "verification.json"),
    ]:
        assert (
            sha(ROOT / "docs/benchmarks/chromaseed_perceptual_v1" / name)
            == locks["chromaseed_weak_ridge_v1"][field]
        )
        parents += 1
    assert parents == 163
    inputs = sum(check_map(ROOT, v.get("input_sha256", {})) for v in locks.values())
    assert inputs == 27
    prior_count = check_map(ROOT, prior["artifact_sha256"]) + check_map(
        ROOT, prior["postprocess_sources"]
    )
    prior_count += check_map(
        RUNS / "chromaseed_camera_support_v1", prior["current_numeric_file_hashes_checked"]
    )
    d_progress = read(RUNS / "chromaseed_camera_support_v1/progress.json")
    assert d_progress["status"] == "series_complete_verified" and d_progress[
        "verification_sha256"
    ] == sha(d_path)
    assert sha(RUNS / "chromaseed_camera_support_v1/results.json") == prior["results_sha256"]
    result, audit, runtime, summary = (
        read(p)
        for p in (
            RUN / "results.json",
            OUT / "audit.json",
            OUT / "runtime.json",
            OUT / "summary.json",
        )
    )
    for obj in (result, audit, runtime, summary):
        assert obj["source_lock_sha256"] == G_LOCK and obj["selection_sha256"] == G_SELECTION
    assert audit["passed"] and audit["results_sha256"] == G_RESULTS
    assert sha(ROOT / "scripts/chromaseed_gated_audit.py") == audit["audit_source_sha256"]
    check_map(ROOT, audit["dependencies"])
    check_map(ROOT, runtime["dependencies"])
    assert runtime["runtime_source_sha256"] == sha(ROOT / "scripts/chromaseed_gated_runtime.py")
    assert runtime["audit_sha256"] == summary["audit_sha256"] == sha(OUT / "audit.json")
    assert summary["results_sha256"] == G_RESULTS and summary["runtime_sha256"] == sha(
        OUT / "runtime.json"
    )
    assert summary["report_source_sha256"] == sha(ROOT / "scripts/chromaseed_gated_report.py")
    selected_hashes = {}
    for record in result["records"]:
        for area, field in (("selected", "model_sha256"), ("evaluated", "prediction_sha256")):
            path = RUN / area / record["role"] / f"{record['family']}_s{record['seed']}.npz"
            assert sha(path) == record[field]
            selected_hashes[path.relative_to(RUN).as_posix()] = record[field]
    assert len(selected_hashes) == 144
    bank_files, receipts = {}, []
    for path in sorted(RUN.rglob("receipt.json")):
        receipt = read(path)
        assert receipt["source_lock_sha256"] == G_LOCK
        check_map(path.parent, receipt["files"])
        bank_files.update(
            {(path.parent / k).relative_to(RUN).as_posix(): v for k, v in receipt["files"].items()}
        )
        receipts.append(receipt)
    assert len(receipts) == 12 and len(bank_files) == 21
    assert sum(r["residual_solutions"] for r in receipts) == 360
    assert sum(r["gate_fits"] for r in receipts) == 4
    assert sum(r["perceptual_base_solves"] for r in receipts) == 36
    assert sum(r["exact_P_base_payloads"] for r in receipts) == 72
    assert sum(r["exact_fallback_payloads"] for r in receipts) == 864
    for name, banks, readouts in (
        ("inner_complete.json", 9, 1512),
        ("final_complete.json", 3, 504),
    ):
        mark = read(RUN / name)
        assert (
            mark["source_lock_sha256"] == G_LOCK
            and mark["banks"] == banks
            and mark["readouts"] == readouts
        )
    assert read(RUN / "final_complete.json")["selection_sha256"] == G_SELECTION
    assert (
        len(runtime["records"]) == 72
        and len(runtime["standalone_fit_records"]) == 24
        and len(runtime["fixed_active_records"]) == 4
    )
    max_runtime_drift = max(
        r[impl]["max_lab_drift"] for r in runtime["records"] for impl in ("canonical", "numpy_only")
    )
    assert max_runtime_drift < 2e-8
    assert (
        runtime["selected_complete_fits_including_warmups"] == 96
        and runtime["fixed_complete_fits_including_warmups"] == 16
    )
    scripts = sorted(ROOT.glob("scripts/chromaseed_gated*.py"))
    tests = sorted(ROOT.glob("tests/test_chromaseed_gated*.py"))
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts + tests)])
    assert command(["git", "diff", "--check"]) == ""
    original = ROOT.parents[1] / "luma-skin-vision-rnd"
    assert command(["git", "status", "--short"], original) == ""
    process_query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(gated_(train|audit|runtime)|camera_support_(run|audit)|selection_stability_(run|audit))\\.py' } | Select-Object ProcessId, Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", process_query]) == ""
    docs = (
        sorted(OUT.glob("*.md"))
        + [
            ROOT / p
            for p in (
                "docs/architecture/chromaseed_gated_model_card.md",
                "docs/research/chromaseed_gated_next_decision.md",
                "docs/superpowers/plans/2026-09-13-chromaseed-gated.md",
            )
        ]
        + [SHORTCUT]
    )
    link_count = 0
    for path in docs:
        for link in re.findall(r"\]\(([^\n)]+)\)", path.read_text(encoding="utf-8")):
            target = unquote(link.strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-z]+://", target):
                continue
            resolved = (path.parent / target).resolve()
            if resolved == (OUT / "verification.json").resolve():
                link_count += 1
                continue
            assert resolved.exists(), (path, target)
            link_count += 1
    artifact_paths = [
        *sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"),
        *docs[3:-1],
    ]
    artifact_paths += [
        ROOT / "docs/architecture/chromaseed_gated_model_card.md",
        ROOT / "docs/research/chromaseed_gated_next_decision.md",
        ROOT / "docs/superpowers/plans/2026-09-13-chromaseed-gated.md",
    ]
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p) for p in artifact_paths}
    tested = [
        ROOT / "scripts/chromaseed_gated.py",
        ROOT / "scripts/chromaseed_gated_train.py",
        *tests,
        ROOT / "scripts/chromaseed_gated_numpy.py",
        ROOT / "docs/research/chromaseed_gated_v1_protocol.md",
    ]
    receipt = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=G_LOCK,
        selection_sha256=G_SELECTION,
        results_sha256=G_RESULTS,
        cache_sha256=CACHE_HASH,
        source_locks_verified=sources,
        source_entries_checked_total=246,
        unique_source_paths_per_run_checked_total=244,
        parent_artifact_bindings_checked=parents,
        diagnostic_input_bindings_checked=inputs,
        current_parent_bindings_checked=27,
        current_input_bindings_checked=2,
        parent_D_verification_sha256=sha(d_path),
        parent_D_artifacts_numeric_postprocess_checked=prior_count,
        selected_numeric_file_hashes_checked=selected_hashes,
        bank_numeric_file_hashes_checked=bank_files,
        run_marker_and_receipt_sha256={
            p.relative_to(RUN).as_posix(): sha(p)
            for p in [
                *RUN.rglob("receipt.json"),
                RUN / "inner_complete.json",
                RUN / "final_complete.json",
                RUN / "workflow.json",
            ]
        },
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        runtime_sha256=sha(OUT / "runtime.json"),
        runtime_max_native_lab_drift=max_runtime_drift,
        complete_timing_fits_including_warmups=112,
        pytest_runs=[
            dict(
                tests_passed=23,
                seconds=1.56,
                command="python -m pytest tests/test_chromaseed_gated.py tests/test_chromaseed_perceptual.py tests/test_chromaseed_perceptual_protocol.py tests/test_chromaseed_condensed_exact.py -q",
                evidence="Captured before primary fit, frozen test/core dependencies unchanged",
            ),
            dict(
                tests_passed=3,
                seconds=0.25,
                command="python -m pytest tests/test_chromaseed_gated_numpy.py -q",
                evidence="Captured after portable implementation, before runtime; tested source hashes unchanged",
            ),
        ],
        tested_sources={p.relative_to(ROOT).as_posix(): sha(p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID36812 returned exit0 directly; independent audit session27067 exit0 confirmed; runtime/report returned exit0 directly. Fresh process scan shows no G/D/S worker.",
        local_markdown_links_checked=link_count,
        figures_visually_inspected=["gated_results.png"],
        artifact_sha256=artifacts,
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name not in ("chromaseed_gated.py", "chromaseed_gated_train.py")
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad goal active; fixed-model gate stability diagnostic planned, not launched",
        evidence_limits="Six reused mixed held people; seed errors averaged; single-camera routed fit fallback is not unseen-camera detection. Ordinary-phone face quality unvalidated.",
    )
    write_json(OUT / "verification.json", receipt)
    progress = read(RUN / "progress.json")
    progress.update(
        status="series_complete_verified",
        updated_unix=time.time(),
        exit_code=0,
        verification_sha256=sha(OUT / "verification.json"),
        active_study_processes=0,
        broader_goal="active",
    )
    write_json(RUN / "progress.json", progress)
    print(
        json.dumps(
            dict(
                passed=True,
                verification_sha256=sha(OUT / "verification.json"),
                source_entries=246,
                parents=parents,
                inputs=inputs,
                selected_files=144,
                bank_files=21,
                local_links=link_count,
                runtime_drift=max_runtime_drift,
            )
        )
    )


if __name__ == "__main__":
    main()
