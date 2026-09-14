"""Freeze the GS verification receipt after independent numerical and source checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

from chromaseed_gated_verify import check_map, command, read
from skin_local_search_train import CACHE_HASH, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "experiments/runs"
RUN = RUNS / "chromaseed_gate_stability_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_gate_stability_v1"
SHORTCUT = ROOT.parents[1] / "output/luma-chromaseed-stability-2026-09-13.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    expected_source = "5102e3ba875d0a21f71d13dd538edf55b04b1efc0deb4110d19d66522c935b26"
    expected_result = "41e8a9060ebf12eed131534e5aced4ac61605688537c27f0a0ea9050ef9a4bc0"
    assert sha(args.cache) == CACHE_HASH
    assert sha(RUN / "source_lock.json") == expected_source
    assert sha(RUN / "results.json") == expected_result
    lock, result = read(RUN / "source_lock.json"), read(RUN / "results.json")
    assert check_map(ROOT, lock["input_sha256"]) == 148
    parent_path = ROOT / "docs/benchmarks/chromaseed_gated_v1/verification.json"
    assert sha(parent_path) == "9b73c11efed3d4fae6e6e41f4987377a68fc841af37a88984194f879a1d00065"
    parent = read(parent_path)
    assert parent["passed"]
    source_checks, locks = [], {}
    for entry in [
        *parent["source_locks_verified"],
        dict(run=RUN.name, source_lock_sha256=expected_source),
    ]:
        path = RUNS / entry["run"] / "source_lock.json"
        assert sha(path) == entry["source_lock_sha256"]
        bound = read(path)
        locks[entry["run"]] = bound
        assert bound["cache_sha256"] == CACHE_HASH
        count, seen = 0, {}
        for field in ("sources", "source_hashes", "source_code_hashes", "compact_hashes"):
            for p, h in bound.get(field, {}).items():
                assert sha(ROOT / p) == h, p
                key = Path(p).as_posix()
                assert key not in seen or seen[key] == h
                seen[key] = h
                count += 1
        source_checks.append(
            dict(
                run=entry["run"],
                source_lock_sha256=sha(path),
                source_entries_checked=count,
                unique_source_paths_checked=len(seen),
            )
        )
    assert sum(r["source_entries_checked"] for r in source_checks) == 296
    assert sum(r["unique_source_paths_checked"] for r in source_checks) == 294
    binding_count = 0
    for child, par, key in [
        ("chromaseed_kernel_v1", "skin_local_search_v1", "legacy_bindings"),
        ("chromaseed_fast_kernel_v1", "chromaseed_kernel_v1", "parent_bindings"),
        ("chromaseed_perceptual_v1", "chromaseed_fast_kernel_v1", "parent_bindings"),
        ("chromaseed_weak_ridge_v1", "chromaseed_perceptual_v1", "parent_bindings"),
        ("chromaseed_gated_v1", "chromaseed_perceptual_v1", "parent_bindings"),
    ]:
        binding_count += check_map(RUNS / par, locks[child][key])
    for field, file in {
        "source_run_lock_sha256": "experiments/runs/chromaseed_fast_kernel_v1/source_lock.json",
        "source_selections_sha256": "experiments/runs/chromaseed_fast_kernel_v1/selections.json",
        "source_results_sha256": "experiments/runs/chromaseed_fast_kernel_v1/results.json",
        "parent_audit_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/audit.json",
        "parent_runtime_sha256": "docs/benchmarks/chromaseed_fast_kernel_v1/runtime.json",
    }.items():
        assert sha(ROOT / file) == locks["chromaseed_condensed_exact_v1"][field]
        binding_count += 1
    for name in ("chromaseed_perceptual_v1", "chromaseed_weak_ridge_v1"):
        assert (
            sha(RUNS / "chromaseed_condensed_exact_v1/source_lock.json")
            == locks[name]["exact_source_lock_sha256"]
        )
        binding_count += 1
    for field, name in (
        ("parent_audit_sha256", "audit.json"),
        ("parent_verification_sha256", "verification.json"),
    ):
        assert (
            sha(ROOT / "docs/benchmarks/chromaseed_perceptual_v1" / name)
            == locks["chromaseed_weak_ridge_v1"][field]
        )
        binding_count += 1
    assert binding_count == 163
    diagnostic_inputs = sum(
        check_map(ROOT, bound.get("input_sha256", {})) for bound in locks.values()
    )
    assert diagnostic_inputs == 175
    parent_checks = check_map(ROOT, parent["artifact_sha256"]) + check_map(
        ROOT, parent["postprocess_sources"]
    )
    for key in (
        "selected_numeric_file_hashes_checked",
        "bank_numeric_file_hashes_checked",
        "run_marker_and_receipt_sha256",
    ):
        parent_checks += check_map(RUNS / "chromaseed_gated_v1", parent[key])
    progress_g = read(RUNS / "chromaseed_gated_v1/progress.json")
    assert (
        progress_g["verification_sha256"] == sha(parent_path)
        and progress_g["status"] == "series_complete_verified"
    )
    dpath = ROOT / "docs/benchmarks/chromaseed_camera_support_v1/verification.json"
    assert sha(dpath) == parent["parent_D_verification_sha256"]
    d = read(dpath)
    check_map(ROOT, d["artifact_sha256"])
    check_map(ROOT, d["postprocess_sources"])
    check_map(RUNS / "chromaseed_camera_support_v1", d["current_numeric_file_hashes_checked"])
    assert check_map(RUN, result["numeric_sha256"]) == 78
    audit, summary = read(OUT / "audit.json"), read(OUT / "summary.json")
    assert audit["passed"]
    for obj in (audit, summary):
        assert (
            obj["source_lock_sha256"] == expected_source
            and obj["results_sha256"] == expected_result
        )
    assert audit["audit_source_sha256"] == sha(ROOT / "scripts/chromaseed_gate_stability_audit.py")
    check_map(ROOT, audit["dependencies"])
    assert summary["report_source_sha256"] == sha(
        ROOT / "scripts/chromaseed_gate_stability_report.py"
    )
    assert summary["audit_sha256"] == sha(OUT / "audit.json")
    assert (
        audit["checks"]["transformed_query_rows"] == 948816
        and audit["checks"]["boundary_control_query_rows"] == 25632
    )
    assert (
        audit["maxima"]["ordinary_prediction"] <= 2e-8
        and audit["maxima"]["boundary_prediction"] <= 0.001
    )
    assert summary["geometry"]["all_six_boundary_input_sets_exactly_equal"]
    assert summary["geometry"]["unique_query_anchor_pairs"] == 534
    assert len(summary["rows"]) == 24 and len(summary["boundary"]) == 8
    scripts = sorted(ROOT.glob("scripts/chromaseed_gate_stability*.py"))
    test = ROOT / "tests/test_chromaseed_gate_stability.py"
    ruff = command([sys.executable, "-m", "ruff", "check", *map(str, scripts), str(test)])
    assert command(["git", "diff", "--check"]) == ""
    assert command(["git", "status", "--short"], ROOT.parents[1] / "luma-skin-vision-rnd") == ""
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|uv' -and $_.CommandLine -match 'chromaseed_(gate_stability_(run|audit)|gated_(train|audit|runtime))\\.py' } | Select-Object ProcessId, Name | ConvertTo-Json -Compress"
    assert command(["powershell", "-NoProfile", "-Command", query]) == ""
    extra = [
        ROOT / p
        for p in (
            "docs/research/chromaseed_gate_stability_next_decision.md",
            "docs/architecture/chromaseed_gate_stability_addendum.md",
            "docs/superpowers/plans/2026-09-13-chromaseed-gate-stability.md",
        )
    ]
    docs = sorted(OUT.glob("*.md")) + extra + [SHORTCUT]
    count = 0
    for path in docs:
        for link in re.findall(r"\]\(([^\n)]+)\)", path.read_text(encoding="utf-8")):
            target = unquote(link.strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-z]+://", target):
                continue
            destination = (path.parent / target).resolve()
            assert destination == (OUT / "verification.json").resolve() or destination.exists(), (
                path,
                target,
            )
            count += 1
    paths = [p for p in OUT.iterdir() if p.is_file() and p.name != "verification.json"] + extra
    tested = [
        "scripts/chromaseed_gate_stability.py",
        "scripts/chromaseed_gate_stability_run.py",
        "tests/test_chromaseed_gate_stability.py",
        "docs/research/chromaseed_gate_stability_v1_protocol.md",
    ]
    verification = dict(
        passed=True,
        completed_unix=time.time(),
        source_lock_sha256=expected_source,
        results_sha256=expected_result,
        previous_goal_turn_classification="progress: G model/runtime/audit artifacts and terminal process state revalidated before GS",
        cache_sha256=CACHE_HASH,
        source_locks_verified=source_checks,
        source_entries_checked_total=296,
        unique_source_paths_per_run_checked_total=294,
        parent_artifact_bindings_checked=163,
        diagnostic_input_bindings_checked=175,
        current_input_bindings_checked=148,
        parent_G_verification_sha256=sha(parent_path),
        parent_G_artifacts_numeric_postprocess_checked=parent_checks,
        current_numeric_file_hashes_checked=result["numeric_sha256"],
        independent_audit_sha256=sha(OUT / "audit.json"),
        independent_audit_checks=audit["checks"],
        independent_audit_maxima=audit["maxima"],
        tests_passed=8,
        pytest_seconds=1.51,
        pytest_command="python -m pytest tests/test_chromaseed_gate_stability.py -q",
        pytest_evidence="Captured final8-pass run after formatting and before diagnostic; primary sources unchanged",
        tested_sources={p: sha(ROOT / p) for p in tested},
        ruff=ruff,
        git_diff_check="passed",
        original_repository_git_status="clean",
        active_study_processes=0,
        terminal_evidence="Primary PID39468 returned exit0 directly; independent audit session1166 exit0 confirmed through its handle; report returned exit0. Fresh scan has no GS/G worker.",
        local_markdown_links_checked=count,
        figures_visually_inspected=["gate_stability.png"],
        artifact_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
        postprocess_sources={
            p.relative_to(ROOT).as_posix(): sha(p)
            for p in scripts
            if p.name not in ("chromaseed_gate_stability.py", "chromaseed_gate_stability_run.py")
        },
        verifier_dependency_sha256={
            "scripts/chromaseed_gated_verify.py": sha(ROOT / "scripts/chromaseed_gated_verify.py")
        },
        shortcut_sha256=sha(SHORTCUT),
        environment={
            k: importlib.metadata.version(k) for k in ("numpy", "scipy", "pytest", "ruff")
        },
        goal_classification="progress; broad goal active; mild fit-side affine augmentation learning with matched controls planned, not launched",
        evidence_limits="Frozen models, reused TRAIN and synthetic transforms. Constructed pairs may be far from original inputs; soft continuity does not solve shared color sensitivity. Ordinary-phone face accuracy unvalidated.",
    )
    write_json(OUT / "verification.json", verification)
    progress = read(RUN / "progress.json")
    progress.update(
        status="series_complete_verified",
        exit_code=0,
        updated_unix=time.time(),
        verification_sha256=sha(OUT / "verification.json"),
        active_study_processes=0,
        broader_goal="active",
    )
    write_json(RUN / "progress.json", progress)
    print(
        dict(
            passed=True,
            verification_sha256=sha(OUT / "verification.json"),
            source_entries=296,
            parent_artifacts=163,
            diagnostic_inputs=175,
            current_numeric_files=78,
            local_links=count,
        )
    )


if __name__ == "__main__":
    main()
