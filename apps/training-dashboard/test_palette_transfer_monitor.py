"""Exercise persisted completion, lineage, liveness and estimate boundaries."""

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import ROLES
from palette_transfer_monitor import ARMS, VARIANTS, snapshot


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (1000, 1000))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    root = tmp_path / "p3"
    registration = write(root / "registration.json", {"bindings": {}})
    write(
        root / "preflight_cpu.json",
        dict(
            passed=True,
            registration_sha256=registration,
            device="cpu",
            records=[{}] * 27,
            original_exact_payloads=108,
            transferred_prefix_exact_payloads=108,
        ),
    )
    options = dict(
        root=root, hr_root=tmp_path / "hr", project=tmp_path, now=1100, probe=lambda _: True
    )
    return root, registration, options


def live_job(root, options):
    heads = {role: {v: "wide" for v in VARIANTS} for role in ROLES}
    source = write(root / "source_lock.json", dict(heads=heads))
    write(root / "job.json", dict(status="running", pid=42, source_lock_sha256=source))
    progress = dict(
        status="training",
        pid=42,
        stage="inner",
        role="mixed",
        variant="patch5m",
        initialization="aligned",
        head_mode="wide",
        fold=0,
        step=2048,
        target_steps=2048,
        seconds=100,
        completed_banks=72,
    )
    write(root / "progress.json", progress)
    return source, progress


def receipt(root, source, role="mixed", variant="patch5m", arm="aligned", fold=0, steps=2048):
    stage = "final" if fold is None else "inner"
    path = (
        root
        / stage
        / role
        / f"{variant}__{arm}"
        / ("bank" if fold is None else f"fold{fold}")
        / "receipt.json"
    )
    write(
        path,
        dict(
            role=role,
            variant=variant,
            initialization=arm,
            fold=fold,
            head_mode="wide",
            steps=steps,
            source_lock_sha256=source,
            write_and_prediction_inclusive_seconds=steps / 10,
        ),
    )
    return path


def test_prepared_is_waiting_not_training_or_accurate(tmp_path):
    root, _, options = fixture(tmp_path)
    row = snapshot(**options)
    assert row["state"] == "waiting_previous"
    assert row["completed_banks"] == 0 and len(row["packages"]) == 72
    assert row["cpu"]["passed"] and row["cpu"]["combinations"] == 27
    assert row["steps_per_second"] is None and row["estimated_remaining_seconds"] is None
    assert row["current"] is None and not row["native_quality_verified"]
    assert len({p["id"] for p in row["packages"]}) == 72
    assert sum(p["stage"] == "inner" for p in row["packages"]) == 54
    assert not any(p["initialization"] == "original" for p in row["packages"])


def test_parent_and_gpu_gates_require_matching_receipts(tmp_path):
    root, registration, options = fixture(tmp_path)
    source = write(options["hr_root"] / "source_lock.json", {"sources": {}})
    write(options["hr_root"] / "job.json", dict(status="complete", pid=99))
    seal = tmp_path / "docs/benchmarks/chromaseed_head_range_v1/verification.json"
    write(seal, dict(passed=True, source_lock_sha256="wrong"))
    options["probe"] = lambda _: False
    assert snapshot(**options)["state"] == "waiting_previous"
    write(seal, dict(passed=True, source_lock_sha256=source))
    assert snapshot(**options)["state"] == "needs_gpu_check"
    write(
        root / "preflight_cuda.json",
        dict(passed=True, registration_sha256=registration, device="cuda"),
    )
    assert snapshot(**options)["state"] == "ready"
    options["probe"] = lambda _: True
    assert snapshot(**options)["state"] == "waiting_previous"


@pytest.mark.parametrize(
    "alive,now,state",
    [(False, 1100, "interrupted"), (None, 1100, "unknown"), (True, 1400, "stale")],
)
def test_dead_unknown_or_stale_never_show_live_speed(tmp_path, alive, now, state):
    root, _, options = fixture(tmp_path)
    live_job(root, options)
    options.update(probe=lambda _: alive, now=now)
    row = snapshot(**options)
    assert row["state"] == state
    assert row["steps_per_second"] is None and row["estimated_remaining_seconds"] is None
    assert not any(p["status"] == "running" for p in row["packages"])


def test_steps_and_reported_counter_do_not_replace_receipts(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    row = snapshot(**options)
    assert row["state"] == "training" and row["steps_per_second"] == 20.48
    assert row["completed_banks"] == 0 and row["packages"][0]["status"] == "running"
    assert row["estimated_remaining_seconds"] is None
    receipt(root, source)
    receipt(root, "foreign-source", fold=1)
    row = snapshot(**options)
    assert row["completed_banks"] == 1 and row["packages"][0]["status"] == "completed"
    assert row["warnings"]


def test_pid_mismatch_and_malformed_json_degrade_without_crashing(tmp_path):
    root, _, options = fixture(tmp_path)
    _, progress = live_job(root, options)
    write(root / "progress.json", dict(progress, pid=7))
    assert snapshot(**options)["state"] == "unknown"
    (root / "progress.json").write_text("{partial", encoding="utf-8")
    row = snapshot(**options)
    assert row["state"] == "degraded" and row["steps_per_second"] is None
    write(root / "progress.json", [])
    assert snapshot(**options)["warnings"]


def test_eta_uses_native_receipts_and_selected_final_lengths(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    for variant in VARIANTS:
        receipt(root, source, variant=variant, fold=2)
    row = snapshot(**options)
    assert row["estimated_remaining_seconds"] == pytest.approx(69 * 204.8 - 100)
    assert row["final_steps_provisional"] and row["eta_basis"] == "p3"
    selection = dict(
        source_lock_sha256=source,
        roles={
            role: {
                "policies": {
                    "per_pair": {v + "__" + arm: {"step": 128} for v in VARIANTS for arm in ARMS}
                }
            }
            for role in ROLES
        },
    )
    write(root / "selections.json", selection)
    row = snapshot(**options)
    assert not row["final_steps_provisional"]
    assert row["estimated_remaining_seconds"] == pytest.approx(51 * 204.8 + 18 * 12.8 - 100)


def test_matched_hr_times_are_explicitly_provisional(tmp_path):
    root, _, options = fixture(tmp_path)
    live_job(root, options)
    for role in ROLES:
        for variant in VARIANTS:
            path = options["hr_root"] / "inner" / role / (variant + "__wide") / "fold0/receipt.json"
            write(path, dict(steps=2048, write_and_prediction_inclusive_seconds=204.8))
    row = snapshot(**options)
    assert row["eta_basis"] == "prior_native"
    assert row["estimated_remaining_seconds"] == pytest.approx(72 * 204.8 - 100)


def test_complete_needs_all_bank_receipts_and_a_seal(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    write(root / "job.json", dict(status="complete", pid=42, source_lock_sha256=source))
    options["probe"] = lambda _: False
    assert snapshot(**options)["state"] == "degraded"
    for role in ROLES:
        for variant in VARIANTS:
            for arm in ARMS:
                for fold in [0, 1, 2, None]:
                    receipt(root, source, role, variant, arm, fold)
    assert snapshot(**options)["state"] == "complete"
    seal = tmp_path / "docs/benchmarks/chromaseed_palette_transfer_v1/verification.json"
    write(seal, dict(passed=True, source_lock_sha256="old"))
    assert not snapshot(**options)["native_quality_verified"]
    write(seal, dict(passed=True, source_lock_sha256=source))
    assert snapshot(**options)["state"] == "verified"


def test_absent_registration_does_not_invent_a_study(tmp_path):
    assert snapshot(root=tmp_path, project=tmp_path, hr_root=tmp_path / "hr") is None


def test_malformed_nested_selection_is_visible_and_does_not_break_api(tmp_path):
    root, _, options = fixture(tmp_path)
    source, _ = live_job(root, options)
    write(root / "selections.json", dict(source_lock_sha256=source, roles=[]))
    row = snapshot(**options)
    assert row["state"] == "degraded" and row["warnings"]
    json.dumps(row, allow_nan=False)
