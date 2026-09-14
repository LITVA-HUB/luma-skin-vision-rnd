"""The monitor must distinguish persisted facts from live estimates."""

import csv
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from monitor import ROLES, VARIANTS, Monitor
from server import export_csv


def write(path, value, when=1000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    os.utime(path, (when, when))


def fixture(tmp_path, alive=True, now=1250):
    run = tmp_path / "experiments/runs/chromaseed_architecture_scale_v1"
    write(run / "job.json", {"status": "running", "pid": 42}, 900)
    write(run / "source_lock.json", {"variants": {v: {"parameters": 100} for v in VARIANTS}})
    monitor = Monitor(
        tmp_path,
        clock=lambda: now,
        process_probe=lambda _: alive,
        gpu_probe=lambda: {"available": True, "name": "RTX 4060", "utilization": 88},
    )
    return monitor, run


def receipt(run, role="mixed", variant="patch_small", fold=0, when=1000, seconds=1000):
    write(
        run / "inner" / role / variant / f"fold{fold}" / "receipt.json",
        {
            "variant": variant,
            "role": role,
            "fold": fold,
            "steps": 2048,
            "full_bank_seconds": seconds,
            "setup_seconds": 1,
            "slots": [[17, 1e-5], [17, 1e-4], [29, 1e-5], [29, 1e-4], [43, 1e-5], [43, 1e-4]],
            "trace": [
                {"step": 128, "seconds": 100, "minibatch_loss": [1, 2, 3, 4, 5, 6]},
                {
                    "step": 2048,
                    "seconds": seconds,
                    "minibatch_loss": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                },
            ],
        },
        when,
    )


def test_full_schedule_and_estimates_do_not_become_completed_facts(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    data = monitor.snapshot()
    assert len(data["packages"]) == 84
    assert data["progress"]["inner_total"] == 63
    assert data["progress"]["final_total"] == 21
    assert data["progress"]["completed"] == 1
    assert data["current"]["id"] == "inner/mixed/patch_small/fold1"
    assert data["current"]["estimated"]
    assert 0 < data["current"]["step"] < 2048
    assert data["gpu"]["utilization"] == 88
    assert data["traces"][0]["points"][0]["losses"] == [1, 2, 3, 4, 5, 6]


def test_stale_running_job_is_not_reported_as_live(tmp_path):
    monitor, run = fixture(tmp_path, alive=False)
    receipt(run)
    data = monitor.snapshot()
    assert data["state"] == "interrupted"
    assert data["current"] is None
    assert data["eta"]["seconds"] is None
    assert not any(p["status"] == "running" for p in data["packages"])


def test_progress_never_claims_full_completion_without_a_receipt(tmp_path):
    monitor, run = fixture(tmp_path, now=100_000)
    receipt(run)
    data = monitor.snapshot()
    assert data["current"]["step"] < data["current"]["target_steps"]
    assert data["current"]["overdue"]
    assert data["progress"]["completed"] == 1


def test_final_checkpoint_changes_are_used_in_duration_estimates(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    selection = {"roles": {r: {"policies": {v: {"step": 128} for v in VARIANTS}} for r in ROLES}}
    write(run / "selections.json", selection, 1200)
    data = monitor.snapshot()
    finals = [p for p in data["packages"] if p["stage"] == "final"]
    assert len(finals) == 21 and all(p["target_steps"] == 128 for p in finals)
    assert all(not p["steps_unknown"] for p in finals)
    assert next(p for p in finals if p["variant"] == "patch_small")["expected_seconds"] < 100


def test_malformed_metadata_returns_degraded_state_not_fake_progress(tmp_path):
    monitor, run = fixture(tmp_path)
    p = run / "inner/mixed/patch_small/fold0/receipt.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"broken":', encoding="utf-8")
    data = monitor.snapshot()
    assert data["warnings"]
    assert data["state"] == "degraded"
    assert data["current"] is None


def test_missing_gpu_reading_is_unavailable_not_zero(tmp_path):
    monitor, _ = fixture(tmp_path)
    monitor.gpu_probe = lambda: {"available": False, "error": "unavailable"}
    data = monitor.snapshot()
    assert not data["gpu"]["available"]
    assert data["gpu"].get("utilization") is None


def test_csv_uses_current_filters_and_preserves_russian_text(tmp_path):
    monitor, run = fixture(tmp_path)
    receipt(run)
    payload = export_csv(
        monitor.snapshot()["packages"],
        {"status": "completed", "role": "mixed", "query": "PATCH_SMALL"},
    )
    assert payload.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 2
    assert rows[0][0] == "Архитектура"
    assert rows[1][0] == "patch_small"
    assert rows[1][-2:] == ["Завершён", "Нет"]
