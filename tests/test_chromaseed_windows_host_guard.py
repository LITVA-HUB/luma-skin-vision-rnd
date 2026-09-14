"""Measured WDDM activity must distinguish a desktop context from a competing fit."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chromaseed_windows_host_guard import assess, make_registry  # noqa: E402


def fixture():
    processes = [
        dict(
            ProcessId=10,
            ParentProcessId=9,
            Name="python.exe",
            CreationDate="own",
            CommandLine="python chromaseed_head_range_host_recovery.py run",
        ),
        dict(
            ProcessId=9,
            ParentProcessId=8,
            Name="python.exe",
            CreationDate="wrapper",
            CommandLine="python chromaseed_head_range_host_recovery.py run",
        ),
        dict(
            ProcessId=20,
            ParentProcessId=1,
            Name="explorer.exe",
            CreationDate="desktop",
            CommandLine=None,
        ),
        dict(
            ProcessId=4, ParentProcessId=0, Name="System", CreationDate="system", CommandLine=None
        ),
        dict(
            ProcessId=30,
            ParentProcessId=1,
            Name="python.exe",
            CreationDate="server",
            CommandLine="python apps/training-dashboard/server.py --port 8766",
        ),
    ]

    def counter(pid, engine, kind, value):
        return dict(
            instance=f"pid_{pid}_luid_0x00000000_0x00000001_phys_0_eng_{engine}_engtype_{kind}",
            value=value,
            status=0,
        )

    counters = [
        counter(20, 0, "3d", 0.4),
        counter(20, 1, "compute 0", 0),
        counter(4, 2, "copy", 0.1),
    ]
    snapshot = dict(
        gpu_uuid="GPU-test",
        driver="610.88",
        nvidia_pids=[20],
        processes=processes,
        samples=[
            dict(timestamp="one", counters=counters),
            dict(timestamp="two", counters=copy.deepcopy(counters)),
        ],
    )
    return snapshot


def test_desktop_gpu_context_is_permitted_only_with_low_measured_activity():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    value = assess(snap, registry, 10, 9)
    assert value["passed"] and value["maximum_background_engine_percent"] == 0.4
    assert value["maximum_background_compute_percent"] == 0
    assert set(registry["desktop_instances"]) == {"20", "4"}


@pytest.mark.parametrize(
    "change",
    [
        "worker",
        "unknown_gpu",
        "reused_pid",
        "driver",
        "uuid",
        "compute",
        "graphics",
        "missing_samples",
        "nan",
        "status",
        "unreadable_python",
        "missing_counter",
        "aggregate",
        "unknown_active",
    ],
)
def test_unsafe_or_unmeasurable_host_is_rejected(change):
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    if change == "worker":
        snap["processes"][-1]["CommandLine"] = "python skin_face_transfer_run.py run"
    elif change == "unknown_gpu":
        snap["nvidia_pids"].append(99)
    elif change == "reused_pid":
        snap["processes"][2]["CreationDate"] = "new desktop process"
    elif change in ("driver", "uuid"):
        snap["driver" if change == "driver" else "gpu_uuid"] = "changed"
    elif change == "compute":
        snap["samples"][1]["counters"][1]["value"] = 0.51
    elif change == "graphics":
        snap["samples"][1]["counters"][0]["value"] = 10.1
    elif change == "missing_samples":
        snap["samples"] = snap["samples"][:1]
    elif change == "nan":
        snap["samples"][0]["counters"][0]["value"] = float("nan")
    elif change == "status":
        snap["samples"][0]["counters"][0]["status"] = 0xC0000BC6
    elif change == "unreadable_python":
        snap["processes"][-1]["CommandLine"] = None
    elif change == "missing_counter":
        for sample in snap["samples"]:
            sample["counters"] = [r for r in sample["counters"] if "compute" not in r["instance"]]
    elif change == "aggregate":
        for sample in snap["samples"]:
            sample["counters"][0]["value"] = 6
            extra = copy.deepcopy(sample["counters"][0])
            extra["instance"] = extra["instance"].replace("pid_20_", "pid_4_")
            sample["counters"].append(extra)
    elif change == "unknown_active":
        snap["samples"][0]["counters"][0]["instance"] = snap["samples"][0]["counters"][0][
            "instance"
        ].replace("pid_20_", "pid_99_")
    with pytest.raises((ValueError, RuntimeError)):
        assess(snap, registry, 10, 9)


def test_own_cuda_is_excluded_but_another_compute_worker_is_not():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    snap["nvidia_pids"].append(10)
    for sample in snap["samples"]:
        sample["counters"].append(
            dict(
                instance="pid_10_luid_0x00000000_0x00000001_phys_0_eng_1_engtype_compute 0",
                value=100.0,
                status=0,
            )
        )
    assert assess(snap, registry, 10, 9)["passed"]
    snap["nvidia_pids"].append(30)
    with pytest.raises(RuntimeError):
        assess(snap, registry, 10, 9)


def test_registry_cannot_whitelist_an_arbitrary_compute_application():
    snap = fixture()
    snap["processes"][2]["Name"] = "trainer.exe"
    with pytest.raises(RuntimeError):
        make_registry(snap, 10, 9)


def test_closed_desktop_process_is_allowed_without_allowing_pid_reuse():
    snap = fixture()
    registry = make_registry(snap, 10, 9)
    snap["nvidia_pids"] = []
    snap["processes"] = [p for p in snap["processes"] if p["ProcessId"] != 20]
    for sample in snap["samples"]:
        sample["counters"] = [
            dict(
                instance="pid_4_luid_0x00000000_0x00000001_phys_0_eng_1_engtype_compute 0",
                value=0,
                status=0,
            )
        ]
    assert assess(snap, registry, 10, 9)["passed"]
