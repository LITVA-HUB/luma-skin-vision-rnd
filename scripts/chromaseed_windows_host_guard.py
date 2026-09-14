"""Recorded WDDM desktop activity guard; no model, optimizer or data operations."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone

from chromaseed_head_range_verification import competing_workers

DESKTOP_NAMES = frozenset(
    name.lower()
    for name in (
        "System",
        "dwm.exe",
        "Taskmgr.exe",
        "explorer.exe",
        "CrossDeviceResume.exe",
        "SearchHost.exe",
        "StartMenuExperienceHost.exe",
        "NVIDIA Overlay.exe",
        "msedgewebview2.exe",
        "TextInputHost.exe",
        "ShellExperienceHost.exe",
        "EdgeGameAssist.exe",
        "Happ.exe",
        "SnippingTool.exe",
        "ShellHost.exe",
        "chrome.exe",
        "PhoneExperienceHost.exe",
        "ChatGPT.exe",
        "codex-computer-use-swift.exe",
        "Telegram.exe",
        "msedge.exe",
    )
)
MAX_ENGINE_PERCENT = 10.0
MAX_COMPUTE_PERCENT = 0.5
COUNTER = re.compile(
    r"^pid_(\d+)_(luid_0x[0-9a-f]+_0x[0-9a-f]+_phys_\d+_eng_\d+)_engtype_(.+)$", re.I
)


def parsed_counter(row):
    match = COUNTER.fullmatch(row["instance"])
    value = float(row["value"])
    if (
        not match
        or row["status"] not in (0, 1)
        or not math.isfinite(value)
        or value < 0
        or value > 100.001
    ):
        raise ValueError("Unusable GPU Engine counter")
    return int(match[1]), match[2].lower(), match[3].lower(), value


def gpu_pids(snapshot, own):
    pids = snapshot["nvidia_pids"]
    if len(pids) != len(set(pids)) or any(type(p) is not int or p <= 0 for p in pids):
        raise ValueError("Invalid NVIDIA process inventory")
    observed = set(pids)
    for sample in snapshot["samples"]:
        for row in sample["counters"]:
            pid, _, _, value = parsed_counter(row)
            if value > 0:
                observed.add(pid)
    return observed - {own}


def identity(process):
    value = {key: process.get(key) for key in ("Name", "CreationDate")}
    if not all(isinstance(v, str) and v for v in value.values()):
        raise RuntimeError("Cannot establish desktop process identity")
    return value


def make_registry(snapshot, current_pid, parent_pid):
    by_pid = {p["ProcessId"]: p for p in snapshot["processes"]}
    desktop = {}
    for pid in sorted(gpu_pids(snapshot, current_pid)):
        process = by_pid.get(pid)
        if process is None or process["Name"].lower() not in DESKTOP_NAMES:
            raise RuntimeError(f"Not an inspected desktop application: {pid}")
        desktop[str(pid)] = identity(process)
    registry = dict(
        gpu_uuid=snapshot["gpu_uuid"],
        driver=snapshot["driver"],
        desktop_instances=desktop,
        maximum_engine_percent=MAX_ENGINE_PERCENT,
        maximum_compute_percent=MAX_COMPUTE_PERCENT,
    )
    assess(snapshot, registry, current_pid, parent_pid)
    return registry


def assess(snapshot, registry, current_pid, parent_pid):
    if any(snapshot[k] != registry[k] for k in ("gpu_uuid", "driver")):
        raise RuntimeError("GPU or driver changed from the registered host")
    if (
        registry["maximum_engine_percent"] != MAX_ENGINE_PERCENT
        or registry["maximum_compute_percent"] != MAX_COMPUTE_PERCENT
    ):
        raise ValueError("Changed registered activity limits")
    processes = snapshot["processes"]
    by_pid = {p["ProcessId"]: p for p in processes}
    if len(by_pid) != len(processes) or current_pid not in by_pid:
        raise ValueError("Incomplete or duplicated process inventory")
    python = [p for p in processes if p["Name"].lower().startswith("python")]
    others = competing_workers(python, current_pid, parent_pid)
    # The original matcher omitted the Seg2 *_run.py entry point. Keep its
    # refusal semantics and also reject every other skin research invocation.
    ancestors = {current_pid}
    ancestor = parent_pid
    while ancestor in by_pid and ancestor not in ancestors:
        ancestors.add(ancestor)
        ancestor = by_pid[ancestor]["ParentProcessId"]
    others = sorted(
        set(others)
        | {
            p["ProcessId"]
            for p in python
            if p["ProcessId"] not in ancestors and "skin_" in p["CommandLine"].lower()
        }
    )
    if others:
        raise RuntimeError(f"Competing research workers: {others}")
    background = gpu_pids(snapshot, current_pid)
    for pid in background:
        expected = registry["desktop_instances"].get(str(pid))
        process = by_pid.get(pid)
        if expected is None or process is None or identity(process) != expected:
            raise RuntimeError(f"New, unknown or changed GPU process: {pid}")
    samples = snapshot["samples"]
    if len(samples) != 2 or len({s["timestamp"] for s in samples}) != 2:
        raise ValueError("Two distinct measured GPU snapshots are required")
    maxima, compute_maxima = [], []
    for sample in samples:
        totals, compute_totals, names = {}, {}, set()
        compute_seen = False
        if not sample["counters"]:
            raise ValueError("Missing GPU counters")
        for row in sample["counters"]:
            pid, engine, kind, value = parsed_counter(row)
            if row["instance"].lower() in names:
                raise ValueError("Duplicated counter instance")
            names.add(row["instance"].lower())
            is_compute = "compute" in kind or "cuda" in kind
            compute_seen |= is_compute
            if pid == current_pid:
                continue
            totals[engine] = totals.get(engine, 0.0) + value
            if is_compute:
                compute_totals[engine] = compute_totals.get(engine, 0.0) + value
        if not compute_seen:
            raise ValueError("Compute engine coverage is unavailable")
        maxima.append(max(totals.values(), default=0.0))
        compute_maxima.append(max(compute_totals.values(), default=0.0))
    peak, compute_peak = max(maxima), max(compute_maxima)
    if peak > MAX_ENGINE_PERCENT or compute_peak > MAX_COMPUTE_PERCENT:
        raise RuntimeError(
            f"Competing GPU activity: engine={peak:.6f}%, compute={compute_peak:.6f}%"
        )
    return dict(
        passed=True,
        background_gpu_pids=sorted(background),
        samples=2,
        maximum_background_engine_percent=peak,
        maximum_background_compute_percent=compute_peak,
        current_pid=current_pid,
        parent_pid=parent_pid,
    )


def collect_snapshot():
    if os.name != "nt":
        raise RuntimeError("The registered WDDM measurement requires Windows")
    began = time.perf_counter()
    flags = subprocess.CREATE_NO_WINDOW

    def command(args, timeout=20):
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            creationflags=flags,
        ).stdout

    gpu = command(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader,nounits"]
    )
    lines = gpu.strip().splitlines()
    if len(lines) != 1 or len(lines[0].split(",")) != 2:
        raise ValueError("Expected the single registered GPU")
    gpu_uuid, driver = (part.strip() for part in lines[0].split(","))
    pids = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    script = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$gpuSamples = @(Get-Counter -Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 2 -ErrorAction Stop)
$gpuMeasurements = @($gpuSamples | ForEach-Object {
    $gpuSample = $_
    [pscustomobject]@{
        timestamp = $gpuSample.Timestamp.ToUniversalTime().ToString('o')
        counters = @($gpuSample.CounterSamples | ForEach-Object {
            [pscustomobject]@{instance=$_.InstanceName; value=$_.CookedValue; status=$_.Status}
        })
    }
})
$workerInventory = @(Get-CimInstance Win32_Process | ForEach-Object {
    $workerRow = $_
    [pscustomobject]@{
        ProcessId=$workerRow.ProcessId
        ParentProcessId=$workerRow.ParentProcessId
        Name=$workerRow.Name
        CreationDate=$(if ($workerRow.CreationDate) {$workerRow.CreationDate.ToUniversalTime().ToString('o')} else {$null})
        CommandLine=$(if ($workerRow.Name -like 'python*') {$workerRow.CommandLine} else {$null})
    }
})
[pscustomobject]@{samples=$gpuMeasurements; processes=$workerInventory} | ConvertTo-Json -Depth 8 -Compress
"""
    value = json.loads(
        command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script])
    )
    value.update(
        gpu_uuid=gpu_uuid,
        driver=driver,
        nvidia_pids=[int(p.strip()) for p in pids.splitlines() if p.strip()],
        created_utc=datetime.now(timezone.utc).isoformat(),
        collection_seconds=time.perf_counter() - began,
    )
    return value
