"""Independent HR contracts, terminal guards, selection and cost accounting."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/runs/chromaseed_head_range_v1"
OUT = ROOT / "docs/benchmarks/chromaseed_head_range_v1"
CONTRACT = RUN / "verification_v1/protocol.json"
AS_RUN = ROOT / "experiments/runs/chromaseed_architecture_scale_v1"
AS_OUT = ROOT / "docs/benchmarks/chromaseed_architecture_scale_v1"
SOURCE = "dfb96f902658009f57d37478ee9e2964c951a2bbce8aaba9acda77ff142652b2"
AS_SEAL = "ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9"
RECOVERY = "5952cf01a7a77db9f8945fed32f09294d5002d62a8d9dee5953c41308c3f637b"
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
MODES = ("unit", "wide", "linear")
SEEDS = (17, 29, 43)
RATES = (1e-5, 1e-4)
TIMES = (128, 512, 2048)
PARAMETERS = dict(
    patch_small=17374,
    patch5m=4962566,
    soft_small=15246,
    soft5m=4846822,
    dynamic_small=15246,
    dynamic5m=4846822,
    pool5m=4851846,
)
FILES = (
    "scripts/chromaseed_head_range_verification.py",
    "scripts/chromaseed_head_range_audit.py",
    "scripts/chromaseed_head_range_runtime.py",
    "scripts/chromaseed_head_range_report.py",
    "tests/test_chromaseed_head_range_verification.py",
    "docs/research/chromaseed_head_range_verification_v1_protocol.md",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_once(path, value):
    """Never replace an existing receipt, including one from an interrupted stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if read(path) != value:
            raise RuntimeError(f"Preserve existing artifact: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def check_hashes(mapping):
    for path, expected in mapping.items():
        if digest(ROOT / path) != expected:
            raise AssertionError(f"Changed bound file: {path}")


def remember(path, artifacts, expected=None):
    path = Path(path)
    actual = digest(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"Artifact hash mismatch: {path}")
    key = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    if key in artifacts and artifacts[key] != actual:
        raise AssertionError(f"Artifact changed during verification: {path}")
    artifacts[key] = actual
    return actual


def process_alive(pid):
    """True/False/unknown; permission errors never count as a dead worker."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return None
        return True
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return False if ctypes.get_last_error() == 87 else None
    try:
        state = kernel.WaitForSingleObject(handle, 0)
        return {0: False, 258: True}.get(state)
    finally:
        kernel.CloseHandle(handle)


def require_terminal(job, probe=process_alive):
    pid = job.get("pid")
    if (
        job.get("status") != "complete"
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or probe(pid) is not False
    ):
        raise RuntimeError("HR primary must be complete and its recorded worker confirmed dead")


def competing_workers(processes, current_pid, parent_pid):
    by_pid = {p["ProcessId"]: p for p in processes}
    ancestors = {current_pid}
    pid = parent_pid
    while pid in by_pid and pid not in ancestors:
        ancestors.add(pid)
        pid = by_pid[pid]["ParentProcessId"]
    others = []
    for p in processes:
        if p["ProcessId"] in ancestors:
            continue
        command = p.get("CommandLine")
        if not command:
            raise RuntimeError(f"Cannot inspect another Python worker: {p['ProcessId']}")
        command = command.lower()
        if "chromaseed_" in command or (
            "skin_" in command
            and any(
                kind in command for kind in ("_train", "_fit", "_runtime", "_audit", "_benchmark")
            )
        ):
            others.append(p["ProcessId"])
    return others


def require_quiet_host():
    """Fail closed before benchmarks/replay when a competing workload is observed."""
    own = os.getpid()
    if os.name != "nt":
        raise RuntimeError("This registered benchmark requires its original Windows host")
    command = (
        "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        "Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%'\" | "
        "Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    snapshot = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    processes = json.loads(snapshot.stdout)
    if isinstance(processes, dict):
        processes = [processes]
    others = competing_workers(processes, own, os.getppid())
    if others:
        raise RuntimeError(f"Competing research workers: {others}")
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    others = [int(p.strip()) for p in query.stdout.splitlines() if p.strip()]
    if any(p != own for p in others):
        raise RuntimeError(f"Competing GPU processes: {others}")


def compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(actual, expected, atol=0.002, rtol=1e-6)
    return float(np.max(np.abs(actual - expected)))


def rank(candidates):
    if not candidates:
        raise ValueError("Missing registered candidate group")
    for c in candidates:
        values = (c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Candidate scores and selection keys must be finite and nonnegative")
    return min(
        candidates,
        key=lambda c: (c["clean"], c["p90"], c["numeric_bytes"], c["step"], c["lr"] or 0),
    )


def select_policies(candidates):
    overall = rank(candidates)  # Validate all rows, including losing candidates.
    return dict(
        per_pair={
            v + "__" + m: rank([c for c in candidates if c["variant"] == v + "__" + m])
            for v in PARAMETERS
            for m in MODES
        },
        per_architecture={
            v: rank([c for c in candidates if c.get("architecture") == v]) for v in PARAMETERS
        },
        overall=overall,
    )


def index_records(records):
    result = {}
    for record in records:
        key = (record["role"], record["variant"], record["seed"])
        if key in result:
            raise ValueError(f"Duplicate record identity: {key}")
        result[key] = record
    return result


def operational_costs(banks, failed_partial, resumed_seconds):
    for bank in banks:
        assert bank["steps"] > 0
        assert 0 <= bank["setup_seconds"] <= bank["full_bank_seconds"]
        assert bank["full_bank_seconds"] <= bank["write_and_prediction_inclusive_seconds"]
    successful = sum(b["full_bank_seconds"] for b in banks)
    discarded = float(failed_partial["seconds"])
    assert math.isfinite(discarded) and discarded >= 0
    assert math.isfinite(resumed_seconds) and resumed_seconds >= 0
    return dict(
        successful_banks=len(banks),
        successful_trajectories=6 * len(banks),
        successful_bank_seconds=successful,
        successful_setup_seconds=sum(b["setup_seconds"] for b in banks),
        successful_write_inclusive_seconds=sum(
            b["write_and_prediction_inclusive_seconds"] for b in banks
        ),
        observed_fit_work_seconds=successful + discarded,
        discarded_partial_seconds=discarded,
        discarded_partial_steps=failed_partial["step"],
        successful_presentations=sum(b["steps"] for b in banks) * 64 * 6,
        discarded_presentations=failed_partial["step"] * 64 * 6,
        peak_allocated_cuda_bytes=max(b["cuda_peak_allocated_bytes"] for b in banks),
        resumed_attempt_seconds=resumed_seconds,
        complete_original_wall_seconds=None,
        scope="setup is inside bank time; write-inclusive is another overlapping scope, not additive; no bank division by six",
    )


def recovery_bindings():
    path = RUN / "recovery_v1"
    protocol = read(path / "protocol.json")
    assert digest(path / "protocol.json") == RECOVERY
    assert protocol["original_source_lock_sha256"] == SOURCE
    artifacts = {
        **protocol["source_bindings"],
        **protocol["preserved_failure"],
        **protocol["completed_bank_files"],
    }
    for name in ("protocol.json", "check_protocol.json", "check.json", "launch.json"):
        remember(path / name, artifacts)
    check, contract = read(path / "check.json"), read(path / "check_protocol.json")
    assert check["status"] == "passed" and check["previous_banks_unchanged"] == 18
    assert check["preserved_files_checked"] == 162 and check["replayed_same_step"]["exact"]
    assert check["check_protocol_sha256"] == digest(path / "check_protocol.json")
    assert contract["recovery_protocol_sha256"] == RECOVERY
    assert contract["previous_progress_sha256"] == digest(path / "failed_progress.json")
    remember(
        ROOT / "scripts/chromaseed_head_range_recovery_check.py",
        artifacts,
        contract["source_sha256"],
    )
    remember(
        RUN / "inner/mixed/soft5m__wide/fold0/receipt.json",
        artifacts,
        check["resumed_bank_receipt_sha256"],
    )
    partial = read(path / "failed_progress.tmp")
    assert partial["step"] == 1664 and partial["seconds"] == 357.0525681999861
    check_hashes(artifacts)
    return artifacts


def freeze_contract():
    assert digest(RUN / "source_lock.json") == SOURCE
    assert digest(AS_OUT / "verification.json") == AS_SEAL
    assert not (OUT / "verification.json").exists()
    lock = read(RUN / "source_lock.json")
    bindings = {name: digest(ROOT / name) for name in FILES}
    recovery = recovery_bindings()
    check_hashes(lock["sources"])
    value = dict(
        source_lock_sha256=SOURCE,
        as_verification_sha256=AS_SEAL,
        sources=bindings,
        recovery_bindings=recovery,
        expected_counts=dict(
            inner_banks=126,
            inner_models=2268,
            inner_vectors=428400,
            candidates=384,
            choices=87,
            final_banks=42,
            final_bank_payloads=252,
            final_models=126,
            final_vectors=50316,
            actual_single_calls=50316,
            inherited_candidates=126,
            inherited_records=81,
        ),
        runtime_counts=dict(
            responses=189,
            timed_calls=36288,
            upstream_fits=126,
            continuation_banks=42,
            selected_payloads_bitwise=126,
            all_bank_payloads_bitwise=252,
        ),
        atol_native_lab=0.002,
        rtol=1e-6,
        primary_tool_session="13677; caller must observe actual terminal exit 0 before full audit",
        classification="verification consumers prepared; no quality gain or primary completion implied",
    )
    write_once(CONTRACT, value)
    return value


def verify_contract(full_inputs=True):
    contract = read(CONTRACT)
    assert contract["source_lock_sha256"] == digest(RUN / "source_lock.json") == SOURCE
    assert contract["as_verification_sha256"] == digest(AS_OUT / "verification.json") == AS_SEAL
    lock = read(RUN / "source_lock.json")
    check_hashes({**lock["sources"], **contract["sources"], **contract["recovery_bindings"]})
    if full_inputs:
        check_hashes(lock["input_sha256"])
    return contract


def verify_stage(name):
    value = read(OUT / name)
    assert value["passed"] and value["verification_protocol_sha256"] == digest(CONTRACT)
    assert value["results_sha256"] == digest(RUN / "results.json")
    check_hashes(value.get("artifact_sha256", {}))
    check_hashes(value["dependencies"])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "guard", "verify"))
    args = parser.parse_args()
    if args.action == "guard":
        require_terminal(read(RUN / "job.json"))
        print("HR primary is complete; its worker is dead", flush=True)
    elif args.action == "freeze":
        freeze_contract()
        print("HR verification consumers frozen", digest(CONTRACT), flush=True)
    else:
        verify_contract()
        print("HR verification contract unchanged", digest(CONTRACT), flush=True)


if __name__ == "__main__":
    main()
