"""Versioned HR runtime host adaptation; original numerical sources remain frozen."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from chromaseed_head_range_verification import (
    CONTRACT,
    OUT,
    ROOT,
    RUN,
    check_hashes,
    digest,
    process_alive,
    read,
    require_terminal,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_windows_host_guard import assess, collect_snapshot, make_registry

RECOVERY = RUN / "verification_v1/host_recovery_v1"
PROTOCOL = RECOVERY / "protocol.json"
AUDIT_SHA = "283dad925b9f8ed36b269418fc935854d0fb3b2d396e26e4f0b760e0fb9e90df"
FILES = (
    "scripts/chromaseed_windows_host_guard.py",
    "scripts/chromaseed_head_range_host_recovery.py",
    "tests/test_chromaseed_windows_host_guard.py",
    "tests/test_chromaseed_head_range_host_recovery.py",
    "docs/research/chromaseed_head_range_host_recovery_v1_protocol.md",
)
NUMERICAL = (
    "response",
    "replay",
    "fit",
    "upstream_fit",
    "Predictor",
    "predict_torch",
    "compare",
    "exact",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def install_runtime(module, guard, supplement, protocol_sha, evidence):
    original = {name: getattr(module, name) for name in NUMERICAL}
    original_contract, original_binding, original_write = (
        module.verify_contract,
        module.binding,
        module.write_once,
    )

    def bound_contract(**kwargs):
        value = original_contract(**kwargs)
        module.check_hashes(supplement)
        return {**value, "sources": {**value["sources"], **supplement}}

    def binding():
        return {**original_binding(), "host_guard_protocol_sha256": protocol_sha}

    def write(path, value):
        if path == module.OUT / "runtime.json":
            artifacts, summary = evidence()
            value = {
                **value,
                "artifact_sha256": {**value["artifact_sha256"], **supplement, **artifacts},
                "host_guard": summary,
            }
        return original_write(path, value)

    module.verify_contract, module.binding, module.write_once = bound_contract, binding, write
    module.require_quiet_host = guard
    assert all(getattr(module, name) is value for name, value in original.items())


def render_disclosure(original, protocol_sha):
    return original + (
        "\n## Условия измерения на Windows\n\n"
        "Первый запуск измерений остановился до первого вызова модели: исходная проверка считала все GPU-контексты Windows конкурентными вычислениями. "
        "В WDDM у системных окон и браузеров также есть такие контексты. Полное время неудачного запуска не было записано; оно не принято за ноль и не включено во время обучения.\n\n"
        "Применён отдельно зафиксированный адаптер проверки среды. Перед каждым измерением или полным повторным обучением сохранены два секундных снимка GPU Engine и список процессов. "
        "Разрешены только заранее зарегистрированные фоновые процессы с теми же PID, именем и временем создания. "
        "Нагрузка прочих процессов на вычислительный движок не превышала 0,5%, на любой движок в сумме — 10%; другие исследовательские процессы запрещены. "
        "Собственный процесс измерения исключён из фоновой нагрузки. Это локальные замеры при работающем рабочем столе, а проверки перед запуском не являются непрерывным контролем всей длительности. "
        "Численные функции, 189 измерений, 42 полных повторения и допуски сохранены. Затраты на проверку среды отделены от времени вызова и обучения.\n\n"
        f"Контракт адаптера SHA-256: `{protocol_sha}`. Все снимки и исходники адаптера включены в итоговые привязки файлов.\n"
    )


def supplement():
    value = read(PROTOCOL)
    check_hashes(value["bindings"])
    return value, {**value["bindings"], str(PROTOCOL): digest(PROTOCOL)}


def freeze():
    if PROTOCOL.exists():
        raise RuntimeError("Preserve registered host protocol")
    require_terminal(read(RUN / "job.json"))
    verify_contract(full_inputs=False)
    if digest(OUT / "audit.json") != AUDIT_SHA:
        raise ValueError("Unexpected completed independent audit")
    if (
        (OUT / "runtime.json").exists()
        or (OUT / "verification.json").exists()
        or list((RUN / "verification_v1/runtime").rglob("*.json"))
    ):
        raise RuntimeError("This recovery is for the zero-measurement failure only")
    failed = RUN / "full_runtime_v1.log"
    # PowerShell redirection may use UTF-16 or UTF-8; retain bytes without re-encoding.
    raw = failed.read_bytes()
    failure_text = raw.decode(
        "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    )
    if "Competing GPU processes:" not in failure_text:
        raise ValueError("Unexpected initial failure")
    RECOVERY.mkdir(parents=True, exist_ok=True)
    commands = [
        [sys.executable, "-X", "utf8", "-m", "pytest", *FILES[2:4], "-q"],
        [sys.executable, "-X", "utf8", "-m", "ruff", "check", *FILES[:4]],
    ]
    checks = []
    for command in commands:
        proc = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180
        )
        checks.append(
            dict(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        )
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
    write_once(RECOVERY / "checks.json", dict(passed=True, checks=checks))
    snapshot = collect_snapshot()
    registry = make_registry(snapshot, os.getpid(), os.getppid())
    write_once(
        RECOVERY / "baseline.json",
        dict(
            snapshot=snapshot,
            registry=registry,
            assessment=assess(snapshot, registry, os.getpid(), os.getppid()),
        ),
    )
    files = [ROOT / name for name in FILES]
    files += [
        CONTRACT,
        OUT / "audit.json",
        RUN / "results.json",
        RUN / "selections.json",
        RUN / "source_lock.json",
        failed,
        RECOVERY / "checks.json",
        RECOVERY / "baseline.json",
    ]
    value = dict(
        created_utc=utc(),
        bindings={str(p): digest(p) for p in files},
        registry=registry,
        original_verification_protocol_sha256=digest(CONTRACT),
        audit_sha256=AUDIT_SHA,
        original_failure=dict(
            actual_tool_session=54282,
            actual_exit_code=1,
            responses=0,
            recipes=0,
            complete_elapsed_seconds=None,
        ),
        numerical_change=False,
        runtime_responses=189,
        runtime_recipes=42,
        host_checks=232,
        single_attempt=True,
        original_functions_preserved=list(NUMERICAL),
        scope="WDDM host gate and additive provenance only; no numerical or selection change",
    )
    write_once(PROTOCOL, value)
    print(
        "HR HOST RECOVERY FROZEN",
        digest(PROTOCOL),
        "desktop instances",
        len(registry["desktop_instances"]),
        flush=True,
    )


def guard_evidence(protocol, launch, paths):
    if len(paths) != protocol["host_checks"]:
        raise ValueError("Incomplete registered host-check coverage")
    collection_seconds, engine, compute = 0.0, 0.0, 0.0
    for path in paths:
        record = read(path)
        decision = assess(
            record["snapshot"], protocol["registry"], launch["pid"], launch["parent_pid"]
        )
        if decision != record["assessment"] or record["protocol_sha256"] != digest(PROTOCOL):
            raise ValueError("Host-check replay changed")
        collection_seconds += record["snapshot"]["collection_seconds"]
        engine = max(engine, decision["maximum_background_engine_percent"])
        compute = max(compute, decision["maximum_background_compute_percent"])
    artifacts = {str(p): digest(p) for p in [RECOVERY / "launch.json", *paths]}
    summary = dict(
        passed=True,
        protocol_sha256=digest(PROTOCOL),
        checks=len(paths),
        collection_seconds=collection_seconds,
        maximum_background_engine_percent=engine,
        maximum_background_compute_percent=compute,
        scope="before each measurement/recipe; desktop active; not continuous isolation",
        original_functions_preserved=list(NUMERICAL),
        original_failure_elapsed_seconds=None,
    )
    return artifacts, summary


def run():
    protocol, bindings = supplement()
    require_terminal(read(RUN / "job.json"))
    if (
        (RECOVERY / "launch.json").exists()
        or (OUT / "runtime.json").exists()
        or (OUT / "verification.json").exists()
    ):
        raise RuntimeError("Single attempt only; preserve prior runtime evidence")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    import chromaseed_head_range_runtime as runtime

    paths = []
    launch = dict(
        pid=os.getpid(),
        parent_pid=os.getppid(),
        utc=utc(),
        protocol_sha256=digest(PROTOCOL),
        actual_functions=list(NUMERICAL),
        adapter="require_quiet_host and additive provenance",
    )
    write_once(RECOVERY / "launch.json", launch)

    def guard():
        snapshot = collect_snapshot()
        path = RECOVERY / "snapshots" / f"{len(paths):04d}.json"
        try:
            decision = assess(snapshot, protocol["registry"], launch["pid"], launch["parent_pid"])
        except (ValueError, RuntimeError) as exc:
            write_once(
                path,
                dict(
                    passed=False,
                    snapshot=snapshot,
                    error=str(exc),
                    protocol_sha256=digest(PROTOCOL),
                ),
            )
            raise
        write_once(
            path,
            dict(
                passed=True,
                snapshot=snapshot,
                assessment=decision,
                protocol_sha256=digest(PROTOCOL),
            ),
        )
        paths.append(path)

    install_runtime(
        runtime, guard, bindings, digest(PROTOCOL), lambda: guard_evidence(protocol, launch, paths)
    )
    began = time.perf_counter()
    try:
        runtime.main()
    except BaseException as exc:
        write_once(
            RECOVERY / "completion.json",
            dict(
                passed=False,
                pid=launch["pid"],
                seconds=time.perf_counter() - began,
                error=repr(exc),
                completed_host_checks=len(paths),
                utc=utc(),
            ),
        )
        raise
    write_once(
        RECOVERY / "completion.json",
        dict(
            passed=True,
            pid=launch["pid"],
            seconds=time.perf_counter() - began,
            runtime_sha256=digest(OUT / "runtime.json"),
            utc=utc(),
        ),
    )


def verify():
    protocol, bindings = supplement()
    launch = read(RECOVERY / "launch.json")
    if process_alive(launch["pid"]) is not False:
        raise RuntimeError("Wait for the actual runtime process to terminate")
    completion = read(RECOVERY / "completion.json")
    if completion.get("passed") is not True or completion["runtime_sha256"] != digest(
        OUT / "runtime.json"
    ):
        raise ValueError("Runtime attempt did not complete")
    runtime = verify_stage("runtime.json")
    paths = sorted((RECOVERY / "snapshots").glob("*.json"))
    artifacts, summary = guard_evidence(protocol, launch, paths)
    if runtime["host_guard"] != summary:
        raise ValueError("Runtime host evidence differs")
    for path, expected in {**artifacts, **bindings}.items():
        if runtime["artifact_sha256"].get(path) != expected:
            raise ValueError("Effective adapter/evidence omitted from runtime binding")
    print("HR HOST RECOVERY VERIFIED", summary, flush=True)
    return runtime


def report():
    verify()
    import chromaseed_head_range_report as reporter

    original = reporter.render
    reporter.render = lambda summary: render_disclosure(original(summary), digest(PROTOCOL))
    original_stage = reporter.verify_stage

    def stage(name):
        value = original_stage(name)
        if name == "runtime.json":
            value = {
                **value,
                "artifact_sha256": {
                    **value["artifact_sha256"],
                    str(RECOVERY / "completion.json"): digest(RECOVERY / "completion.json"),
                },
            }
        return value

    reporter.verify_stage = stage
    reporter.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run", "verify", "report"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify, "report": report}[args.action]()
