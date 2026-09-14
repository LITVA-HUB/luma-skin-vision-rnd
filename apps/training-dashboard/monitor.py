"""Read-only observability for the existing AS run; no ML imports or writers."""

from __future__ import annotations

import csv
import ctypes
import json
import math
import os
import statistics
import subprocess
import time
from collections import deque
from pathlib import Path

VARIANTS = (
    "patch_small",
    "patch5m",
    "soft_small",
    "soft5m",
    "dynamic_small",
    "dynamic5m",
    "pool5m",
)
ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")
LABELS = {
    "patch_small": "Участки · малая",
    "patch5m": "Участки · 4,96 млн",
    "soft_small": "Уточнения · малая",
    "soft5m": "Уточнения · 4,85 млн",
    "dynamic_small": "Динамическая · малая",
    "dynamic5m": "Динамическая · 4,85 млн",
    "pool5m": "Объединение признаков · 4,85 млн",
}


def process_alive(pid):
    """Check the actual worker, rather than trusting a stale job.json."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None if ctypes.get_last_error() == 5 else False
    try:
        code = ctypes.c_ulong()
        return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
    finally:
        kernel.CloseHandle(handle)


def query_gpu():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode:
            return {"available": False, "error": "Не удалось прочитать датчики GPU"}
        rows = list(csv.reader(result.stdout.strip().splitlines()))
        row = next((r for r in rows if "4060" in r[0]), rows[0])
        keys = (
            "utilization",
            "memory_used_mb",
            "memory_total_mb",
            "temperature",
            "power_w",
            "power_limit_w",
        )
        value = {"available": True, "name": row[0].strip()}
        for key, raw in zip(keys, row[1:], strict=True):
            try:
                number = float(raw)
                value[key] = number if math.isfinite(number) else None
            except ValueError:
                value[key] = None
        return value
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return {"available": False, "error": "Датчики GPU временно недоступны"}


class Monitor:
    def __init__(self, root, clock=time.time, process_probe=process_alive, gpu_probe=query_gpu):
        self.root = Path(root)
        self.run = self.root / "experiments/runs/chromaseed_architecture_scale_v1"
        self.out = self.root / "docs/benchmarks/chromaseed_architecture_scale_v1"
        self.clock, self.process_probe, self.gpu_probe = clock, process_probe, gpu_probe
        self.cache, self.warnings = {}, []
        self.gpu_at, self.gpu = 0, {"available": False}
        self.gpu_history = deque(maxlen=600)

    def read(self, path):
        try:
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if path in self.cache and self.cache[path][0] == signature:
                return self.cache[path][1]
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("object required")
            self.cache[path] = signature, value
            return value
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            self.warnings.append(
                f"Не удалось прочитать {path.name}; ожидаем следующего обновления."
            )
            return None

    def quality(self, selections, parameters):
        by_role = {}
        for path in self.out.joinpath("inner_reviews").glob("*.json"):
            value = self.read(path)
            if (
                not value
                or not value.get("records")
                or "person_delta_e00" not in value["records"][0]
            ):
                continue
            role = value.get("role")
            if role not in ROLES or len(value["records"]) <= by_role.get(role, {}).get(
                "candidate_count", 0
            ):
                continue
            by_role[role] = {
                "role": role,
                "rows": value.get("rows"),
                "people": value.get("people"),
                "candidate_count": len(value["records"]),
                "updated_at": path.stat().st_mtime,
                "np": value.get("prior_np_inner_delta_e00"),
                "we": value.get("prior_we_policy", {}).get("clean"),
                "models": self.quality_models(value["records"], parameters),
                "source": "inner_review",
            }
        for role, value in (selections or {}).get("roles", {}).items():
            candidates = value.get("candidates", [])
            if not candidates:
                continue
            records = [
                dict(c, person_delta_e00=c["clean"], rate=c["lr"])
                for c in candidates
                if c.get("variant") in VARIANTS
            ]
            old = by_role.get(role, {})
            by_role[role] = dict(
                old,
                role=role,
                candidate_count=len(candidates),
                source="frozen_selection",
                models=self.quality_models(records, parameters),
                np=next((c["clean"] for c in candidates if c["variant"] == "np"), None),
                we=next((c["clean"] for c in candidates if c["variant"] == "we"), None),
            )
        return list(by_role.values())

    @staticmethod
    def quality_models(records, parameters):
        result = []
        for variant in VARIANTS:
            group = [r for r in records if r.get("variant") == variant]
            if not group:
                continue
            best = min(group, key=lambda r: r["person_delta_e00"])
            result.append(
                dict(
                    variant=variant,
                    label=LABELS[variant],
                    error=best["person_delta_e00"],
                    step=best["step"],
                    rate=best.get("rate"),
                    parameters=parameters.get(variant),
                    candidates=[
                        dict(step=r["step"], rate=r.get("rate"), error=r["person_delta_e00"])
                        for r in group
                    ],
                )
            )
        return sorted(result, key=lambda r: r["error"])

    def snapshot(self):
        now = self.clock()
        self.warnings = []
        job = self.read(self.run / "job.json") or {}
        lock = self.read(self.run / "source_lock.json") or {}
        selections = self.read(self.run / "selections.json")
        parameters = {v: p.get("parameters") for v, p in lock.get("variants", {}).items()}
        packages, traces, rates, durations = [], [], {}, {}
        for stage in ("inner", "final"):
            for role in ROLES:
                for variant in VARIANTS:
                    for fold in range(3) if stage == "inner" else (None,):
                        suffix = f"fold{fold}" if fold is not None else "bank"
                        identity = f"{stage}/{role}/{variant}/{suffix}"
                        path = self.run / identity / "receipt.json"
                        saved = self.read(path)
                        policy = (
                            (selections or {})
                            .get("roles", {})
                            .get(role, {})
                            .get("policies", {})
                            .get(variant)
                        )
                        steps = 2048 if stage == "inner" else (policy or {}).get("step")
                        p = dict(
                            id=identity,
                            stage=stage,
                            role=role,
                            variant=variant,
                            label=LABELS[variant],
                            fold=fold,
                            parameters=parameters.get(variant),
                            target_steps=steps,
                            steps_unknown=steps is None,
                            status="queued",
                            step=0,
                            estimated=False,
                            seconds=None,
                            completed_at=None,
                            expected_seconds=None,
                        )
                        if (
                            saved
                            and saved.get("steps")
                            and isinstance(saved.get("full_bank_seconds"), (int, float))
                        ):
                            p.update(
                                status="completed",
                                step=saved["steps"],
                                target_steps=saved["steps"],
                                steps_unknown=False,
                                seconds=saved["full_bank_seconds"],
                                completed_at=path.stat().st_mtime,
                            )
                            durations.setdefault(variant, []).append(p["seconds"] / p["step"])
                            points = [
                                dict(
                                    step=t["step"], seconds=t["seconds"], losses=t["minibatch_loss"]
                                )
                                for t in saved.get("trace", [])
                                if "minibatch_loss" in t
                            ]
                            if points:
                                traces.append(
                                    dict(
                                        id=identity,
                                        variant=variant,
                                        role=role,
                                        stage=stage,
                                        fold=fold,
                                        completed_at=p["completed_at"],
                                        points=points,
                                        slots=saved.get("slots", []),
                                    )
                                )
                                deltas = [
                                    (b["step"] - a["step"]) / (b["seconds"] - a["seconds"])
                                    for a, b in zip(points, points[1:])
                                    if b["seconds"] > a["seconds"]
                                ]
                                if deltas:
                                    rates.setdefault(variant, []).append(statistics.median(deltas))
                        packages.append(p)
        quality = self.quality(selections, parameters)
        guessed_steps = {
            m["variant"]: m["step"] for q in quality if q["role"] == "mixed" for m in q["models"]
        }
        per_step = {v: statistics.median(values) for v, values in durations.items()}
        for p in packages:
            if p["variant"] in per_step:
                p["expected_seconds"] = per_step[p["variant"]] * (
                    p["target_steps"] or guessed_steps.get(p["variant"], 512)
                )
        done = [p for p in packages if p["status"] == "completed"]
        pending = [p for p in packages if p["status"] != "completed"]
        alive = self.process_probe(job.get("pid")) if job else False
        verification = self.read(self.out / "verification.json")
        state = (
            ("verified" if verification and verification.get("passed") else "complete")
            if job.get("status") == "complete"
            else (
                "not_started"
                if not job
                else "running"
                if alive is True
                else "interrupted"
                if alive is False
                else "unknown"
            )
        )
        if self.warnings:
            state = "degraded"
        job_path = self.run / "job.json"
        started = job_path.stat().st_mtime if job_path.exists() else None
        current = None
        # Unknown final steps imply the selector is still running: do not invent
        # an active final fit just because it is next in the schedule.
        if state == "running" and pending and not pending[0]["steps_unknown"]:
            current = pending[0]
            start = max((p["completed_at"] for p in done), default=started or now)
            elapsed = max(0, now - start)
            expected = current["expected_seconds"]
            estimate = (
                min(current["target_steps"] - 1, int(elapsed / expected * current["target_steps"]))
                if expected
                else 0
            )
            current.update(
                status="running",
                estimated=True,
                step=estimate,
                seconds=elapsed,
                overdue=bool(expected and elapsed > expected * 1.75),
                steps_per_second=statistics.median(rates.get(current["variant"], []))
                if rates.get(current["variant"])
                else None,
            )
        eta = {"seconds": None, "low_seconds": None, "high_seconds": None, "estimated": True}
        if (
            state == "running"
            and pending
            and all(p["expected_seconds"] is not None for p in pending)
        ):
            central = low = high = 0
            for p in pending:
                duration = p["expected_seconds"]
                lo = per_step[p["variant"]] * 128 if p["steps_unknown"] else duration
                hi = per_step[p["variant"]] * 2048 if p["steps_unknown"] else duration
                elapsed = p["seconds"] if p is current else 0
                # Until a receipt exists, keep a small nonzero remaining margin.
                central += max(duration - elapsed, duration * 0.03)
                low += max(lo - elapsed, lo * 0.03)
                high += max(hi - elapsed, hi * 0.1)
            eta.update(
                seconds=central,
                low_seconds=low * 0.9,
                high_seconds=high * 1.15,
                finish_at=now + central,
                final_steps_frozen=selections is not None,
            )
        if state in ("complete", "verified"):
            eta.update(seconds=0, low_seconds=0, high_seconds=0, estimated=False)
        if now - self.gpu_at >= 2.5:
            self.gpu = dict(self.gpu_probe(), sampled_at=now)
            self.gpu_at = now
            self.gpu_history.append(self.gpu.copy())
        traces.sort(key=lambda t: t["completed_at"], reverse=True)
        events = [
            dict(
                id=p["id"],
                timestamp=p["completed_at"],
                variant=p["variant"],
                role=p["role"],
                stage=p["stage"],
                fold=p["fold"],
                seconds=p["seconds"],
                steps=p["step"],
                kind="completed",
                message="Пакет завершён и сохранён",
            )
            for p in done
        ]
        events.sort(key=lambda e: e["timestamp"], reverse=True)
        return dict(
            app="luma-training-monitor",
            version=1,
            sampled_at=now,
            state=state,
            process_alive=alive,
            pid=job.get("pid"),
            run="ChromaSeed AS",
            gpu=self.gpu,
            gpu_history=list(self.gpu_history),
            current=current,
            eta=eta,
            elapsed_seconds=job.get("seconds")
            if state in ("complete", "verified")
            else max(0, now - started)
            if started
            else None,
            progress=dict(
                completed=len(done),
                total=84,
                inner_completed=sum(p["stage"] == "inner" for p in done),
                inner_total=63,
                final_completed=sum(p["stage"] == "final" for p in done),
                final_total=21,
            ),
            phase="training"
            if current
            else "selection"
            if len(done) == 63 and not selections
            else "evaluation"
            if len(done) == 84 and state == "running"
            else state,
            packages=packages,
            traces=traces,
            quality=quality,
            events=events,
            warnings=list(dict.fromkeys(self.warnings)),
            poll_seconds=3,
            measurement_note="Текущие шаги и ETA оценены по сохранённым пакетам; потери — фактические сохранённые измерения. ETA не включает независимую проверку после обучения.",
        )
