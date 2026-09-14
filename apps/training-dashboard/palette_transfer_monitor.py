"""Read-only P3 telemetry; no training imports, GPU work or inferred completion."""

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from monitor import LABELS, ROLES, process_alive

VARIANTS = ("patch5m", "soft5m", "dynamic5m")
ARMS = ("aligned", "shuffled")
PROJECT = Path(__file__).resolve().parents[2]


def positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def snapshot(
    root=Path("D:/Luma-RnD/chromaseed_palette_transfer_v1"),
    hr_root=Path("D:/Luma-RnD/chromaseed_head_range_v1"),
    project=PROJECT,
    now=None,
    probe=process_alive,
):
    root, hr_root, project = Path(root), Path(hr_root), Path(project)
    now = time.time() if now is None else now
    warnings, hashes, mtimes = [], {}, {}

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def nested(value, *keys):
        for key in keys:
            if not isinstance(value, dict):
                warnings.append("Структура настроек серии повреждена; ожидаем корректные данные.")
                return None
            value = value.get(key, {})
        return value

    def read(path):
        try:
            raw = path.read_bytes()
            value = json.loads(raw, parse_constant=invalid_constant)
            if not isinstance(value, dict):
                raise ValueError("object required")
            hashes[path] = hashlib.sha256(raw).hexdigest()
            mtimes[path] = path.stat().st_mtime
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            warnings.append(
                f"Не удалось прочитать {path.relative_to(root) if path.is_relative_to(root) else path.name}."
            )
            return {}

    registration = read(root / "registration.json")
    if not registration and not warnings:
        return None
    registration_sha = hashes.get(root / "registration.json")
    cpu, gpu = read(root / "preflight_cpu.json"), read(root / "preflight_cuda.json")

    def gate(value, device):
        return bool(
            registration_sha
            and value.get("passed") is True
            and value.get("registration_sha256") == registration_sha
            and value.get("device") == device
        )

    cpu_ok, gpu_ok = gate(cpu, "cpu"), gate(gpu, "cuda")
    read(hr_root / "source_lock.json")
    hr_job = read(hr_root / "job.json")
    hr_seal = read(project / "docs/benchmarks/chromaseed_head_range_v1/verification.json")
    hr_sha = hashes.get(hr_root / "source_lock.json")
    hr_verified = bool(
        hr_sha
        and hr_seal.get("passed") is True
        and hr_seal.get("source_lock_sha256") == hr_sha
        and hr_job.get("status") == "complete"
        and probe(hr_job.get("pid")) is False
    )
    lock = read(root / "source_lock.json")
    source = hashes.get(root / "source_lock.json")
    job, progress = read(root / "job.json"), read(root / "progress.json")
    selections = read(root / "selections.json")
    if selections and selections.get("source_lock_sha256") != source:
        warnings.append("Выбор настроек относится к другой версии серии.")
        selections = {}
    state = (
        "preparing"
        if not cpu_ok
        else "waiting_previous"
        if not hr_verified
        else "needs_gpu_check"
        if not gpu_ok
        else "ready"
    )
    if job.get("status") == "running":
        live = probe(job.get("pid"))
        if live is False:
            state = "interrupted"
        elif live is None:
            state = "unknown"
        elif not progress:
            state = "degraded" if warnings else "initializing"
        elif progress.get("pid") != job.get("pid"):
            state = "unknown"
        elif now - mtimes.get(root / "progress.json", 0) > 180:
            state = "stale"
        else:
            state = (
                progress.get("status")
                if progress.get("status") in ("training", "evaluating")
                else "initializing"
            )
        if source is None or job.get("source_lock_sha256") != source:
            state = "degraded"
            warnings.append("Версия активного запуска не подтверждена.")
    elif job.get("status") in ("failed", "complete"):
        state = job["status"]
    elif warnings:
        state = "degraded"

    entries, rates = [], {}
    final_provisional = False
    # The order is the runner's actual schedule: all inner banks, then all finals.
    for stage in ("inner", "final"):
        for role in ROLES:
            for variant in VARIANTS:
                mode = nested(lock, "heads", role, variant)
                mode = mode if mode in ("unit", "wide", "linear") else None
                for arm in ARMS:
                    pair = variant + "__" + arm
                    for fold in range(3) if stage == "inner" else (None,):
                        selected = nested(
                            selections, "roles", role, "policies", "per_pair", pair, "step"
                        )
                        provisional = stage == "final" and selected not in (128, 512, 2048)
                        final_provisional |= provisional
                        target = selected if stage == "final" and not provisional else 2048
                        suffix = "bank" if fold is None else f"fold{fold}"
                        ident = f"{stage}/{role}/{pair}/{suffix}"
                        receipt = read(root / ident / "receipt.json")
                        valid = bool(
                            receipt
                            and source
                            and receipt.get("source_lock_sha256") == source
                            and receipt.get("steps") == target
                            and receipt.get("variant") == variant
                            and receipt.get("initialization") == arm
                            and receipt.get("role") == role
                            and receipt.get("fold") == fold
                            and receipt.get("head_mode") == mode
                        )
                        if receipt and not valid:
                            warnings.append(
                                f"Пакет {ident}: сохранённые данные не совпадают с планом."
                            )
                        current = progress.get("pid") == job.get("pid") and all(
                            progress.get(k) == v
                            for k, v in dict(
                                stage=stage,
                                role=role,
                                variant=variant,
                                initialization=arm,
                                fold=fold,
                            ).items()
                        )
                        seconds = (
                            receipt.get(
                                "write_and_prediction_inclusive_seconds",
                                receipt.get("full_bank_seconds"),
                            )
                            if valid
                            else None
                        )
                        if valid and positive(seconds):
                            rates.setdefault((variant, mode), []).append(seconds / target)
                        observed_step = progress.get("step") if current else None
                        if (
                            not isinstance(observed_step, int)
                            or isinstance(observed_step, bool)
                            or not 0 <= observed_step <= target
                        ):
                            observed_step = None
                        entries.append(
                            dict(
                                id=ident,
                                stage=stage,
                                role=role,
                                variant=variant,
                                label=LABELS[variant],
                                initialization=arm,
                                head_mode=mode,
                                fold=fold,
                                target_steps=target,
                                step=target if valid else observed_step,
                                status="completed"
                                if valid
                                else "running"
                                if current and state == "training"
                                else "queued",
                                seconds=seconds,
                                current=current,
                            )
                        )
    completed = sum(p["status"] == "completed" for p in entries)
    if state == "complete" and (completed != 72 or job.get("source_lock_sha256") != source):
        state = "degraded"
        warnings.append("Завершение серии ещё не подтверждено всеми 72 пакетами.")
    seal = read(project / "docs/benchmarks/chromaseed_palette_transfer_v1/verification.json")
    verified = bool(
        state == "complete"
        and source
        and seal.get("passed") is True
        and seal.get("source_lock_sha256") == source
    )
    if verified:
        state = "verified"
    current = next((p for p in entries if p["current"]), None)
    if current:
        current = dict(current)
        losses = progress.get("minibatch_loss")
        current["loss"] = (
            statistics.mean(losses)
            if isinstance(losses, list)
            and len(losses) == 6
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses)
            else None
        )
    speed = None
    if (
        state == "training"
        and current
        and positive(current["step"])
        and positive(progress.get("seconds"))
    ):
        speed = current["step"] / progress["seconds"]

    # Forecast only native six-model banks, including prediction/write overhead.
    # CPU/synthetic preflight timings cannot stand in for production speed.
    remaining, eta_basis, fallback = 0.0, "p3", {}
    estimated = state == "training"
    if estimated:
        for item in entries:
            if item["status"] == "completed":
                continue
            pair = (item["variant"], item["head_mode"])
            times = rates.get(pair)
            if not times:
                key = (item["role"], *pair)
                if key not in fallback:
                    prior_root = (
                        project / "experiments/runs/chromaseed_architecture_scale_v1"
                        if pair[1] == "unit"
                        else hr_root
                    )
                    prior_pair = pair[0] if pair[1] == "unit" else f"{pair[0]}__{pair[1]}"
                    values = []
                    for fold in range(3):
                        old = read(
                            prior_root
                            / "inner"
                            / item["role"]
                            / prior_pair
                            / f"fold{fold}/receipt.json"
                        )
                        seconds = old.get(
                            "write_and_prediction_inclusive_seconds", old.get("full_bank_seconds")
                        )
                        if positive(seconds) and positive(old.get("steps")):
                            values.append(seconds / old["steps"])
                    fallback[key] = values
                times = fallback[key]
                eta_basis = "prior_native"
            if not times:
                estimated = False
                continue
            elapsed = progress.get("seconds", 0) if item["current"] else 0
            elapsed = elapsed if positive(elapsed) else 0
            remaining += max(0, statistics.median(times) * item["target_steps"] - elapsed)
    # Hide live signals when metadata is unreadable, retaining saved facts for inspection.
    if warnings and state in ("training", "evaluating", "initializing", "ready"):
        state, speed, estimated = "degraded", None, False
        for item in entries:
            if item["status"] == "running":
                item["status"] = "queued"
    return dict(
        state=state,
        completed_banks=completed,
        total_banks=72,
        packages=entries,
        current=current,
        steps_per_second=speed,
        estimated_remaining_seconds=remaining if estimated else None,
        eta_basis=eta_basis if estimated else None,
        final_steps_provisional=final_provisional,
        cpu=dict(
            passed=cpu_ok,
            combinations=len(cpu["records"])
            if cpu_ok and isinstance(cpu.get("records"), list)
            else 0,
        ),
        gpu=dict(passed=gpu_ok),
        hr_verified=hr_verified,
        native_quality_verified=verified,
        error=job.get("error"),
        warnings=warnings[:3],
        quality_claim="Эффект переноса палитры на точность цвета ещё не проверен."
        if not verified
        else "Итоговая проверка сохранена. Качество оценивается по ошибке цвета ΔE00; это не процент верных фотографий.",
    )
