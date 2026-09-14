"""Publish local verified quality/size evidence; later invocations only reverify hashes."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from chromaseed_kernel_audit import js, nz
from chromaseed_long_training_run import NP
from chromaseed_refine_audit import error_summary
from chromaseed_widen_run import OUT, ROOT, RUN, bank_path, check_map, load_data
from skin_local_search_train import sha, write_json

ROLE_NAMES = ("mixed", "slr_to_ipod", "ipod_to_slr")
VARIANTS = ("np", "tiny", "m31", "m61", "m111", "m832")


def main():
    receipt = OUT / "verification.json"
    if receipt.exists():
        value = js(receipt)
        check_map(
            {
                **value["sources"],
                **value["inputs"],
                **value["postprocess_sources"],
                **value["artifact_sha256"],
            }
        )
        print("WIDE READ-ONLY VERIFIED", sha(receipt), flush=True)
        return
    lock, selection, result = [
        js(RUN / n) for n in ("source_lock.json", "selections.json", "results.json")
    ]
    audit, runtime = js(OUT / "audit.json"), js(OUT / "runtime.json")
    assert audit["passed"] and runtime["passed"]
    assert audit["results_sha256"] == runtime["results_sha256"] == sha(RUN / "results.json")
    assert runtime["audit_sha256"] == sha(OUT / "audit.json")
    check_map(
        {
            **lock["sources"],
            **lock["input_sha256"],
            **audit["artifact_sha256"],
            **audit["dependencies"],
            **runtime["dependencies"],
        }
    )
    checks = []
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_chromaseed_widen.py",
            "tests/test_chromaseed_widen_audit.py",
            "-q",
        ],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "scripts/chromaseed_widen.py",
            "scripts/chromaseed_widen_run.py",
            "scripts/chromaseed_widen_audit.py",
            "scripts/chromaseed_widen_runtime.py",
            "scripts/chromaseed_widen_report.py",
            "tests/test_chromaseed_widen.py",
            "tests/test_chromaseed_widen_audit.py",
        ],
    ]
    for command in commands:
        output = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        checks.append(
            dict(
                command=command[1:],
                exit_code=output.returncode,
                stdout=output.stdout,
                stderr=output.stderr,
            )
        )
        assert output.returncode == 0, output.stdout + output.stderr
    write_json(OUT / "checks.json", dict(passed=True, checks=checks))
    data = load_data()
    rows = []
    selected = [r for r in result["records"] if r["stress_evaluated"]]
    patient_errors = {}
    for variant in VARIANTS:
        recs = [r for r in selected if r["variant"] == variant]
        timing = [t for t in runtime["records"] if t["variant"] == variant]
        item = dict(
            variant=variant,
            parameters=recs[0]["parameters"],
            numeric_bytes=recs[0]["numeric_bytes"],
            archive_bytes_min=min(r["archive_bytes"] for r in recs),
            archive_bytes_max=max(r["archive_bytes"] for r in recs),
            median_cpu_us=float(np.median([t["median_us"] for t in timing])),
            roles={},
        )
        for role in ROLE_NAMES:
            rr = [r for r in recs if r["role"] == role]
            assert len(rr) == 3
            fit = next(
                v for v in runtime["recipes"] if v["variant"] == variant and v["role"] == role
            )
            per = []
            for r in rr:
                saved = nz(RUN / "evaluated" / role / f"{r['name']}.npz")
                ix = saved["row_indices"]
                per.append(
                    error_summary(
                        saved["predictions"][0],
                        data["target"][ix],
                        data["patient"][ix],
                        data["site"][ix],
                    )[1]
                )
            patient_errors[variant, role] = np.mean(per, axis=0)
            item["roles"][role] = dict(
                delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                seed_errors=[r["metrics"]["person_mean"] for r in rr],
                step=rr[0]["step"],
                lr=rr[0]["lr"],
                complete_three_seed_recipe_seconds=fit[
                    "median_reconstruction_and_validation_seconds"
                ],
            )
        rows.append(item)
    comparisons = []
    for vi, variant in enumerate(VARIANTS[1:]):
        for ri, role in enumerate(ROLE_NAMES):
            diff = patient_errors[variant, role] - patient_errors["np", role]
            rng = np.random.default_rng(834001 + 100 * vi + ri)
            draws = rng.integers(0, len(diff), (20000, len(diff)))
            low, high = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
            comparisons.append(
                dict(
                    variant=variant,
                    role=role,
                    people=len(diff),
                    difference_from_np=float(diff.mean()),
                    paired_person_bootstrap_95=[float(low), float(high)],
                )
            )
    historical = js(NP / "results.json")
    fg = {
        role: float(
            np.mean(
                [
                    r["metrics"]["person_mean"]
                    for r in historical["records"]
                    if r["role"] == role and r["family"] == "fg_norm_static"
                ]
            )
        )
        for role in ROLE_NAMES
    }
    assert np.isfinite(list(fg.values())).all()
    banks = [
        js(bank_path(role, v, f) / "receipt.json")
        for role in ROLE_NAMES
        for v in VARIANTS[1:]
        for f in (0, 1, 2, None)
    ]
    overall = {}
    for role in ROLE_NAMES:
        c = selection["roles"][role]["policies"]["overall"]
        matching = [
            r for r in result["records"] if r["role"] == role and "overall" in r["policies"]
        ]
        assert len(matching) == 3
        overall[role] = dict(
            variant=c["variant"],
            step=c["step"],
            lr=c["lr"],
            inner_delta_e00=c["clean"],
            outer_delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in matching])),
        )
    summary = dict(
        rows=rows,
        paired_comparisons=comparisons,
        overall=overall,
        historical_fg=fg,
        research_bank_seconds=sum(b["full_bank_seconds"] for b in banks),
        research_workflow_seconds=js(RUN / "job.json")["seconds"],
        gpu_peak_allocated_bytes=max(b["cuda_peak_allocated_bytes"] for b in banks),
        audit_counts=audit["counts"],
        upstream_reconstructions=runtime["upstream_singleton_fits"],
        selected_reconstructions=runtime["selected_payloads_bitwise"],
        full_bank_reconstructions=runtime["all_bank_payloads_bitwise"],
        interpretation="Exploratory reused roles; per-capacity choices selected only inside fit rows. Bootstrap is conditional on those choices; camera and person composition are confounded. No new ordinary-phone facial evidence.",
    )
    write_json(OUT / "summary.json", summary)
    largest = next(r for r in rows if r["variant"] == "m832")
    baseline = next(r for r in rows if r["variant"] == "np")
    changes = {
        role: f"{baseline['roles'][role]['delta_e00']:.2f} → {largest['roles'][role]['delta_e00']:.2f}"
        for role in ROLE_NAMES
    }
    lines = [
        "# Luma ChromaSeed: приоритет качества",
        "",
        f"Модель на 832 259 параметров улучшила смешанный сценарий ({changes['mixed']} ΔE00) и перенос iPod → SLR ({changes['ipod_to_slr']}). Перенос SLR → iPod ухудшился ({changes['slr_to_ipod']}). Увеличение размера дало полезные результаты, но общего победителя для всех условий пока нет.",
        "",
        "Проверены модели от 1 179 до 832 259 параметров. Все варианты обучены с тремя случайными инициализациями; настройки обучения выбраны по отдельным людям внутри обучающей части. Размер не ограничивал выбор по качеству.",
        "",
        "Приведённый результат самой крупной модели — сравнение конкретного размера. Общий алгоритм выбора по внутренней проверке предпочёл m111 / m61 / m111; он не выбирал m832. Эти решения сохранены, несмотря на более привлекательные внешние результаты m832 в двух сценариях.",
        "",
        "Ошибка — средняя по людям ΔE00, затем средняя трёх seed. **Меньше — лучше. Это не процент точности.**",
        "",
        "| Модель | Параметры | Данные модели, КБ | Смешанная группа | SLR → iPod | iPod → SLR | CPU, мкс |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        errors = " | ".join(f"{r['roles'][role]['delta_e00']:.4f}" for role in ROLE_NAMES)
        lines.append(
            f"| {r['variant']} | {r['parameters']:,} | {r['numeric_bytes'] / 1000:.2f} | {errors} | {r['median_cpu_us']:.1f} |"
        )
    lines.extend(
        [
            "",
            "NP — ранее проверенная исходная модель. tiny — маленький вариант в текущем сравнении. m31/m61/m111/m832 — расширенные модели. Для каждой строки показана заранее выбранная внутри обучения настройка, а не лучший результат после просмотра внешней группы.",
            "",
            "Исторический проверенный контроль FG на этих же ролях: "
            + " / ".join(f"{fg[k]:.4f}" for k in ROLE_NAMES)
            + " ΔE00. Его обучение в этой серии не повторялось; он приведён как контекст, чтобы не принимать выигрыш у NP за превосходство над всеми предыдущими методами.",
            "",
            "## Что выбрано внутри обучения",
            "",
            "| Роль | Модель | Шаги | Скорость обучения | Ошибка внутри | Ошибка снаружи |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, c in overall.items():
        lines.append(
            f"| {role} | {c['variant']} | {c['step']} | {c['lr']} | {c['inner_delta_e00']:.4f} | {c['outer_delta_e00']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Обучение и воспроизводимость",
            "",
            f"Полная серия:60 банков,360 траекторий, до8192 обновлений по64 примера. Это повторные предъявления966 существующих наблюдений24 людей. Чистое время всех исследовательских банков: {summary['research_bank_seconds']:.1f} с; весь первичный процесс с сохранением и оценкой: {summary['research_workflow_seconds']:.1f} с.",
            "",
            "| Модель | Шаги: смешанная / SLR→iPod / iPod→SLR | Полная сборка трёх seed, секунды |",
            "| --- | --- | --- |",
        ]
    )
    for r in rows:
        steps = " / ".join(str(r["roles"][k]["step"]) for k in ROLE_NAMES)
        times = " / ".join(
            f"{r['roles'][k]['complete_three_seed_recipe_seconds']:.3f}" for k in ROLE_NAMES
        )
        lines.append(f"| {r['variant']} | {steps} | {times} |")
    lines.extend(
        [
            "",
            "В стоимость сборки включены три исходных обучения ND, экспорт NP и весь банк из шести продолжений с исходным расписанием. Время не делится на число моделей. CPU измеряет один ответ со всеми64 участками, включая их нормализацию и обработку; декодирование фото и извлечение признаков сюда не входят.",
            "",
            f"Независимая проверка:1080 внутренних моделей,204000 внутренних ответов,93 оценки кандидатов,18 выборов;369 конечных/контрольных моделей и711612 вызовов настоящего потребителя на стресс-преобразованиях. Максимальное расхождение независимого расчёта: {audit['max_consumer_native_lab']:.3g} Lab. Полностью повторены162 исходных обучения;162 выбранных экспорта и270 полных банковых экспортов совпали побитно. Результаты16 тестов и линтера сохранены в checks.json.",
            "",
            "## Как читать результат",
            "",
            "Роли переиспользованы в предыдущих исследованиях; группы людей и камер связаны. Доверительные интервалы парной разницы по людям находятся в summary.json и условны на уже сделанном выборе. На смешанной внешней группе только6 людей, на переносах16 и8. Эти данные показывают свойства моделей на имеющемся наборе, но не подтверждают качество на обычных селфи покупателей.",
            "",
            "P8 завершил первичное обучение до этой серии, но его отдельный аудит и измерение полной стоимости отложены после смены приоритета. WIDE не использует его как проверенного победителя. Старые результаты и отрицательные эксперименты сохранены.",
            "",
            "[Инвентаризация предыдущих крупных моделей и прямого обучения по ΔE00](../../research/chromaseed_quality_inventory_during_widen.md) поясняет, какие направления уже были проверены и почему их цифры нельзя смешивать с текущими ролями.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    card = ROOT / "docs/architecture/chromaseed_widen_model_card.md"
    card.write_text(
        "# Luma ChromaSeed WIDE\n\nResearch skin-region Lab regression, not face identification. One color36 vector plus64x18 existing patch descriptors. Shared local ReLU encoder, mean/spread/max pooling, wider ReLU color head. FP32 payload, cached FP64 NumPy consumer. Capacities1179/30915/60611/110979/832259; initial function matches the corresponding NP head. No ensemble inference.\n\nOnly original TRAIN966rows24people; three historically reused person-disjoint role splits, inner selection. Instrument-native D65/10-degree Lab. Synthetic affine stress retains original target by assumption. No new ordinary-phone face/end-to-end validation or Skolkovo eligibility claim.\n\nSee ../benchmarks/chromaseed_widen_v1/report.md and summary.json for all selected capacities, adverse transfer outcomes, actual CPU response and full warm-training costs. Do not export a favorable outer checkpoint in place of the frozen selection. Previous pixel/R/pooling architectures already exist; this is a matched capacity study, not a novelty proof.\n",
        encoding="utf-8",
    )
    decision = ROOT / "docs/research/chromaseed_widen_next_decision.md"
    decision.write_text(
        "# Next quality decision\n\nUser priority is maximum instrument-referenced skin-color quality; small size is secondary. This capacity comparison is verified progress, not completion of the broad goal. Inspect the selected policy and all per-capacity results before deciding the next architecture. Greater parameter count alone is not evidence of better generalization.\n\nPreserve all WIDE/P8/WA/LT/NP/ND and earlier source locks. WIDE report.py is a read-only verifier after sealing; never rerun bound primary/audit/runtime writers. P8 independent audit/runtime remain unfinished and explicitly deferred by the quality-priority change. No GPU worker should be assumed live without its actual session status.\n\nRead chromaseed_quality_inventory_during_widen.md: existing CNN/pixel-vote/copula and true CIEDE2000-objective studies were inspected during WIDE; direct perceptual loss already exists and has mixed outcomes. WIDE inner selection puts11/12 larger-capacity choices at the smallest positive checkpoint512, and all15 capacity choices at the lower rate0.0001. Before concluding that more capacity cannot help, preregister earlier checkpoints and lower rates with preserved full-bank shape and NP warm lineage; this directly tests an observed search boundary. Deeper representations or perceptual objectives remain later adaptations of existing work, not first use. No follow-up code or fitting is launched by this decision. Do not automatically repeat an old experiment, choose a known outer checkpoint or acquire new data/weights without checking the applicable scope. Ordinary-phone face accuracy still needs new unexposed, appropriately licensed measured evidence. No next fit is implemented or launched by this note.\n",
        encoding="utf-8",
    )
    post = {
        p: sha(ROOT / p)
        for p in (
            "scripts/chromaseed_widen_audit.py",
            "scripts/chromaseed_widen_runtime.py",
            "scripts/chromaseed_widen_report.py",
            "tests/test_chromaseed_widen_audit.py",
            "docs/research/chromaseed_quality_inventory_during_widen.md",
            "scripts/skin_mskcc_train_pixels.py",
            "scripts/skin_copula_model.py",
            "scripts/skin_capture_model.py",
            "docs/benchmarks/skin_mskcc_pixels_v1/report.md",
            "docs/benchmarks/skin_copula_v1/report.md",
            "docs/benchmarks/skin_capture_v1/report.md",
            "docs/research/skin_capture_protocol_v1.md",
            card.relative_to(ROOT).as_posix(),
            decision.relative_to(ROOT).as_posix(),
        )
    }
    artifacts = dict(audit["artifact_sha256"])
    for name in ("audit.json", "runtime.json", "checks.json", "summary.json", "report.md"):
        p = OUT / name
        artifacts[p.relative_to(ROOT).as_posix()] = sha(p)
    value = dict(
        passed=True,
        source_lock_sha256=sha(RUN / "source_lock.json"),
        selection_sha256=sha(RUN / "selections.json"),
        results_sha256=sha(RUN / "results.json"),
        sources=lock["sources"],
        inputs=lock["input_sha256"],
        postprocess_sources=post,
        artifact_sha256=artifacts,
        classification="verified progress; broader quality goal active",
        p8="primary complete, audit/runtime deferred; not accepted",
    )
    check_map({**value["sources"], **value["inputs"], **post, **artifacts})
    write_json(receipt, value)
    print("WIDE SEALED", sha(receipt), flush=True)
    print("WIDE QUALITY", overall, flush=True)


if __name__ == "__main__":
    main()
