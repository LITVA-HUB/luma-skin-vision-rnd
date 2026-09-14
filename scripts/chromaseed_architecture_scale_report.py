"""Seal the measured AS screen after independent audit and complete training replay."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from chromaseed_architecture_scale_run import OUT, ROOT, RUN, VARIANTS, WE_OUT, check_map, load_data
from chromaseed_kernel_audit import js, nz
from chromaseed_refine_audit import error_summary
from skin_local_search_train import sha, write_json

ROLES = ("mixed", "slr_to_ipod", "ipod_to_slr")


def main():
    seal = OUT / "verification.json"
    if seal.exists():
        v = js(seal)
        check_map(
            {**v["sources"], **v["inputs"], **v["postprocess_sources"], **v["artifact_sha256"]}
        )
        print("AS READ-ONLY VERIFIED", sha(seal), flush=True)
        return
    lock, selection, results = [
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
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_architecture_scale.py", "-q"],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "scripts/chromaseed_architecture_scale.py",
            "scripts/chromaseed_architecture_scale_run.py",
            "scripts/chromaseed_architecture_scale_audit.py",
            "scripts/chromaseed_architecture_scale_runtime.py",
            "scripts/chromaseed_architecture_scale_report.py",
            "tests/test_chromaseed_architecture_scale.py",
        ],
    ]
    checks = []
    for command in commands:
        proc = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        checks.append(
            dict(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    write_json(OUT / "checks.json", checks)
    data = load_data()
    rows, people = [], {}
    for variant in (*VARIANTS, "np", "we"):
        row = dict(variant=variant, roles={})
        for role in ROLES:
            rr = [r for r in results["records"] if r["role"] == role and r["variant"] == variant]
            assert len(rr) == 3
            if variant in VARIANTS:
                row["parameters"] = rr[0]["parameters"]
                row["numeric_bytes"] = rr[0]["numeric_bytes"]
                timing = [
                    r for r in runtime["responses"] if r["role"] == role and r["variant"] == variant
                ]
                recipe = next(
                    r for r in runtime["recipes"] if r["role"] == role and r["variant"] == variant
                )
                person = []
                pass_errors = []
                for r in rr:
                    out = nz(ROOT / r["output"])
                    ix = out["row_indices"]
                    m, p = error_summary(
                        out["predictions"],
                        data["target"][ix],
                        data["patient"][ix],
                        data["site"][ix],
                    )
                    person.append(p)
                    pass_errors.append(
                        [
                            error_summary(
                                pred, data["target"][ix], data["patient"][ix], data["site"][ix]
                            )[0]["person_mean"]
                            for pred in out["passes"].transpose(1, 0, 2)
                        ]
                    )
                people[variant, role] = np.mean(person, axis=0)
                extra = dict(
                    cpu_median_us=float(np.median([t["median_us"] for t in timing])),
                    complete_three_seed_bank_seconds=recipe["build_seconds"],
                    per_pass_delta_e00=np.mean(pass_errors, axis=0).tolist(),
                )
            else:
                extra = dict(timing="imported WE control; no new timing claim")
            row["roles"][role] = dict(
                delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                seeds=[r["metrics"]["person_mean"] for r in rr],
                step=rr[0]["step"],
                lr=rr[0]["lr"],
                **extra,
            )
        rows.append(row)
    pairs = []
    for fi, family in enumerate(("patch", "soft", "dynamic")):
        for ri, role in enumerate(ROLES):
            delta = people[family + "5m", role] - people[family + "_small", role]
            draws = np.random.default_rng(701003 + fi * 100 + ri).integers(
                0, len(delta), size=(20000, len(delta))
            )
            interval = np.quantile(delta[draws].mean(1), [0.025, 0.975])
            pairs.append(
                dict(
                    family=family,
                    role=role,
                    large_minus_small=float(delta.mean()),
                    conditional_paired_person_95=interval.tolist(),
                    people=len(delta),
                    people_improved=int(np.sum(delta < 0)),
                )
            )
    overall = {}
    for role in ROLES:
        c = selection["roles"][role]["policies"]["overall"]
        selected = [r for r in results["records"] if r["role"] == role and r["overall"]]
        assert len(selected) == 3
        overall[role] = dict(
            variant=c["variant"],
            parent_variant=c.get("parent_variant"),
            step=c["step"],
            lr=c["lr"],
            inner_delta_e00=c["clean"],
            delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in selected])),
        )
    banks = [
        js(p) for kind in ("inner", "final") for p in sorted((RUN / kind).rglob("receipt.json"))
    ]
    assert len(banks) == 84
    counts = dict(
        better=sum(p["large_minus_small"] < -1e-9 for p in pairs),
        worse=sum(p["large_minus_small"] > 1e-9 for p in pairs),
        same=sum(abs(p["large_minus_small"]) <= 1e-9 for p in pairs),
    )
    summary = dict(
        rows=rows,
        scaling_pairs=pairs,
        scaling_counts=counts,
        overall=overall,
        research_banks=len(banks),
        research_trajectories=6 * len(banks),
        research_presentations=sum(b["steps"] * 64 * 6 for b in banks),
        research_bank_seconds=sum(b["full_bank_seconds"] for b in banks),
        research_workflow_seconds=js(RUN / "job.json")["seconds"],
        cuda_peak_allocated_bytes=max(b["cuda_peak_allocated_bytes"] for b in banks),
        audit_counts=audit["counts"],
        maximum_native_lab=audit["maximum_native_lab"],
        limitations="reused/confounded original TRAIN roles, small cohort, conditional comparisons, no ordinary-phone face validation",
    )
    write_json(OUT / "summary.json", summary)
    lines = [
        "# Luma ChromaSeed AS: прежние архитектуры при размере около 5 млн параметров",
        "",
        f"Завершено сравнение семи вариантов. В девяти сопоставлениях крупной и малой версии одной архитектуры: {counts['better']} улучшений, {counts['worse']} ухудшений, {counts['same']} совпадений. Настройки выбраны внутри обучающей части до внешней оценки.",
        "",
        "Все числа качества — ошибка ΔE00: меньше лучше. Сначала усредняем по людям, затем по трём отдельным запускам; ансамбля предсказаний нет.",
        "",
        "| Модель | Параметры | Весовые данные, МБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        errors = " | ".join(f"{row['roles'][role]['delta_e00']:.4f}" for role in ROLES)
        if row["variant"] in VARIANTS:
            latency = np.median([row["roles"][role]["cpu_median_us"] for role in ROLES])
            lines.append(
                f"| {row['variant']} | {row['parameters']:,} | {row['numeric_bytes'] / 1e6:.3f} | {errors} | {latency:.1f} |"
            )
        else:
            lines.append(f"| {row['variant']} — прежний контроль | — | — | {errors} | — |")
    lines.extend(
        [
            "",
            "NP — фиксированная исходная модель; WE — прежняя политика выбора, поэтому её архитектура зависит от роли. Малые и большие версии AS обучались одинаково. Их начальный ответ совпадает с NP; её 643 замороженных параметра включены в размер. Старые результаты R с другим стартовым алгоритмом напрямую в эту таблицу не подставлялись.",
            "",
            "## Выбор по внутренним данным",
            "",
            "| Роль | Выбранный вариант | Шаги | Скорость | Внутренняя ошибка | Внешняя ошибка |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for role, c in overall.items():
        label = c["variant"] + (f" ({c['parent_variant']})" if c["parent_variant"] else "")
        lines.append(
            f"| {role} | {label} | {c['step']} | {c['lr']} | {c['inner_delta_e00']:.4f} | {c['delta_e00']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Это заранее определённая политика выбора. Более удачная строка внешней таблицы не заменяет её задним числом. Все 132 оценки и 24 решения сохранены в selections.json.",
            "",
            "## Что даёт увеличение размера",
            "",
            "| Семейство | Роль | Большая минус малая, ΔE00 | Условный интервал 95% | Людей с улучшением |",
            "| --- | --- | ---: | --- | ---: |",
        ]
    )
    for p in pairs:
        ci = p["conditional_paired_person_95"]
        lines.append(
            f"| {p['family']} | {p['role']} | {p['large_minus_small']:+.4f} | [{ci[0]:+.4f}; {ci[1]:+.4f}] | {p['people_improved']}/{p['people']} |"
        )
    lines.extend(
        [
            "",
            "Интервалы построены по фиксированным парным ошибкам людей после выбора модели. Они не учитывают многократное исследование этих же групп. Внешние группы содержат только 6, 16 и 8 людей; камера и состав людей связаны. Это исследовательское сравнение, а не независимое подтверждение качества на покупателях.",
            "",
            "## Четыре уточнения и стоимость",
            "",
            "| Модель | Роль | Ошибки после каждого прохода | Полная сборка трёх запусков, с |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in rows:
        if row["variant"] not in VARIANTS:
            continue
        for role in ROLES:
            r = row["roles"][role]
            passes = " → ".join(f"{e:.4f}" for e in r["per_pass_delta_e00"])
            lines.append(
                f"| {row['variant']} | {role} | {passes} | {r['complete_three_seed_bank_seconds']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Всегда выполняются зарегистрированные проходы. Таблица промежуточных ответов — диагностика; по ней не выбиралась внешняя точка остановки. Динамическое внимание меняет используемые участки, но их кодирование и расчёт оценок остаются плотными.",
            "",
            "Время сборки — один полный повтор трёх исходных обучений ND, экспорта NP и всего пакета из шести новых моделей, без деления на шесть. Это не многократно усреднённое время обучения одной модели. CPU: один поток, фактический ответ на один пример после прогрева, с обработкой всех 64 участков. Декодирование фотографии, выделение кожи и работа на телефоне не измерены.",
            "",
            f"Исследование: {summary['research_banks']} пакета / {summary['research_trajectories']} траектории, {summary['research_presentations']:,} предъявлений существующих примеров; {summary['research_bank_seconds']:.1f} секунды обучения банков. Пик выделенной CUDA-памяти: {summary['cuda_peak_allocated_bytes'] / 1e9:.2f} ГБ.",
            "",
            f"Независимо проверены 1134 внутренних экспорта / 214200 ответов, все решения, 63 конечные модели и 25158 фактических вызовов потребителя со всеми промежуточными проходами. Максимальное расхождение FP64 NumPy и FP32 CUDA: {audit['maximum_native_lab']:.6g} Lab при заранее заданном допуске0.002. Полные повторные обучения дали побитное совпадение 63 исходных моделей, 63 выбранных и126 банковых экспортов. Четыре теста и линтер прошли.",
            "",
            "Перед основным запуском сохранены пробы FP32 и BF16. BF16 не ускорил эту реализацию пакета, поэтому основная серия использует FP32. Это локальное наблюдение, не общий вывод о смешанной точности.",
            "",
            "Поиск ограничен 2048 шагами и двумя скоростями при неизменном горизонте8192. Нельзя объявлять найденный результат пределом качества большой архитектуры. Исходных измеренных наблюдений по-прежнему966 у24 людей; повторения не являются новыми данными. Качество обычных селфи, полного подбора косметики, патентоспособность и право на резидентство Сколково здесь не подтверждаются.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    card = ROOT / "docs/architecture/chromaseed_architecture_scale_model_card.md"
    card.write_text(
        "# ChromaSeed AS model card\n\nSeven patch/pooling/four-pass soft/dynamic residual regressors at17,374/15,246 and approximately4.85–4.96million deployed parameters. Includes a frozen643-parameter NP anchor; all residual weights are FP32. Training and CUDA inference FP32, NumPy consumer FP64 after FP32 input normalization. Inputs color36 and64x18 descriptors; native D65/10-degree Lab output. No identity or ethnicity inference.\n\nSee ../benchmarks/chromaseed_architecture_scale_v1/report.md for all selected results, negative outcomes, latency and full construction cost. Original TRAIN966observations/24people only, repeatedly reused roles and camera/person confounding; no independent phone-face validation. This architecture scale screen is progress, not completion of the broader quality goal.\n",
        encoding="utf-8",
    )
    decision = ROOT / "docs/research/chromaseed_architecture_scale_next_decision.md"
    boundary = sum(
        selection["roles"][role]["policies"][v]["step"] == 2048 for role in ROLES for v in VARIANTS
    )
    decision.write_text(
        f"# Next decision after AS\n\nUser quality priority and larger-model authorization persist. The seven-variant screen is verified progress, not broad-goal completion. Matched scale outcomes: {counts}. {boundary}/21 per-variant choices reach2048. Preserve the frozen overall policies and every adverse comparison.\n\nInspect the inner learning curves, model-family scaling comparisons and all four-pass errors before registering the next bounded study. Longer lower-rate training and richer local representations remain hypotheses, not already measured improvements; do not choose a next model solely from this reused outer table. The descriptors and tiny cohort limit what adding parameters can establish. Ordinary-phone facial acquisition remains the key external evidence gap.\n\nAS primary/audit/runtime/report/card/decision are sealed. Only report.py may reverify read-only. Preserve WE/WIDE and all earlier writers; P8 remains explicitly deferred. Original TRAIN only for this completed screen; no new assets, agents or publication.\n",
        encoding="utf-8",
    )
    post = {
        p: sha(ROOT / p)
        for p in (
            "scripts/chromaseed_architecture_scale_audit.py",
            "scripts/chromaseed_architecture_scale_runtime.py",
            "scripts/chromaseed_architecture_scale_report.py",
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
        parent_verification_sha256=sha(WE_OUT / "verification.json"),
        sources=lock["sources"],
        inputs=lock["input_sha256"],
        postprocess_sources=post,
        artifact_sha256=artifacts,
        classification="verified progress; broad quality goal remains active",
    )
    check_map({**value["sources"], **value["inputs"], **post, **artifacts})
    write_json(seal, value)
    print("AS SEALED", sha(seal), flush=True)
    print("AS QUALITY", overall, "scaling", counts, flush=True)


if __name__ == "__main__":
    main()
