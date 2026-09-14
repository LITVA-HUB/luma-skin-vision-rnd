"""Source-bound HR report and final seal, only after full audit and exact replay."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from chromaseed_head_range_verification import (
    AS_OUT,
    CONTRACT,
    FILES,
    MODES,
    OUT,
    PARAMETERS,
    ROLES,
    ROOT,
    RUN,
    check_hashes,
    digest,
    index_records,
    operational_costs,
    read,
    remember,
    require_terminal,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_kernel_audit import nz
from chromaseed_refine_audit import error_summary
from chromaseed_widen_run import load_data
from threadpoolctl import threadpool_limits


def run_checks(contract):
    path = OUT / "checks.json"
    if path.exists():
        value = read(path)
        assert value["verification_protocol_sha256"] == digest(CONTRACT)
        assert value["sources"] == contract["sources"]
        assert value["passed"] and all(c["exit_code"] == 0 for c in value["checks"])
        return value
    commands = [
        [sys.executable, "-m", "pytest", "tests/test_chromaseed_head_range_verification.py", "-q"],
        [sys.executable, "-m", "ruff", "check", *[f for f in FILES if f.endswith(".py")]],
    ]
    checks = []
    for command in commands:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        checks.append(
            dict(command=command, exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        )
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
    value = dict(
        passed=True,
        verification_protocol_sha256=digest(CONTRACT),
        sources=contract["sources"],
        checks=checks,
    )
    write_once(path, value)
    return value


def summarize(data, selection, results, runtime, audit):
    records = index_records(results["records"])
    oldruntime = read(AS_OUT / "runtime.json")
    responses = index_records(runtime["responses"])
    rows = []
    variants = tuple(v + "__" + m for v in PARAMETERS for m in MODES) + ("np", "we")
    for variant in variants:
        row = dict(variant=variant, roles={})
        for role in ROLES:
            rr = [records[role, variant, seed] for seed in (17, 29, 43)]
            first = rr[0]
            item = dict(
                delta_e00=float(np.mean([r["metrics"]["person_mean"] for r in rr])),
                seeds=[r["metrics"]["person_mean"] for r in rr],
                step=first["step"],
                lr=first["lr"],
                fraction_gt5=float(np.mean([r["metrics"]["gt5"] for r in rr])),
                fraction_gt10=float(np.mean([r["metrics"]["gt10"] for r in rr])),
                n_images=first["metrics"]["n_images"],
                n_people=first["metrics"]["n_people"],
            )
            architecture = first["architecture"]
            if architecture in PARAMETERS:
                row["parameters"] = PARAMETERS[architecture]
                row["numeric_bytes"] = 4 * PARAMETERS[architecture] + 458
                mode = first["head_mode"]
                timings = [responses[role, variant, r["seed"]] for r in rr]
                recipes = oldruntime["recipes"] if mode == "unit" else runtime["recipes"]
                recipe_variant = architecture if mode == "unit" else variant
                recipe = next(
                    r for r in recipes if r["role"] == role and r["variant"] == recipe_variant
                )
                pass_errors = []
                for r in rr:
                    payload = r.get("inherited_as_record", r)
                    saved = nz(ROOT / payload["output"])
                    ix = saved["row_indices"]
                    pass_errors.append(
                        [
                            error_summary(
                                p, data["target"][ix], data["patient"][ix], data["site"][ix]
                            )[0]["person_mean"]
                            for p in saved["passes"].transpose(1, 0, 2)
                        ]
                    )
                item.update(
                    cpu_median_us=float(np.median([t["median_us"] for t in timings])),
                    complete_three_seed_bank_seconds=recipe["build_seconds"],
                    construction_provenance="sealed AS measurement"
                    if mode == "unit"
                    else "new HR measurement",
                    response_provenance="new HR measurement",
                    per_pass_delta_e00=np.mean(pass_errors, axis=0).tolist(),
                )
            else:
                item["timing_provenance"] = (
                    "inherited NP/WE control; no new response or construction measurement"
                )
            row["roles"][role] = item
        rows.append(row)
    indexed = {r["variant"]: r for r in rows}
    comparisons = []
    for variant in PARAMETERS:
        for role in ROLES:
            baseline = indexed[variant + "__unit"]["roles"][role]["delta_e00"]
            for mode in ("wide", "linear"):
                current = indexed[variant + "__" + mode]["roles"][role]["delta_e00"]
                comparisons.append(
                    dict(
                        architecture=variant,
                        role=role,
                        head_mode=mode,
                        head_minus_unit_delta_e00=current - baseline,
                    )
                )
    counts = dict(
        better=sum(r["head_minus_unit_delta_e00"] < -1e-9 for r in comparisons),
        worse=sum(r["head_minus_unit_delta_e00"] > 1e-9 for r in comparisons),
        same=sum(abs(r["head_minus_unit_delta_e00"]) <= 1e-9 for r in comparisons),
    )
    overall, per_architecture = {}, {}
    for role in ROLES:
        policies = selection["roles"][role]["policies"]
        c = policies["overall"]
        measured = indexed[c["variant"]]["roles"][role]
        assert sum(r["overall"] for r in results["records"] if r["role"] == role) == 3
        overall[role] = dict(
            variant=c["variant"],
            parent_variant=c.get("parent_variant"),
            step=c["step"],
            lr=c["lr"],
            inner_delta_e00=c["clean"],
            **{
                k: measured[k]
                for k in ("delta_e00", "fraction_gt5", "fraction_gt10", "n_images", "n_people")
            },
        )
        per_architecture[role] = {
            v: dict(
                variant=c["variant"],
                step=c["step"],
                lr=c["lr"],
                inner_delta_e00=c["clean"],
                delta_e00=indexed[c["variant"]]["roles"][role]["delta_e00"],
            )
            for v, c in policies["per_architecture"].items()
        }
    paths = sorted(p for kind in ("inner", "final") for p in (RUN / kind).rglob("receipt.json"))
    banks = [read(p) for p in paths]
    assert len(paths) == len(set(paths)) == 168
    assert len({(r["role"], r["variant"], r["head_mode"], r["fold"]) for r in banks}) == 168
    cost = operational_costs(
        banks, read(RUN / "recovery_v1/failed_progress.tmp"), read(RUN / "job.json")["seconds"]
    )
    return dict(
        rows=rows,
        head_comparisons=comparisons,
        head_comparison_counts=counts,
        overall=overall,
        per_architecture=per_architecture,
        costs=cost,
        audit_counts=audit["counts"],
        maximum_native_lab=audit["maximum_native_lab"],
        runtime_counts=runtime["counts"],
        audit_seconds=audit["seconds"],
        runtime_invocation_seconds=runtime["invocation_seconds"],
        runtime_reconstruction_seconds=sum(r["build_seconds"] for r in runtime["recipes"]),
        preflight_seconds=read(RUN / "preflight.json").get("seconds"),
        limitations="original 966 TRAIN observations/24 people; repeatedly reused camera/person roles; exploratory, no independent ordinary-phone accuracy",
    )


def render(summary):
    counts, cost = summary["head_comparison_counts"], summary["costs"]
    lines = [
        "# Luma ChromaSeed HR: диапазон поправки к оттенку кожи",
        "",
        "Завершено сравнение семи архитектур с тремя вариантами выходного слоя. Настройки выбирались внутри обучающей части и были зафиксированы до итоговых обучений и внешних прогнозов.",
        "",
        f"В 42 сопоставлениях нового выходного слоя с исходным: {counts['better']} улучшений, {counts['worse']} ухудшений, {counts['same']} совпадений. Это описательное сравнение уже использованных групп, без утверждения статистически подтверждённого превосходства.",
        "",
        "Все значения качества — ошибка ΔE00: меньше лучше. Усреднение сначала по людям, затем по трём отдельным запускам. Предсказания трёх запусков не объединяются в ансамбль.",
        "",
        "| Архитектура / выход | Параметры | Числовые данные, МБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        errors = " | ".join(f"{row['roles'][role]['delta_e00']:.4f}" for role in ROLES)
        if "parameters" in row:
            latency = np.median([row["roles"][role]["cpu_median_us"] for role in ROLES])
            lines.append(
                f"| {row['variant']} | {row['parameters']:,} | {row['numeric_bytes'] / 1e6:.3f} | {errors} | {latency:.1f} |"
            )
        else:
            lines.append(f"| {row['variant']} — прежний контроль | — | — | {errors} | — |")
    lines += [
        "",
        "unit = tanh(z); wide = 4·tanh(z/4); linear = z. Все три функции совпадают в нуле по значению и производной. Начальная модель NP одинакова; её 643 замороженных параметра входят в размер. В рекуррентных моделях выполняются четыре уточнения с коэффициентом 0,25.",
        "",
        "Исходные unit и NP/WE перенесены из проверенной серии AS с сохранением происхождения. 126 новых конечных моделей wide/linear обучены в HR. Все 189 времён ответа архитектур измерены заново; прежние времена построения unit приведены отдельно с указанием источника.",
        "",
        "## Зафиксированный выбор по внутренним данным",
        "",
        "| Роль | Выбранная модель | Шаги | Скорость обучения | Внутренняя ΔE00 | Внешняя ΔE00 | Ошибка >5, % | Ошибка >10, % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, c in summary["overall"].items():
        lines.append(
            f"| {role} | {c['variant']} | {c['step']} | {c['lr']} | {c['inner_delta_e00']:.4f} | {c['delta_e00']:.4f} | {100 * c['fraction_gt5']:.2f} | {100 * c['fraction_gt10']:.2f} |"
        )
    lines += [
        "",
        "Проценты обозначают долю ошибок цвета выше заданного порога, усреднённую по трём запускам. Они не являются долей правильно подобранных косметических товаров. Более удачная внешняя строка не заменяет внутренний выбор задним числом. Все 384 оценки, 87 решений и 207 итоговых записей сохранены.",
        "",
        "| Архитектура | Роль | Выбранный выход | Внутренняя ΔE00 | Внешняя ΔE00 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for role, choices in summary["per_architecture"].items():
        for architecture, c in choices.items():
            lines.append(
                f"| {architecture} | {role} | {c['variant']} | {c['inner_delta_e00']:.4f} | {c['delta_e00']:.4f} |"
            )
    lines += [
        "",
        "## Все промежуточные ответы и стоимость построения",
        "",
        "| Модель | Роль | Ошибка после каждого прохода | Полная сборка пакета, с | Источник времени |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in summary["rows"]:
        if "parameters" not in row:
            continue
        for role, r in row["roles"].items():
            passes = " → ".join(f"{e:.4f}" for e in r["per_pass_delta_e00"])
            source = (
                "AS, прежнее измерение"
                if row["variant"].endswith("__unit")
                else "HR, новое измерение"
            )
            lines.append(
                f"| {row['variant']} | {role} | {passes} | {r['complete_three_seed_bank_seconds']:.3f} | {source} |"
            )
    lines += [
        "",
        "Четыре прохода выполняются всегда. Промежуточная ошибка показана для анализа; по внешним данным остановка не выбиралась. Динамическая модель меняет используемые участки, но кодирование и расчёт оценок всех участков остаются плотными.",
        "",
        "Сборка — три полных исходных обучения, экспорт NP и весь пакет из шести продолжений. Время не делится на шесть и не выдаётся за время обучения одной модели. Однопоточный CPU замерен на реальном вызове для одного набора признаков после прогрева. Декодирование фото, выделение кожи и работа на телефоне сюда не входят.",
        "",
        "## Затраты и воспроизводимость",
        "",
        f"Основная серия: {cost['successful_banks']} завершённых пакетов, {cost['successful_trajectories']} траекторий, {cost['successful_presentations']:,} предъявлений. Сумма времени пакетов с подготовкой: {cost['successful_bank_seconds']:.1f} с; включая запись и сохранение прогнозов — {cost['successful_write_inclusive_seconds']:.1f} с. Эти величины перекрываются и не складываются.",
        "",
        f"До восстановления был потерян частичный пакет: {cost['discarded_partial_steps']} шагов, {cost['discarded_partial_seconds']:.3f} с и {cost['discarded_presentations']:,} предъявлений. Наблюдаемая сумма работы обучения с этой потерей: {cost['observed_fit_work_seconds']:.1f} с. Ранее завершённые 18 пакетов учтены один раз.",
        "",
        f"Таймер возобновлённой попытки: {cost['resumed_attempt_seconds']:.1f} с. Полное время от самого первого запуска отдельно не записано и остаётся неизвестным. Пик выделенной CUDA-памяти: {cost['peak_allocated_cuda_bytes'] / 1e9:.3f} ГБ.",
        "",
        f"Независимый потребитель проверил 2268 внутренних моделей и 428400 векторов, все выборы настроек, 252 конечных банковых экспорта и 126 выбранных файлов; 50316 фактических одиночных вызовов сопоставлены со всеми промежуточными проходами. Максимальное расхождение NumPy/CUDA: {summary['maximum_native_lab']:.6g} Lab при допуске 0,002 и rtol 1e-6.",
        "",
        "Полностью повторены 42 рецепта построения: 126 исходных обучений, 252 экспорта продолжений, 126 выбранных моделей и их прогнозы совпали побитно. Измерены 189 моделей, 36288 вызовов после 20 прогревов на модель. 81 прежняя итоговая запись и 126 внутренних кандидатов AS проверены как унаследованные, без заявления о новом обучении этих контролей.",
        "",
        f"Время аудита: {summary['audit_seconds']:.1f} с; полные повторные построения для проверки: {summary['runtime_reconstruction_seconds']:.1f} с. Эти затраты и предварительные пробы отделены от основной серии. Тесты проверки и линтер завершены успешно.",
        "",
        "Исходных измеренных наблюдений по-прежнему 966 у 24 людей. Внешние группы содержат 6, 16 и 8 людей и использовались в предыдущих исследованиях; камера связана с составом людей. Это исследовательский результат на прежних данных. Новые палитры P1/P2, LaPa и пять пользовательских фотографий в HR не подмешивались. Здесь не подтверждено качество обычных селфи и полного подбора косметики.",
        "",
    ]
    return "\n".join(lines)


def text_once(path, content):
    if path.exists():
        assert path.read_text(encoding="utf-8") == content, (
            f"Preserve existing report artifact: {path}"
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main():
    seal = OUT / "verification.json"
    if seal.exists():
        value = read(seal)
        assert value["passed"]
        check_hashes(
            {
                **value["sources"],
                **value["inputs"],
                **value["postprocess_sources"],
                **value["artifact_sha256"],
            }
        )
        print("HR READ-ONLY VERIFIED", digest(seal), flush=True)
        return
    require_terminal(read(RUN / "job.json"))
    contract = verify_contract()
    audit, runtime = verify_stage("audit.json"), verify_stage("runtime.json")
    assert audit["counts"] == contract["expected_counts"]
    assert runtime["counts"] == contract["runtime_counts"]
    assert runtime["audit_sha256"] == digest(OUT / "audit.json")
    assert runtime["inherited_unit_construction_sha256"] == digest(AS_OUT / "runtime.json")
    run_checks(contract)
    selection, results = read(RUN / "selections.json"), read(RUN / "results.json")
    with threadpool_limits(limits=1):
        summary = summarize(load_data(), selection, results, runtime, audit)
    write_once(OUT / "summary.json", summary)
    text_once(OUT / "report.md", render(summary))
    card = ROOT / "docs/architecture/chromaseed_head_range_model_card.md"
    text_once(
        card,
        "# Luma ChromaSeed HR\n\nSeven residual architecture variants, three output ranges (unit/wide/linear), 15,246–4,962,566 deployed parameters including frozen NP anchor. Four shared refinement passes for soft/dynamic families; dense scoring with dynamic masks is not a sparse-runtime claim. Input: color36 and 64×18 local descriptors; output: native D65/10° Lab. No identity matching.\n\nAll results, adverse comparisons, thresholds, costs and limits: ../benchmarks/chromaseed_head_range_v1/report.md. Original TRAIN 966 observations/24 people, historically reused roles; no fresh ordinary-phone validation. Wide/linear newly fitted; unit/NP/WE inherited from sealed AS with provenance. P1/P2 auxiliary training is separate.\n",
    )
    decision = ROOT / "docs/research/chromaseed_head_range_next_decision.md"
    boundary = sum(
        c["step"] == 2048
        for role in selection["roles"].values()
        for c in role["policies"]["per_pair"].values()
    )
    text_once(
        decision,
        "# Next decision after HR\n\nVerified progress; broad quality goal remains active. Preserve frozen choices and all adverse comparisons. "
        f"Descriptive output-range outcomes: {summary['head_comparison_counts']}; {boundary}/63 per-pair choices reach the 2048-step boundary. "
        "These repeatedly used outer groups must not be presented as independent model confirmation.\n\n"
        "The six P2 palette encoders are separately trained and layer-compatible; native improvement has not been tested here. Next register a matched original/aligned/shuffled initialization study, keeping the native architecture, roles, optimizer and budget equal and selecting on inner folds. Do not promote a model solely because of the reused outer table or auxiliary TRAIN MSE. New phone-photo color references remain a separate evidence gap. Preserve sealed sources and controls.\n",
    )
    lock = read(RUN / "source_lock.json")
    artifacts = {**audit["artifact_sha256"], **runtime["artifact_sha256"]}
    for name in ("audit.json", "runtime.json", "checks.json", "summary.json", "report.md"):
        remember(OUT / name, artifacts)
    for path in (CONTRACT, card, decision):
        remember(path, artifacts)
    value = dict(
        passed=True,
        source_lock_sha256=digest(RUN / "source_lock.json"),
        selection_sha256=digest(RUN / "selections.json"),
        results_sha256=digest(RUN / "results.json"),
        parent_verification_sha256=digest(AS_OUT / "verification.json"),
        sources=lock["sources"],
        inputs=lock["input_sha256"],
        postprocess_sources=contract["sources"],
        artifact_sha256=artifacts,
        classification="verified progress; broad quality goal remains active",
    )
    require_terminal(read(RUN / "job.json"))
    check_hashes(
        {**value["sources"], **value["inputs"], **value["postprocess_sources"], **artifacts}
    )
    write_once(seal, value)
    print("HR SEALED", digest(seal), flush=True)
    print(
        "HR QUALITY",
        summary["overall"],
        "head comparisons",
        summary["head_comparison_counts"],
        flush=True,
    )


if __name__ == "__main__":
    main()
