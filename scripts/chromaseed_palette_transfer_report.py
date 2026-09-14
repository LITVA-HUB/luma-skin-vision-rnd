"""Source-bound P3 report retaining all palette contrasts and complete cost scopes."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from chromaseed_head_range_audit import bank_path as prior_bank
from chromaseed_head_range_report import text_once
from chromaseed_kernel_audit import nz
from chromaseed_palette_transfer_audit import bank_path
from chromaseed_palette_transfer_verification import (
    ARMS,
    CONTRACT,
    EXPECTED,
    FILES,
    HR_OUT,
    OUT,
    P2,
    PARAMETERS,
    ROLES,
    ROOT,
    RUN,
    RUNTIME_COUNTS,
    SEEDS,
    check_hashes,
    digest,
    index_records,
    native_costs,
    payload,
    primary_gate,
    read,
    remember,
    verify_contract,
    verify_stage,
    write_once,
)
from chromaseed_refine_audit import error_summary
from chromaseed_widen_run import load_data
from threadpoolctl import threadpool_limits


def comparisons(rows):
    indexed = {r["variant"]: r for r in rows}
    result = []
    for variant in PARAMETERS:
        for role in ROLES:
            errors = {
                arm: indexed[variant + "__" + arm]["roles"][role]["delta_e00"] for arm in ARMS
            }
            assert all(np.isfinite(e) and e >= 0 for e in errors.values())
            original = errors["aligned"] - errors["original"]
            shuffled = errors["aligned"] - errors["shuffled"]
            result.append(
                dict(
                    architecture=variant,
                    role=role,
                    aligned_minus_original=original,
                    aligned_minus_shuffled=shuffled,
                    better_than_both=original < -1e-9 and shuffled < -1e-9,
                )
            )
    return result


def summarize(data, selection, results, runtime, audit):
    records, responses = index_records(results["records"]), index_records(runtime["responses"])
    variants = tuple(v + "__" + a for v in PARAMETERS for a in ARMS) + ("np", "we")
    rows = []
    for variant in variants:
        row = dict(variant=variant, roles={})
        for role in ROLES:
            rr = [records[role, variant, seed] for seed in SEEDS]
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
                timings = [responses[role, variant, r["seed"]] for r in rr]
                recipe = next(
                    r for r in runtime["recipes"] if r["role"] == role and r["variant"] == variant
                )
                pass_errors = []
                for r in rr:
                    source = payload(r)
                    saved = nz(ROOT / source["output"])
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
                    head_mode=first["head_mode"],
                    initialization=first["initialization"],
                    cpu_median_us=float(np.median([r["median_us"] for r in timings])),
                    cpu_p95_us=float(np.median([r["p95_us"] for r in timings])),
                    complete_three_seed_bank_seconds=recipe["build_seconds"],
                    per_pass_delta_e00=np.mean(pass_errors, axis=0).tolist(),
                    timing_provenance="new P3 response and complete native-bank measurements",
                )
            else:
                item["timing_provenance"] = "inherited NP/WE control; not remeasured"
            row["roles"][role] = item
        rows.append(row)
    indexed = {r["variant"]: r for r in rows}
    overall, per_architecture = {}, {}
    for role in ROLES:
        policies = selection["roles"][role]["policies"]

        def decision(choice):
            measured = indexed[choice["variant"]]["roles"][role]
            return dict(
                variant=choice["variant"],
                step=choice["step"],
                lr=choice["lr"],
                inner_delta_e00=choice["clean"],
                **{
                    k: measured[k]
                    for k in ("delta_e00", "fraction_gt5", "fraction_gt10", "n_images", "n_people")
                },
            )

        overall[role] = decision(policies["overall"])
        assert sum(r["overall"] for r in results["records"] if r["role"] == role) == 3
        per_architecture[role] = {v: decision(c) for v, c in policies["per_architecture"].items()}
    banks = [
        read(bank_path(role, v, a, fold) / "receipt.json")
        for role in ROLES
        for v in PARAMETERS
        for a in ARMS[1:]
        for fold in (0, 1, 2, None)
    ]
    assert (
        len(banks) == 72
        and len({(b["role"], b["variant"], b["initialization"], b["fold"]) for b in banks}) == 72
    )
    cost = native_costs(banks, read(RUN / "job.json")["seconds"])
    heads = read(RUN / "source_lock.json")["heads"]
    reused = [
        read(prior_bank(role, v, heads[role][v], fold) / "receipt.json")
        for role in ROLES
        for v in PARAMETERS
        for fold in (0, 1, 2, None)
    ]
    cost["reused_original_banks"] = len(reused)
    cost["reused_original_fit_seconds"] = sum(r["full_bank_seconds"] for r in reused)
    cost["reused_scope"] = (
        "36 borrowed banks, already computed in AS/HR; excluded from new P3 native runtime"
    )
    cost["failed_primary_attempts"] = 0  # verify_contract rejects an unaccounted recovery.
    p2fit, p2data = read(P2 / "fit.json"), read(P2 / "data_profile.json")
    auxiliary = dict(
        spatial_patches=2432,
        simulated_views=19456,
        source_groups=19,
        encoders=6,
        data_preparation_seconds=p2data["seconds"],
        fitting_seconds=p2fit["fitting_seconds"],
        fitting_workflow_seconds=p2fit["complete_workflow_seconds"],
        scope="Shared P2 data preparation and six-model auxiliary fit; fit is inside workflow, not additive; earlier acquisition excluded",
    )
    contrasts = comparisons(rows)
    return dict(
        rows=rows,
        palette_comparisons=contrasts,
        contrast_counts=dict(
            better_than_both=sum(r["better_than_both"] for r in contrasts),
            better_than_original=sum(r["aligned_minus_original"] < -1e-9 for r in contrasts),
            worse_than_original=sum(r["aligned_minus_original"] > 1e-9 for r in contrasts),
        ),
        overall=overall,
        per_architecture=per_architecture,
        costs=cost,
        auxiliary_costs=auxiliary,
        audit_counts=audit["counts"],
        runtime_counts=runtime["counts"],
        maximum_native_lab=audit["maximum_native_lab"],
        audit_seconds=audit["seconds"],
        runtime_invocation_seconds=runtime["invocation_seconds"],
        runtime_reconstruction_seconds=sum(r["build_seconds"] for r in runtime["recipes"]),
        runtime_validation_inclusive_seconds=sum(
            r["validation_inclusive_seconds"] for r in runtime["recipes"]
        ),
        preflight_seconds={
            d: read(RUN / f"preflight_{d}.json")["seconds"] for d in ("cpu", "cuda")
        },
        limitations="Original TRAIN 966 observations/24 people; historically reused camera/person roles, not fresh confirmation. No ordinary-phone accuracy claim.",
    )


def render(summary):
    counts, cost, aux = summary["contrast_counts"], summary["costs"], summary["auxiliary_costs"]
    lines = [
        "# Luma ChromaSeed P3: перенос палитры на оценку цвета кожи",
        "",
        "Сравниваются три архитектуры и три начальные основы: обычная, обученная на палитре и обученная на перемешанных соответствиях палитры. Выходной слой для каждой архитектуры заранее определён внутренним выбором HR; настройки P3 выбраны по внутренним группам до итоговых обучений.",
        "",
        f"Из девяти описательных сопоставлений палитра лучше обоих контролей в {counts['better_than_both']}, лучше обычного старта в {counts['better_than_original']}, хуже обычного старта в {counts['worse_than_original']}. Все исходы сохранены; это не утверждение статистически подтверждённого превосходства.",
        "",
        "ΔE00 — ошибка цвета: меньше лучше. Усреднение сначала по людям, затем по трём отдельным seed; предсказания seed не объединяются в ансамбль.",
        "",
        "| Архитектура / начало | Параметры | Числовые данные, МБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |",
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
        "## Все сопоставления палитры",
        "",
        "Отрицательная разница означает меньшую ошибку у aligned. Этот раздел не меняет выбор модели.",
        "",
        "| Архитектура | Роль | aligned − original, ΔE00 | aligned − shuffled, ΔE00 |",
        "| --- | --- | ---: | ---: |",
    ]
    for r in summary["palette_comparisons"]:
        lines.append(
            f"| {r['architecture']} | {r['role']} | {r['aligned_minus_original']:.4f} | {r['aligned_minus_shuffled']:.4f} |"
        )
    lines += [
        "",
        "## Выбор, зафиксированный до итоговой оценки",
        "",
        "| Роль | Модель | Шаги | Learning rate | Внутренняя ΔE00 | Итоговая ΔE00 | Ошибка >5, % | Ошибка >10, % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for role, r in summary["overall"].items():
        lines.append(
            f"| {role} | {r['variant']} | {r['step']} | {r['lr']} | {r['inner_delta_e00']:.4f} | {r['delta_e00']:.4f} | {100 * r['fraction_gt5']:.2f} | {100 * r['fraction_gt10']:.2f} |"
        )
    lines += [
        "",
        "## Скорость и полная стоимость",
        "",
        "| Модель | Роль | Выход | CPU median, мкс | CPU p95, мкс | Полный пакет, с | Ошибка по проходам ΔE00 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        if "parameters" not in row:
            continue
        for role, r in row["roles"].items():
            passes = ", ".join(f"{v:.4f}" for v in r["per_pass_delta_e00"])
            lines.append(
                f"| {row['variant']} | {role} | {r['head_mode']} | {r['cpu_median_us']:.1f} | {r['cpu_p95_us']:.1f} | {r['complete_three_seed_bank_seconds']:.3f} | {passes} |"
            )
    lines += [
        "",
        f"Новые нативные обучения: {cost['native_banks']} пакета / {cost['native_trajectories']} траектории; {cost['native_bank_seconds']:.3f} с обучения, включая {cost['native_setup_seconds']:.3f} с подготовки. Время с записью и прогнозами — {cost['native_write_inclusive_seconds']:.3f} с; это пересекающийся объём работы, его нельзя прибавлять к времени обучения. Весь основной запуск после подготовки входных данных — {cost['primary_runtime_seconds']:.3f} с.",
        "",
        f"Повторно использованы {cost['reused_original_banks']} контрольных пакетов AS/HR, ранее потребовавшие {cost['reused_original_fit_seconds']:.3f} с обучения. Они не учитываются как новые обучения P3. Неучтённые аварийные восстановления запрещают итоговое подтверждение.",
        "",
        f"Отдельная подготовка P2: {aux['data_preparation_seconds']:.3f} с на данные; {aux['fitting_seconds']:.3f} с обучения шести цветовых основ внутри {aux['fitting_workflow_seconds']:.3f} с полного запуска предобучения. Общая стоимость сбора исходных данных этим таймером не измерена.",
        "",
        f"Проверка P3: {summary['audit_seconds']:.3f} с аудита. Повторное полное построение 27 пакетов — {summary['runtime_reconstruction_seconds']:.3f} с; вместе с проверкой побитового совпадения — {summary['runtime_validation_inclusive_seconds']:.3f} с. Это дополнительная проверочная работа, отдельно от основного обучения.",
        "",
        "Время ответа измерено заново для 81 модели: 20 прогревов и 3×64 одиночных вызова на одном потоке CPU. Вход уже содержит color36 и 64×18 локальных признаков; это не время обработки фотографии и не измерение телефона. Каждый полный пакет заново строит три родительские NP-модели и все шесть продолжений. Время пакета не делится на шесть.",
        "",
        "## Проверка и ограничения",
        "",
        f"Проверено: {summary['audit_counts']}. Максимальное расхождение независимого NumPy и сохранённого ответа: {summary['maximum_native_lab']:.9g} в координатах Lab (допуск 0,002 / 1e-6). Проверяются все промежуточные уточнения и 21 564 реальных одиночных вызова.",
        "",
        "19 спектральных источников дали 2 432 участка и 19 456 имитированных цветовых условий; это не новые люди и не дополнительные измеренные фотографии. В нативном обучении остаются исходные 966 наблюдений / 24 человека из TRAIN. Исторически использованные группы не подтверждают точность на новых пользовательских снимках. Пять личных фото без измеренного эталона не используются как цветовая разметка.",
        "",
        "Передаются только 105 856 начальных весов, затем все веса остаточной сети обучаются. Палитровый вспомогательный выход не входит в готовую модель. Родительская NP-модель из 643 весов входит в размер. Soft/dynamic выполняют четыре общих уточнения; динамическая маска связей не означает разреженные вычисления. Исследование ограничено выбранным HR выходом и 2 048 шагами: оно не исчерпывает взаимодействия архитектуры, предобучения и большей длины обучения.",
        "",
        "Полный исходный код, происхождение весов, строки данных, выбор настроек и все результаты связаны контрольными суммами. Общая цель качества остаётся активной.",
        "",
    ]
    return "\n".join(lines)


def run_checks(contract):
    path = OUT / "checks.json"
    if path.exists():
        value = read(path)
        assert value["passed"] and value["sources"] == contract["sources"]
        assert value["verification_protocol_sha256"] == digest(CONTRACT)
        return value
    commands = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_chromaseed_palette_transfer_verification.py",
            "-q",
        ],
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


def main():
    seal_path = OUT / "verification.json"
    if seal_path.exists():
        value = read(seal_path)
        assert value["passed"] and value["verification_protocol_sha256"] == digest(CONTRACT)
        check_hashes(
            {**value["sources"], **value["postprocess_sources"], **value["artifact_sha256"]}
        )
        print("P3 READ-ONLY VERIFIED", digest(seal_path), flush=True)
        return
    primary_gate()
    contract = verify_contract()
    audit, runtime = verify_stage("audit.json"), verify_stage("runtime.json")
    assert audit["counts"] == EXPECTED and runtime["counts"] == RUNTIME_COUNTS
    assert runtime["audit_sha256"] == digest(OUT / "audit.json")
    assert runtime["inherited_control_timing_sha256"] == digest(HR_OUT / "runtime.json")
    run_checks(contract)
    selection, results = read(RUN / "selections.json"), read(RUN / "results.json")
    with threadpool_limits(limits=1):
        summary = summarize(load_data(), selection, results, runtime, audit)
    write_once(OUT / "summary.json", summary)
    text_once(OUT / "report.md", render(summary))
    card = ROOT / "docs/architecture/chromaseed_palette_transfer_model_card.md"
    text_once(
        card,
        "# Luma ChromaSeed P3\n\nThree architectures, 4,846,822–4,962,566 deployed parameters; original/aligned/shuffled initializers, 105,856 transferred encoder weights. All residual weights fine-tuned, 643-parameter NP anchor frozen and counted. Native D65/10° Lab from prepared color36 and 64×18 descriptors. Four shared passes in soft/dynamic; dense computation with dynamic masks. No identity matching.\n\nAll native outcomes, adverse comparisons, per-pass metrics, complete cost scopes and limits: ../benchmarks/chromaseed_palette_transfer_v1/report.md. Reused TRAIN camera/person groups are exploratory, not fresh ordinary-phone validation.\n",
    )
    decision = ROOT / "docs/research/chromaseed_palette_transfer_next_decision.md"
    boundary = sum(
        c["step"] == 2048
        for r in selection["roles"].values()
        for c in r["policies"]["per_pair"].values()
    )
    text_once(
        decision,
        "# Next decision after P3\n\nVerified study progress; broad quality goal remains active. "
        f"Palette contrasts: {summary['contrast_counts']}. {boundary}/27 per-pair choices reach the 2048-step boundary. "
        "Preserve all failures and controls; choose the next registered intervention from inner evidence and resource costs, never promote an external-table winner. Longer training, changed capacity or initializer/head interactions remain possible subsequent experiments. Fresh measured phone-image references remain an independent evidence gap.\n",
    )
    artifacts = {
        **audit["artifact_sha256"],
        **runtime["artifact_sha256"],
        **contract["auxiliary_cost_bindings"],
        **contract["native_input_bindings"],
    }
    for name in ("audit.json", "runtime.json", "checks.json", "summary.json", "report.md"):
        remember(OUT / name, artifacts)
    for path in (CONTRACT, card, decision):
        remember(path, artifacts)
    value = dict(
        passed=True,
        verification_protocol_sha256=digest(CONTRACT),
        source_lock_sha256=digest(RUN / "source_lock.json"),
        selection_sha256=digest(RUN / "selections.json"),
        results_sha256=digest(RUN / "results.json"),
        parent_verification_sha256=digest(HR_OUT / "verification.json"),
        sources=read(RUN / "source_lock.json")["bindings"],
        postprocess_sources=contract["sources"],
        artifact_sha256=artifacts,
        classification="verified study progress; broad quality goal remains active",
    )
    primary_gate()
    check_hashes({**value["sources"], **value["postprocess_sources"], **artifacts})
    write_once(seal_path, value)
    print("P3 SEALED", digest(seal_path), flush=True)
    print("P3 QUALITY", summary["overall"], summary["contrast_counts"], flush=True)


if __name__ == "__main__":
    main()
