# `scripts/chromaseed_palette_transfer_report.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/chromaseed_palette_transfer_report.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Source-bound P3 report retaining all palette contrasts and complete cost scopes.

SHA-256 исходника: `42c82eff154d49e64ef632df979925da2115878a99c8195ed1681d63af94e743`. Строк: **406**.

## Зависимости

```python
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
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `comparisons` | FunctionDef | См. реализацию | [L44](../../../../scripts/chromaseed_palette_transfer_report.py#L44) |
| `summarize` | FunctionDef | См. реализацию | [L67](../../../../scripts/chromaseed_palette_transfer_report.py#L67) |
| `render` | FunctionDef | См. реализацию | [L206](../../../../scripts/chromaseed_palette_transfer_report.py#L206) |
| `run_checks` | FunctionDef | См. реализацию | [L294](../../../../scripts/chromaseed_palette_transfer_report.py#L294) |
| `main` | FunctionDef | См. реализацию | [L337](../../../../scripts/chromaseed_palette_transfer_report.py#L337) |

## Устройство, вычисление ответа и обучение

Ниже точные определения конструкторов, прямых проходов, формул ёмкости и fit/экспорта. Размерности задаются конструкторами и константами выше; наследуемые операции находятся в перечисленных импортируемых модулях. Повторяющиеся имена относятся к разным классам и различаются строкой исходника.

<details><summary>render · L206–291</summary>

```python
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
```

</details>
