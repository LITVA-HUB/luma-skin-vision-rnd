# skin_correspondence_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Повторяемость реальных приборных измерений и многовидовые диагностики; это не известный предел ошибки модели.

[Полная папка артефактов](../../../../docs/benchmarks/skin_correspondence_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_correspondence_v1/report.md)

SHA-256 отчёта: `e09072535729fd0add82402e5084b888a0dc6c527899a52897afc41a51242f5c`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_correspondence_next_decision.md](../../../../docs/research/skin_correspondence_next_decision.md)
- [skin_correspondence_protocol_v1.md](../../../../docs/research/skin_correspondence_protocol_v1.md)
- [skin_correspondence_sources_2026_09_11.md](../../../../docs/research/skin_correspondence_sources_2026_09_11.md)
- [skin_correspondence_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_correspondence_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_correspondence_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_correspondence.py](../modules/scripts__skin_correspondence.md)
- [tests/test_skin_correspondence.py](../tests/tests__test_skin_correspondence.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [source_lock.json](../../../../docs/benchmarks/skin_correspondence_v1/source_lock.json) | {} | `f0b8021acba405c7afb6d87a7e2bee5a7808579bfad0d08c9a7d0696c55275cc` |
| [summary.json](../../../../docs/benchmarks/skin_correspondence_v1/summary.json) | {"scope": "SOURCE DIAGNOSTIC; no model fitting; no independent test"} | `3f337d87dcfea54ce204fd16f260a17336acd4ef8aefc1d8e772570673545e2f` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Role | Pairwise reading mean | Reading-to-mean | Leave-one-reading-out |
| --- | --- | --- | --- |
| train | 2.5757 | 1.5195 | 2.2809 |
| validation | 2.6139 | 1.5354 | 2.3043 |

### Таблица 2

| Protocol | Model | Single image mean | Diagnostic multi-view mean | Shared squared-Lab fraction | Error >5 against all 3 readings |
| --- | --- | --- | --- | --- | --- |
| mixed | plain_mse | 3.4771 | 3.0120 | 75.26% | 10.98% |
| mixed | mixture_mse | 3.4406 | 2.8719 | 71.65% | 11.11% |
| mixed | graph_always | 3.6394 | 3.1481 | 74.79% | 14.14% |
| from_SLR | plain_mse | 5.8301 | 5.4606 | 81.31% | 43.43% |
| from_SLR | mixture_mse | 5.0193 | 4.6598 | 81.59% | 36.11% |
| from_SLR | graph_always | 5.8339 | 5.6420 | 88.29% | 47.98% |
| from_ipod | plain_mse | 5.5089 | 4.0398 | 61.05% | 42.42% |
| from_ipod | mixture_mse | 5.9635 | 4.1870 | 58.64% | 48.48% |
| from_ipod | graph_always | 4.9736 | 3.9778 | 63.33% | 34.60% |
