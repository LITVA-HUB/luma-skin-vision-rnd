# chromaseed_weight_average_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_weight_average_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_weight_average_v1/report.md)

SHA-256 отчёта: `5630d8c2c28e8cec0ad1c357eaf609652ca62e817b75945a0d8ced0f69680a53`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_weight_average_model_card.md](../../../../docs/architecture/chromaseed_weight_average_model_card.md)
- [chromaseed_weight_average_next_decision.md](../../../../docs/research/chromaseed_weight_average_next_decision.md)
- [chromaseed_weight_average_v1_protocol.md](../../../../docs/research/chromaseed_weight_average_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_weight_average_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_weight_average.py](../modules/scripts__chromaseed_weight_average.md)
- [scripts/chromaseed_weight_average_audit.py](../modules/scripts__chromaseed_weight_average_audit.md)
- [scripts/chromaseed_weight_average_report.py](../modules/scripts__chromaseed_weight_average_report.md)
- [scripts/chromaseed_weight_average_run.py](../modules/scripts__chromaseed_weight_average_run.md)
- [scripts/chromaseed_weight_average_runtime.py](../modules/scripts__chromaseed_weight_average_runtime.md)
- [tests/test_chromaseed_weight_average.py](../tests/tests__test_chromaseed_weight_average.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_weight_average_v1/audit.json) | {"passed": true, "seconds": 19.36301700002514} | `59621e5f652cd3401d92f282a7623ad283890f4b24d2f782b3ca9d641a2bd339` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_weight_average_v1/runtime.json) | {"passed": true, "seconds": 55.99361040000804} | `ba135db6b87a240635233d3e14986dd276e682bf4d8dadb4f0aee2248cf492dc` |
| [summary.json](../../../../docs/benchmarks/chromaseed_weight_average_v1/summary.json) | {} | `784934b162f0326d651dc72b7272a748ece511f89b68bdba9a9dbf533fd1b497` |
| [verification.json](../../../../docs/benchmarks/chromaseed_weight_average_v1/verification.json) | {"passed": true} | `f82f7f7d31b5639abbfef57ce96c79709022205b5d583291cf932cd823b1bdc0` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Вариант | Mixed | SLR → iPod | iPod → SLR |
| --- | --- | --- | --- |
| Исходная NP | 5.77054 | 8.29435 | 8.58953 |
| LT: внутренне выбранный одиночный этап | 5.71611 | 8.29435 | 8.68347 |
| Среднее соседних этапов | 5.69029 | 8.29603 | 8.63398 |
| Среднее начальной последовательности этапов | 5.69356 | 8.38140 | 8.64008 |
| Среднее трёх последних этапов | 5.61611 | 8.40851 | 8.88284 |
| WA: общий внутренний выбор | 5.71611 | 8.38140 | 8.64008 |
| Прежняя FG,20 284 числовых байта | 5.43865 | 8.59700 | 8.70502 |

### Таблица 2

| Сценарий | Политика | Этапы | Вариации / lr | Внутренняя ΔE00 | Внешняя ΔE00 |
| --- | --- | --- | --- | --- | --- |
| mixed | last | 2048 | 0 / 0.0001 | 4.19591 | 5.71611 |
| mixed | pair | 2048,8192 | 16 / 0.0001 | 4.19985 | 5.69029 |
| mixed | prefix | 512,2048,8192 | 0 / 0.0001 | 4.20123 | 5.69356 |
| mixed | tail3 | 2048,8192,32768 | 0 / 0.0001 | 4.20857 | 5.61611 |
| mixed | overall | 2048 | 0 / 0.0001 | 4.19591 | 5.71611 |
| slr_to_ipod | last | 0 | 0 / 0.0001 | 4.78737 | 8.29435 |
| slr_to_ipod | pair | 0,512 | 0 / 0.0003 | 4.78790 | 8.29603 |
| slr_to_ipod | prefix | 512,2048,8192,32768 | 256 / 0.0001 | 4.76558 | 8.38140 |
| slr_to_ipod | tail3 | 2048,8192,32768 | 256 / 0.0001 | 4.76969 | 8.40851 |
| slr_to_ipod | overall | 512,2048,8192,32768 | 256 / 0.0001 | 4.76558 | 8.38140 |
| ipod_to_slr | last | 2048 | 0 / 0.0001 | 4.44242 | 8.68347 |
| ipod_to_slr | pair | 2048,8192 | 0 / 0.0001 | 4.43953 | 8.63398 |
| ipod_to_slr | prefix | 512,2048,8192 | 0 / 0.0001 | 4.43834 | 8.64008 |
| ipod_to_slr | tail3 | 2048,8192,32768 | 0 / 0.0001 | 4.47551 | 8.88284 |
| ipod_to_slr | overall | 512,2048,8192 | 0 / 0.0001 | 4.43834 | 8.64008 |
