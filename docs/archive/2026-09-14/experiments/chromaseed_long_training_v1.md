# chromaseed_long_training_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_long_training_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_long_training_v1/report.md)

SHA-256 отчёта: `2f61341b522fc3475a25a5f8b3aaae01c035b46b41a4def09528ce2d6db1262d`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_long_training_model_card.md](../../../../docs/architecture/chromaseed_long_training_model_card.md)
- [chromaseed_long_training_next_decision.md](../../../../docs/research/chromaseed_long_training_next_decision.md)
- [chromaseed_long_training_v1_protocol.md](../../../../docs/research/chromaseed_long_training_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_long_training_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_long_training_audit.py](../modules/scripts__chromaseed_long_training_audit.md)
- [scripts/chromaseed_long_training_fit.py](../modules/scripts__chromaseed_long_training_fit.md)
- [scripts/chromaseed_long_training_report.py](../modules/scripts__chromaseed_long_training_report.md)
- [scripts/chromaseed_long_training_run.py](../modules/scripts__chromaseed_long_training_run.md)
- [scripts/chromaseed_long_training_runtime.py](../modules/scripts__chromaseed_long_training_runtime.md)
- [tests/test_chromaseed_long_training.py](../tests/tests__test_chromaseed_long_training.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_long_training_v1/audit.json) | {"passed": true, "seconds": 86.91073960001813} | `f6745e723dcc0a685bf2448bda3401d7a358b8dd7d5d736a6ae6fde186078943` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_long_training_v1/runtime.json) | {"passed": true, "seconds": 16.646489900012966} | `874dcd8a18a98b7b63f6009953dc3dea38d740a93b14974b636b8ee8fdff981f` |
| [summary.json](../../../../docs/benchmarks/chromaseed_long_training_v1/summary.json) | {} | `ebc0bd7ed7d7091643e73fdfdb9c29ce0e6321fe425d94617903833cf27f06bb` |
| [verification.json](../../../../docs/benchmarks/chromaseed_long_training_v1/verification.json) | {"passed": true} | `18b9127d82ca32e854be7efbf3906dd4f1fa73a749bf9d41ab7a78eae7df795b` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Режим | Mixed | SLR → iPod | iPod → SLR |
| --- | --- | --- | --- |
| NP: исходная маленькая модель | 5.77054 | 8.29435 | 8.58953 |
| LT: выбор по внутренним данным | 5.71611 | 8.29435 | 8.68347 |
| LT: 131 072 шага, без добавок, lr=0,0003 | 5.48847 | 8.92431 | 9.19793 |
| LT: 131 072 шага, 16 вариаций, lr=0,0003 | 5.46621 | 9.27539 | 9.28298 |
| LT: 131 072 шага, 256 вариаций, lr=0,0003 | 5.46195 | 9.12002 | 9.13320 |
| FG: прежний ориентир, 20 284 числовых байта | 5.43865 | 8.59700 | 8.70502 |

### Таблица 2

| Сценарий / число добавок | Шаг | Начальный lr | Внутренняя ΔE00 | Внешняя ΔE00 |
| --- | --- | --- | --- | --- |
| mixed / 0 | 2048 | 0.0001 | 4.19591 | 5.71611 |
| mixed / 16 | 2048 | 0.0001 | 4.19776 | 5.71819 |
| mixed / 256 | 2048 | 0.0003 | 4.19835 | 5.69360 |
| mixed / overall | 2048 | 0.0001 | 4.19591 | 5.71611 |
| slr_to_ipod / 0 | 0 | — | 4.78737 | 8.29435 |
| slr_to_ipod / 16 | 0 | — | 4.78737 | 8.29435 |
| slr_to_ipod / 256 | 0 | — | 4.78737 | 8.29435 |
| slr_to_ipod / overall | 0 | — | 4.78737 | 8.29435 |
| ipod_to_slr / 0 | 2048 | 0.0001 | 4.44242 | 8.68347 |
| ipod_to_slr / 16 | 2048 | 0.0001 | 4.45117 | 8.60649 |
| ipod_to_slr / 256 | 2048 | 0.0001 | 4.45093 | 8.62258 |
| ipod_to_slr / overall | 2048 | 0.0001 | 4.44242 | 8.68347 |
