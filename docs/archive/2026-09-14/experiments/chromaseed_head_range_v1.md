# chromaseed_head_range_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Качество проверено; runtime оборван, итоговая печать отсутствует.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_head_range_v1)

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_head_range_host_recovery_v1_protocol.md](../../../../docs/research/chromaseed_head_range_host_recovery_v1_protocol.md)
- [chromaseed_head_range_launch_2026-09-14.md](../../../../docs/research/chromaseed_head_range_launch_2026-09-14.md)
- [chromaseed_head_range_readiness_2026-09-14.md](../../../../docs/research/chromaseed_head_range_readiness_2026-09-14.md)
- [chromaseed_head_range_recovery_status_2026-09-14.md](../../../../docs/research/chromaseed_head_range_recovery_status_2026-09-14.md)
- [chromaseed_head_range_recovery_v1.md](../../../../docs/research/chromaseed_head_range_recovery_v1.md)
- [chromaseed_head_range_v1_protocol.md](../../../../docs/research/chromaseed_head_range_v1_protocol.md)
- [chromaseed_head_range_verification_preparation_2026-09-14.md](../../../../docs/research/chromaseed_head_range_verification_preparation_2026-09-14.md)
- [chromaseed_head_range_verification_v1_protocol.md](../../../../docs/research/chromaseed_head_range_verification_v1_protocol.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_head_range.py](../modules/scripts__chromaseed_head_range.md)
- [scripts/chromaseed_head_range_audit.py](../modules/scripts__chromaseed_head_range_audit.md)
- [scripts/chromaseed_head_range_fit.py](../modules/scripts__chromaseed_head_range_fit.md)
- [scripts/chromaseed_head_range_host_recovery.py](../modules/scripts__chromaseed_head_range_host_recovery.md)
- [scripts/chromaseed_head_range_inner_check.py](../modules/scripts__chromaseed_head_range_inner_check.md)
- [scripts/chromaseed_head_range_preflight.py](../modules/scripts__chromaseed_head_range_preflight.md)
- [scripts/chromaseed_head_range_recovery.py](../modules/scripts__chromaseed_head_range_recovery.md)
- [scripts/chromaseed_head_range_recovery_check.py](../modules/scripts__chromaseed_head_range_recovery_check.md)
- [scripts/chromaseed_head_range_report.py](../modules/scripts__chromaseed_head_range_report.md)
- [scripts/chromaseed_head_range_run.py](../modules/scripts__chromaseed_head_range_run.md)
- [scripts/chromaseed_head_range_runtime.py](../modules/scripts__chromaseed_head_range_runtime.md)
- [scripts/chromaseed_head_range_verification.py](../modules/scripts__chromaseed_head_range_verification.md)
- [tests/test_chromaseed_head_range.py](../tests/tests__test_chromaseed_head_range.md)
- [tests/test_chromaseed_head_range_fit.py](../tests/tests__test_chromaseed_head_range_fit.md)
- [tests/test_chromaseed_head_range_host_recovery.py](../tests/tests__test_chromaseed_head_range_host_recovery.md)
- [tests/test_chromaseed_head_range_recovery.py](../tests/tests__test_chromaseed_head_range_recovery.md)
- [tests/test_chromaseed_head_range_run.py](../tests/tests__test_chromaseed_head_range_run.md)
- [tests/test_chromaseed_head_range_verification.py](../tests/tests__test_chromaseed_head_range_verification.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_head_range_v1/audit.json) | {"passed": true, "seconds": 890.8774619000033, "scope": "independent NumPy and metrics for new HR fits; exact remapping of sealed AS controls"} | `283dad925b9f8ed36b269418fc935854d0fb3b2d396e26e4f0b760e0fb9e90df` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

У этой папки нет табличного итогового отчёта. Отсутствующие результаты не восстановлены из предположений.
