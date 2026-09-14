# skin_local_reference_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Локальная аффинная регрессия на TRAIN leave-person-out улучшает ridge. Известный статистический механизм.

[Полная папка артефактов](../../../../docs/benchmarks/skin_local_reference_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_local_reference_v1/report.md)

SHA-256 отчёта: `83927b5423fe1ae1cff920e517b8c21f8b8d9d1a7586917425cf6709e464aea0`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_local_reference_provenance.md](../../../../docs/ip/skin_local_reference_provenance.md)
- [skin_local_reference_next_decision.md](../../../../docs/research/skin_local_reference_next_decision.md)
- [skin_local_reference_protocol_v1.md](../../../../docs/research/skin_local_reference_protocol_v1.md)
- [skin_local_reference_transfer_protocol_v1.md](../../../../docs/research/skin_local_reference_transfer_protocol_v1.md)
- [skin_local_reference_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_local_reference_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_local_reference_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_local_reference.py](../modules/scripts__skin_local_reference.md)
- [scripts/skin_local_reference_report.py](../modules/scripts__skin_local_reference_report.md)
- [scripts/skin_local_reference_run.py](../modules/scripts__skin_local_reference_run.md)
- [scripts/skin_local_reference_transfer.py](../modules/scripts__skin_local_reference_transfer.md)
- [scripts/skin_local_reference_transfer_verify.py](../modules/scripts__skin_local_reference_transfer_verify.md)
- [scripts/skin_local_reference_verify.py](../modules/scripts__skin_local_reference_verify.md)
- [tests/test_skin_local_reference.py](../tests/tests__test_skin_local_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_local_reference_v1/audit.json) | {"status": "PASS"} | `fc3fe7fdd3db785d33fd3583f991afe3a510d7dee386effde0c9aab3438900e2` |
| [results.json](../../../../docs/benchmarks/skin_local_reference_v1/results.json) | {"seconds": 1.0994490000011865} | `de93c5227498f7be240e9ac47c4cbc77c9ba8908518274aebc71755eaad16b54` |
| [source_lock.json](../../../../docs/benchmarks/skin_local_reference_v1/source_lock.json) | {} | `10b203fa6d753c31946fe8c032518f31207792de1c45da12e9d8445639b63025` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

У этой папки нет табличного итогового отчёта. Отсутствующие результаты не восстановлены из предположений.
