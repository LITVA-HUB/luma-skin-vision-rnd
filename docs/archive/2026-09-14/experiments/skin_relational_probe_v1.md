# skin_relational_probe_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Сравнение участков и ложных пар; 1 421 пара не создаёт новых независимых людей или cross-camera пар.

[Полная папка артефактов](../../../../docs/benchmarks/skin_relational_probe_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_relational_probe_v1/report.md)

SHA-256 отчёта: `e910d355918ec0d6eabf7a2f2f38962db66c5f181bb3848b390d475c6e6bc69e`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_relational_probe_provenance.md](../../../../docs/ip/skin_relational_probe_provenance.md)
- [skin_relational_probe_next_decision.md](../../../../docs/research/skin_relational_probe_next_decision.md)
- [skin_relational_probe_protocol_v1.md](../../../../docs/research/skin_relational_probe_protocol_v1.md)
- [skin_relational_probe_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_relational_probe_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_relational_probe_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_relational_probe.py](../modules/scripts__skin_relational_probe.md)
- [scripts/skin_relational_probe_report.py](../modules/scripts__skin_relational_probe_report.md)
- [scripts/skin_relational_probe_run.py](../modules/scripts__skin_relational_probe_run.md)
- [scripts/skin_relational_probe_verify.py](../modules/scripts__skin_relational_probe_verify.md)
- [tests/test_skin_relational_probe.py](../tests/tests__test_skin_relational_probe.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_relational_probe_v1/audit.json) | {"status": "PASS"} | `ad53b51090e8dc70ebe7ddbaacd434c84db0889f6d0bdc0fb79f32291d693577` |
| [results.json](../../../../docs/benchmarks/skin_relational_probe_v1/results.json) | {} | `547b9b387f09cc9eed2df368c9d02d3b0c32cc01c1bfeab5f7391524f3fd20ed` |
| [source_lock.json](../../../../docs/benchmarks/skin_relational_probe_v1/source_lock.json) | {} | `3589302865671ce11785306048d033712294826cee5d104c90142efd659c4ac0` |
| [summary.json](../../../../docs/benchmarks/skin_relational_probe_v1/summary.json) | {} | `0295342f02e6fd0507239a8b008b66ef5c47fb40380bad83772e7ed85b146b03` |
| [test_receipt.json](../../../../docs/benchmarks/skin_relational_probe_v1/test_receipt.json) | {"passed": 367, "seconds": 33.63} | `1a4a10ba9c4fa8f1f904796d7c1d2d442e51314c2340fce270491cfcbda288a3` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Representation | Scope | Closest-control preference | Control within1 DeltaE00 | Control within2 DeltaE00 | True-color rank association |
| --- | --- | --- | --- | --- | --- |
| median_rgb | unfitted descriptor | 59.30% | 57.56% | 59.62% | 0.2278 |
| color36 | unfitted descriptor | 64.85% | 66.15% | 63.48% | 0.2516 |
| patch54 | unfitted descriptor | 63.78% | 68.29% | 64.12% | 0.2201 |
| context_image | TRAIN encoder, descriptive | 70.92% | 61.07% | 71.30% | 0.3686 |
| frozen_lab_image | TRAIN encoder, descriptive | 68.59% | 60.22% | 66.20% | 0.4962 |
| context_combined | TRAIN encoder, descriptive | 69.68% | 59.92% | 69.40% | 0.3576 |
| frozen_lab_combined | TRAIN encoder, descriptive | 70.18% | 63.27% | 69.21% | 0.4924 |
| ridge3 | LOPO linear/mean | 68.33% | 67.86% | 65.51% | 0.3427 |
| ridge36 | LOPO linear/mean | 70.08% | 72.64% | 70.09% | 0.3955 |
| mean_lab | LOPO linear/mean | 50.00% | 50.00% | 50.00% | undefined (constant) |

### Таблица 2

| LOPO control | Mean DeltaE00 | Median | p95 | Equal-person mean |
| --- | --- | --- | --- | --- |
| ridge3 | 6.0395 | 5.3623 | 11.9183 | 6.0176 |
| ridge36 | 5.4259 | 4.7431 | 11.4364 | 5.3388 |
| mean_lab | 10.9115 | 9.8601 | 22.6492 | 11.0235 |
