# skin_crossfit_correction_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Внутренний выигрыш 7,40% при +188 035 параметрах. Гипотеза превосходства OOF не подтвердилась.

[Полная папка артефактов](../../../../docs/benchmarks/skin_crossfit_correction_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_crossfit_correction_v1/report.md)

SHA-256 отчёта: `8077aed501186dd49e9b96906cd5b1f68d5f0d6a6dd9122940ea53c7fa80905e`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_crossfit_correction_provenance.md](../../../../docs/ip/skin_crossfit_correction_provenance.md)
- [skin_crossfit_correction_next_decision.md](../../../../docs/research/skin_crossfit_correction_next_decision.md)
- [skin_crossfit_correction_protocol_v1.md](../../../../docs/research/skin_crossfit_correction_protocol_v1.md)
- [skin_crossfit_correction_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_crossfit_correction_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_crossfit_correction_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_crossfit_correction.py](../modules/scripts__skin_crossfit_correction.md)
- [scripts/skin_crossfit_correction_report.py](../modules/scripts__skin_crossfit_correction_report.md)
- [scripts/skin_crossfit_correction_train.py](../modules/scripts__skin_crossfit_correction_train.md)
- [scripts/skin_crossfit_correction_verify.py](../modules/scripts__skin_crossfit_correction_verify.md)
- [tests/test_skin_crossfit_correction.py](../tests/tests__test_skin_crossfit_correction.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_crossfit_correction_v1/audit.json) | {"status": "PASS"} | `3a699d89d86e325209bd689fcb6f619efdab5af8249dced27f446a5d3cb84b21` |
| [results.json](../../../../docs/benchmarks/skin_crossfit_correction_v1/results.json) | {"status": "COMPLETE"} | `427d13e0ea442257025a0db670b0fdb5ef1357de43ee3f6941a2723d9ce5a5b5` |
| [source_lock.json](../../../../docs/benchmarks/skin_crossfit_correction_v1/source_lock.json) | {} | `b457c2a5179f6d7875af68c2dcde32aa39bf71ce5ab1192988cc665e6b816798` |
| [summary.json](../../../../docs/benchmarks/skin_crossfit_correction_v1/summary.json) | {} | `469cdb2c1223fed791a8eecd60f6f5047757ab4cbfe7e1a91e5b4f70f7b1d211` |
| [test_receipt.json](../../../../docs/benchmarks/skin_crossfit_correction_v1/test_receipt.json) | {"passed": 376, "seconds": 34.22} | `974d9ca9c30689ee9e5f6f7039c4719d61b212899c2976c276c2f2cb9a9f4608` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Correction training | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |
| --- | --- | --- | --- | --- | --- |
| base | 6.1082 | 5.1353 | 14.9960 | 15.37% | 5.6609 |
| in_full | 5.8579 | 4.8002 | 13.6359 | 13.79% | 5.4313 |
| in_matched | 5.6560 | 4.6887 | 12.9644 | 12.50% | 5.2608 |
| out_person | 5.8945 | 4.8164 | 13.4503 | 14.94% | 5.5260 |

### Таблица 2

| Arm | People improved vs core, after seed averaging | Mean person error change |
| --- | --- | --- |
| in_full | 4/6 | -0.2548 |
| in_matched | 5/6 | -0.4607 |
| out_person | 5/6 | -0.2287 |

### Таблица 3

| Prediction table on support people | Mean base error DeltaE00 |
| --- | --- |
| in_full | 3.5823 |
| in_matched | 3.3675 |
| out_person | 4.3241 |
