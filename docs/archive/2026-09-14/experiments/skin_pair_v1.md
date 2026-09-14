# skin_pair_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Paired invariance, quotient, output consistency и VICReg; пара снимков не устранила зависимость от условий съёмки.

[Полная папка артефактов](../../../../docs/benchmarks/skin_pair_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_pair_v1/report.md)

SHA-256 отчёта: `4b385859c629523c4e78b782f9d11414cea5008b240b2f41eb6060580dc2e6bb`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_pair_invariance_protocol_v1.md](../../../../docs/research/skin_pair_invariance_protocol_v1.md)
- [skin_pair_next_decision.md](../../../../docs/research/skin_pair_next_decision.md)
- [skin_pair_research_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_pair_research_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_pair_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_pair_audit.py](../modules/scripts__skin_pair_audit.md)
- [scripts/skin_pair_invariance.py](../modules/scripts__skin_pair_invariance.md)
- [scripts/skin_pair_report.py](../modules/scripts__skin_pair_report.md)
- [scripts/skin_pair_train.py](../modules/scripts__skin_pair_train.md)
- [tests/test_skin_pair_invariance.py](../tests/tests__test_skin_pair_invariance.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_pair_v1/audit.json) | {"status": "PASS"} | `a247db4f8ca88da0cec67d3ffb0ac9afc4587cfaf0b626913710a2bff11485f6` |
| [source_lock.json](../../../../docs/benchmarks/skin_pair_v1/source_lock.json) | {} | `8b41c0e479cf61d4d66e52a2e57a15c5898832bb47f143453005bb6bb5f5d8ed` |
| [summary.json](../../../../docs/benchmarks/skin_pair_v1/summary.json) | {"scope": "Exploratory source validation and source camera-held-out fitting; not a new independent test"} | `71758c6b73b236c08eff0e46fb2875c29f8c79828451453e5b5956697db3e486` |
| [transfer_choice.json](../../../../docs/benchmarks/skin_pair_v1/transfer_choice.json) | {} | `f57a701e61e90191748db0b40ae476d754ecbd38445006fdb0945cbe7027f8a0` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Arm | Mean DeltaE00, seed17 /29 /43 | Mean over seeds | Same-site capture disagreement |
| --- | --- | --- | --- |
| raw | 3.4851 / 3.5127 / 3.4587 | 3.4855 | 2.7488 |
| standardized | 3.5806 / 3.6027 / 3.6441 | 3.6091 | 2.9142 |
| quotient3 | 4.0912 / 4.1960 / 4.0431 | 4.1101 | 3.0339 |
| output | 3.5188 / 3.4939 / 3.5242 | 3.5123 | 2.5056 |
| vicreg | 3.5507 / 3.5174 / 3.5668 | 3.5450 | 2.5573 |

### Таблица 2

| Training camera -> evaluation | Arm | TRAIN /selection /evaluation people | Evaluation images | Mean DeltaE00, seed17 /29 /43 | Mean over seeds |
| --- | --- | --- | --- | --- | --- |
| SLR -> iPod | output | 8 /3 /3 | 132 | 6.1419 / 6.3260 / 5.9898 | 6.1526 |
| SLR -> iPod | raw | 8 /3 /3 | 132 | 5.3482 / 5.4913 / 5.9794 | 5.6063 |
| iPod -> SLR | output | 16 /3 /3 | 132 | 6.3551 / 6.5163 / 6.8254 | 6.5656 |
| iPod -> SLR | raw | 16 /3 /3 | 132 | 5.8650 / 6.3605 / 6.1943 | 6.1399 |
