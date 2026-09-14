# skin_local_teacher_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Локальное соответствие teacher/RGB: aligned, global и shuffled контроли, отрицательный перенос.

[Полная папка артефактов](../../../../docs/benchmarks/skin_local_teacher_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_local_teacher_v1/report.md)

SHA-256 отчёта: `3897b7769d2e8a8025ba011165bb7fe8253bda425ed0fba4b604ab760c1c2119`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_local_teacher_next_decision.md](../../../../docs/research/skin_local_teacher_next_decision.md)
- [skin_local_teacher_protocol_v1.md](../../../../docs/research/skin_local_teacher_protocol_v1.md)
- [skin_local_teacher_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_local_teacher_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_local_teacher_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_local_teacher_audit.py](../modules/scripts__skin_local_teacher_audit.md)
- [scripts/skin_local_teacher_features.py](../modules/scripts__skin_local_teacher_features.md)
- [scripts/skin_local_teacher_model.py](../modules/scripts__skin_local_teacher_model.md)
- [scripts/skin_local_teacher_report.py](../modules/scripts__skin_local_teacher_report.md)
- [scripts/skin_local_teacher_train.py](../modules/scripts__skin_local_teacher_train.md)
- [tests/test_skin_local_teacher.py](../tests/tests__test_skin_local_teacher.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_local_teacher_v1/audit.json) | {"status": "PASS"} | `55d9eec955073c35689c8fac9fb2756958891d93a23b365ad1709e554a9b130f` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_local_teacher_v1/audit_mixed.json) | {"status": "PASS"} | `f84fb5cb41750ead31da02187188172e4ba781f8a4fa419217381b71388a7d27` |
| [feature_audit.json](../../../../docs/benchmarks/skin_local_teacher_v1/feature_audit.json) | {"status": "PASS"} | `e28b5e3b2bdfda58b63b37ff741cdc1c78e95d45938b13bdc5c69023aab904a6` |
| [feature_receipt.json](../../../../docs/benchmarks/skin_local_teacher_v1/feature_receipt.json) | {"seconds": 3.7869146000011824} | `cb7cb866168d7ff3b213da67cb1498048aeb3365d347acb628e5f3e809fc6ae4` |
| [pooling_precision_diagnosis.json](../../../../docs/benchmarks/skin_local_teacher_v1/pooling_precision_diagnosis.json) | {} | `03616773b58004029d562a4b4347e3fe33c74769b03467afb5482a953840bc02` |
| [source_lock.json](../../../../docs/benchmarks/skin_local_teacher_v1/source_lock.json) | {} | `1523a80d6ea1ffc7321cd42878a08e9996142c99a5cc379465f5583607dc6787` |
| [summary.json](../../../../docs/benchmarks/skin_local_teacher_v1/summary.json) | {"scope": "SOURCE ONLY; three seed scores averaged, not an ensemble"} | `42b072df167783e8460d2dc35510ab1e958fc0b2a80ec969cc83f2a596a2d156` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Correspondence | Mean DeltaE00 | p95 | Three seed means |
| --- | --- | --- | --- | --- |
| mixed | plain | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | aligned | 4.1720 | 8.2774 | 4.2477, 4.2023, 4.0660 |
| mixed | global | 4.0992 | 8.0044 | 4.1424, 4.0603, 4.0948 |
| mixed | shuffled | 4.1826 | 8.1786 | 4.2621, 4.2069, 4.0787 |
| mixed | historical_plain_mse | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | historical_mixture_mse | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| from_SLR | plain | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | aligned | 6.0116 | 13.2466 | 6.1183, 5.8730, 6.0436 |
| from_SLR | global | 5.9560 | 12.7014 | 5.8623, 6.1725, 5.8331 |
| from_SLR | shuffled | 5.9191 | 12.8089 | 5.8949, 5.8426, 6.0199 |
| from_SLR | historical_plain_mse | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | historical_mixture_mse | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_ipod | plain | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | aligned | 7.8197 | 13.3947 | 8.2828, 7.8802, 7.2963 |
| from_ipod | global | 7.7831 | 13.6317 | 8.4186, 7.0437, 7.8870 |
| from_ipod | shuffled | 7.9399 | 13.7294 | 8.6006, 7.9304, 7.2886 |
| from_ipod | historical_plain_mse | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | historical_mixture_mse | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |

### Таблица 2

| Protocol | Correspondence | Mean at 80% | Active head + required teacher parameters | Peak fit+selection MiB |
| --- | --- | --- | --- | --- |
| mixed | plain | 3.4407 | 929297 + 0 | 108.68 |
| mixed | aligned | 4.1590 | 1011601 + 22056576 | 232.62 |
| mixed | global | 4.0446 | 1011601 + 22056576 | 232.62 |
| mixed | shuffled | 4.2122 | 1011601 + 22056576 | 232.62 |
| from_SLR | plain | 5.1087 | 929297 + 0 | 105.23 |
| from_SLR | aligned | 6.0805 | 1011601 + 22056576 | 161.50 |
| from_SLR | global | 6.2725 | 1011601 + 22056576 | 161.50 |
| from_SLR | shuffled | 5.8207 | 1011601 + 22056576 | 161.50 |
| from_ipod | plain | 5.7914 | 929297 + 0 | 106.65 |
| from_ipod | aligned | 7.9010 | 1011601 + 22056576 | 193.51 |
| from_ipod | global | 7.8224 | 1011601 + 22056576 | 193.51 |
| from_ipod | shuffled | 7.9990 | 1011601 + 22056576 | 193.51 |

### Таблица 3

| Protocol | Correspondence | Reference | Seed wins | Person mean difference | 95% descriptive interval |
| --- | --- | --- | --- | --- | --- |
| mixed | plain | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| mixed | aligned | mixture_mse | 0/3 | 0.7314 | [0.2394, 1.1096] |
| mixed | global | mixture_mse | 0/3 | 0.6586 | [0.1762, 0.9987] |
| mixed | shuffled | mixture_mse | 0/3 | 0.7420 | [0.2773, 1.0745] |
| from_SLR | plain | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| from_SLR | aligned | mixture_mse | 0/3 | 0.9923 | [-0.8010, 2.5213] |
| from_SLR | global | mixture_mse | 0/3 | 0.9367 | [-0.7605, 2.2760] |
| from_SLR | shuffled | mixture_mse | 0/3 | 0.8998 | [-0.8244, 2.2329] |
| from_ipod | plain | plain_mse | 0/3 | 0.4546 | [0.1312, 0.8048] |
| from_ipod | aligned | plain_mse | 0/3 | 2.3109 | [1.7867, 3.1140] |
| from_ipod | global | plain_mse | 0/3 | 2.2742 | [1.5822, 3.1059] |
| from_ipod | shuffled | plain_mse | 0/3 | 2.4310 | [1.7080, 3.3812] |
