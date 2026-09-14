# skin_train_branch_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Граф/свёртка только при обучении; направленные gains без универсальной победы и независимого подтверждения.

[Полная папка артефактов](../../../../docs/benchmarks/skin_train_branch_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_train_branch_v1/report.md)

SHA-256 отчёта: `0c6518a67bf975d269421a255d8b5c62c14a6345a100f3281bb7c7d1fd57d7f1`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_train_branch_protocol_v1.md](../../../../docs/research/skin_train_branch_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_train_branch_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_train_branch_audit.py](../modules/scripts__skin_train_branch_audit.md)
- [scripts/skin_train_branch_model.py](../modules/scripts__skin_train_branch_model.md)
- [scripts/skin_train_branch_report.py](../modules/scripts__skin_train_branch_report.md)
- [scripts/skin_train_branch_train.py](../modules/scripts__skin_train_branch_train.md)
- [tests/test_skin_train_branch.py](../tests/tests__test_skin_train_branch.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_train_branch_v1/audit.json) | {"status": "PASS"} | `853848fa02f80f94997896dc7affcea9cdd5bacb3554272a7f95bbb27f1afc28` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_train_branch_v1/audit_mixed.json) | {"status": "PASS"} | `02ed77667016460e784b6320d5e5f7d349957d0cf81ec4862db08ebdb9025aaa` |
| [source_lock.json](../../../../docs/benchmarks/skin_train_branch_v1/source_lock.json) | {} | `29cf07ed73f00dbef4976f7693749914922c5ebaa94a5425b438564ef7a00acc` |
| [summary.json](../../../../docs/benchmarks/skin_train_branch_v1/summary.json) | {"scope": "Post-discovery source follow-up; no independent test, no ensemble; strong local controls retained"} | `d9a2095824f6be4996f3eeffcd56653125ae591cce882bb9eeaeab195763b9e9` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
| --- | --- | --- | --- | --- |
| mixed | graph_always | 3.6394 | 7.1259 | 3.5921, 3.6626, 3.6634 |
| mixed | conv_always | 5.2915 | 10.8496 | 5.0627, 5.1969, 5.6149 |
| mixed | graph_drop | 3.4493 | 7.1375 | 3.4517, 3.4374, 3.4589 |
| mixed | conv_drop | 3.4777 | 7.2189 | 3.4505, 3.5157, 3.4669 |
| mixed | original_plain | 3.4855 | 7.1819 | 3.4851, 3.5127, 3.4587 |
| mixed | ordinary_conv3 | 3.4901 | 7.2079 | 3.4654, 3.5320, 3.4728 |
| mixed | capture_plain_mse | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | capture_mixture_mse | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| from_SLR | graph_always | 5.8339 | 10.9445 | 4.6711, 6.0218, 6.8088 |
| from_SLR | conv_always | 6.6652 | 12.6219 | 6.8186, 6.3140, 6.8629 |
| from_SLR | graph_drop | 5.4829 | 10.7564 | 5.7274, 5.1249, 5.5964 |
| from_SLR | conv_drop | 6.0488 | 12.1188 | 5.7965, 6.3821, 5.9676 |
| from_SLR | original_plain | 5.6063 | 10.9750 | 5.3482, 5.4913, 5.9794 |
| from_SLR | ordinary_conv3 | 6.6501 | 13.1755 | 5.9482, 6.8641, 7.1381 |
| from_SLR | capture_plain_mse | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | capture_mixture_mse | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_ipod | graph_always | 4.9736 | 9.4600 | 5.2933, 4.8977, 4.7298 |
| from_ipod | conv_always | 9.7375 | 15.8605 | 10.2993, 9.1095, 9.8036 |
| from_ipod | graph_drop | 5.7700 | 10.1397 | 5.7413, 5.6008, 5.9681 |
| from_ipod | conv_drop | 5.6434 | 9.7481 | 5.2833, 5.7227, 5.9242 |
| from_ipod | original_plain | 6.1399 | 10.7483 | 5.8650, 6.3605, 6.1943 |
| from_ipod | ordinary_conv3 | 5.7738 | 10.5538 | 5.6855, 6.0182, 5.6176 |
| from_ipod | capture_plain_mse | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | capture_mixture_mse | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |

### Таблица 2

| Protocol | Method | Mean at80% | Training active parameters | Max fit+selection MiB |
| --- | --- | --- | --- | --- |
| mixed | graph_always | 3.6342 | 990727 | 127.56 |
| mixed | conv_always | 5.4543 | 993028 | 125.05 |
| mixed | graph_drop | 3.4315 | 990727 | 127.16 |
| mixed | conv_drop | 3.4888 | 993028 | 124.65 |
| from_SLR | graph_always | 5.8116 | 990727 | 123.23 |
| from_SLR | conv_always | 6.4832 | 993028 | 121.64 |
| from_SLR | graph_drop | 5.4922 | 990727 | 123.23 |
| from_SLR | conv_drop | 5.6943 | 993028 | 120.72 |
| from_ipod | graph_always | 4.8452 | 990727 | 125.23 |
| from_ipod | conv_always | 9.9902 | 993028 | 122.72 |
| from_ipod | graph_drop | 5.7012 | 990727 | 125.23 |
| from_ipod | conv_drop | 5.5595 | 993028 | 122.72 |
