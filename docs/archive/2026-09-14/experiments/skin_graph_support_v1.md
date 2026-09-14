# skin_graph_support_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Комбинация графа и реальных paired patches помогает слабому transfer baseline, но не сильнейшим рецептам.

[Полная папка артефактов](../../../../docs/benchmarks/skin_graph_support_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_graph_support_v1/report.md)

SHA-256 отчёта: `4f66faa581b99e6dedeb15b19b25e78f7145e51e00c471229ab30c94591344ac`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_graph_support_protocol_v1.md](../../../../docs/research/skin_graph_support_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_graph_support_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_graph_support_model.py](../modules/scripts__skin_graph_support_model.md)
- [scripts/skin_graph_support_train.py](../modules/scripts__skin_graph_support_train.md)
- [scripts/skin_graph_support_verify.py](../modules/scripts__skin_graph_support_verify.md)
- [tests/test_skin_graph_support.py](../tests/tests__test_skin_graph_support.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_graph_support_v1/audit.json) | {"status": "PASS"} | `3d04e02b57ca9e2da0c44d6f49ca5535a57bb0a5e52b92ed0d8f7eb08dc9be84` |
| [source_lock.json](../../../../docs/benchmarks/skin_graph_support_v1/source_lock.json) | {} | `afa4aeac4ee722ffcfc0e2deb7f2d8b112c26610e35074fe51c70602da51b369` |
| [summary.json](../../../../docs/benchmarks/skin_graph_support_v1/summary.json) | {"scope": "SOURCE exploratory only, no new independent test"} | `db13122456a0749b966681ad5956037e9a64e0ec2db73b84488f28de494481fd` |
| [test_receipt.json](../../../../docs/benchmarks/skin_graph_support_v1/test_receipt.json) | {"passed": 348} | `425efc8b44e57ab8983f24befd6d0bbdcf51cd25c1bda3e658de4878bf179bcb` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol / family | Initializations | Mean | Median | p95 | Common 80% |
| --- | --- | --- | --- | --- | --- |
| mixed/graph_paired | 3 | 3.5143 | 3.0417 | 7.0727 | 3.4239 |
| mixed/graph_raw | 3 | 3.4621 | 2.9707 | 7.1813 | 3.3877 |
| mixed/plain_paired | 3 | 3.5438 | 3.1156 | 7.0690 | 3.4662 |
| mixed/plain_raw | 3 | 3.4855 | 2.9738 | 7.1819 | 3.4136 |
| from_SLR/graph_paired | 3 | 5.2801 | 4.9537 | 9.7880 | 5.1110 |
| from_SLR/graph_raw | 3 | 5.4616 | 5.0482 | 10.3841 | 5.6054 |
| from_SLR/plain_paired | 3 | 5.4804 | 5.0277 | 10.3911 | 5.2830 |
| from_SLR/plain_raw | 3 | 5.6063 | 5.1288 | 10.9750 | 5.7226 |
| from_ipod/graph_paired | 3 | 5.6610 | 5.5284 | 9.8796 | 5.9994 |
| from_ipod/graph_raw | 3 | 5.9890 | 5.8624 | 10.6926 | 6.4210 |
| from_ipod/plain_paired | 3 | 5.8331 | 5.7398 | 10.2474 | 6.2410 |
| from_ipod/plain_raw | 3 | 6.1399 | 5.9801 | 10.7483 | 6.5662 |

### Таблица 2

| Protocol | Candidate vs control | Mean difference | Patient 95% interval |
| --- | --- | --- | --- |
| mixed | graph_paired vs plain_raw | 0.0288 | [-0.0754, 0.1235] |
| mixed | graph_paired vs plain_paired | -0.0295 | [-0.0812, 0.0280] |
| mixed | graph_paired vs graph_raw | 0.0522 | [-0.0331, 0.1375] |
| mixed | graph_raw vs plain_raw | -0.0234 | [-0.0669, 0.0242] |
| mixed | plain_paired vs plain_raw | 0.0583 | [-0.0289, 0.1456] |
| from_SLR | graph_paired vs plain_raw | -0.3263 | [-1.2608, 0.4712] |
| from_SLR | graph_paired vs plain_paired | -0.2004 | [-0.3579, -0.0473] |
| from_SLR | graph_paired vs graph_raw | -0.1815 | [-0.7901, 0.6997] |
| from_SLR | graph_raw vs plain_raw | -0.1448 | [-0.4707, 0.2650] |
| from_SLR | plain_paired vs plain_raw | -0.1259 | [-0.9029, 0.6671] |
| from_ipod | graph_paired vs plain_raw | -0.4790 | [-0.6524, -0.2328] |
| from_ipod | graph_paired vs plain_paired | -0.1721 | [-0.3209, -0.0135] |
| from_ipod | graph_paired vs graph_raw | -0.3281 | [-0.4861, -0.1061] |
| from_ipod | graph_raw vs plain_raw | -0.1509 | [-0.1663, -0.1266] |
| from_ipod | plain_paired vs plain_raw | -0.3068 | [-0.3697, -0.2193] |
