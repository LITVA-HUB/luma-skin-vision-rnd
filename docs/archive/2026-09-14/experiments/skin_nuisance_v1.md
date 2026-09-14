# skin_nuisance_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Learned graph, grid, global graph и constant bias; инвариантность не должна уничтожать полезный абсолютный сигнал.

[Полная папка артефактов](../../../../docs/benchmarks/skin_nuisance_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_nuisance_v1/report.md)

SHA-256 отчёта: `dc147616e87262ecc176f266799bf54352af8d1bd4f24c8b9b6c2e349c5b0e7a`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_nuisance_protocol_v1.md](../../../../docs/research/skin_nuisance_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_nuisance_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_nuisance_audit.py](../modules/scripts__skin_nuisance_audit.md)
- [scripts/skin_nuisance_model.py](../modules/scripts__skin_nuisance_model.md)
- [scripts/skin_nuisance_train.py](../modules/scripts__skin_nuisance_train.md)
- [tests/test_skin_nuisance.py](../tests/tests__test_skin_nuisance.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_nuisance_v1/audit.json) | {"status": "PASS"} | `4d5317f1f508906b685f01e70a153d4799fde31ce72a6abdb554ae9efeaa6483` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_nuisance_v1/audit_mixed.json) | {"status": "PASS"} | `cababa1d8bae03512c0b9a461c7a9e9ec3671ead424cad4a8ee8d59cdc686e76` |
| [source_lock.json](../../../../docs/benchmarks/skin_nuisance_v1/source_lock.json) | {} | `ce2f43aa6d162b3c0b690fb130283c33835c389b949f956e5184fe836362b959` |
| [summary.json](../../../../docs/benchmarks/skin_nuisance_v1/summary.json) | {"scope": "Adaptive source research; native instrument skin color; seed score averaging is not ensembling"} | `202a60c79d3936d02502f63c22237884cde34d4e3abad2d7d2597bc0338f711c` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
| --- | --- | --- | --- | --- |
| mixed | learned | 3.6394 | 7.1259 | 3.5921, 3.6626, 3.6634 |
| mixed | fixed_grid | 3.6128 | 7.2335 | 3.5904, 3.6336, 3.6144 |
| mixed | global | 3.7441 | 7.5480 | 3.7177, 3.7498, 3.7647 |
| mixed | bias | 5.5085 | 10.6669 | 5.0731, 5.7739, 5.6786 |
| mixed | historical_plain | 3.4855 | 7.1819 | 3.4851, 3.5127, 3.4587 |
| mixed | capture_plain | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | capture_mixture | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | training_only_graph | 3.6394 | 7.1259 | 3.5921, 3.6626, 3.6634 |
| from_SLR | learned | 5.8339 | 10.9445 | 4.6711, 6.0218, 6.8088 |
| from_SLR | fixed_grid | 5.2823 | 10.1647 | 4.7810, 5.8392, 5.2268 |
| from_SLR | global | 5.2972 | 9.9990 | 5.2410, 5.0996, 5.5512 |
| from_SLR | bias | 6.4028 | 12.2688 | 6.5107, 6.1860, 6.5117 |
| from_SLR | historical_plain | 5.6063 | 10.9750 | 5.3482, 5.4913, 5.9794 |
| from_SLR | capture_plain | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | capture_mixture | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | training_only_graph | 5.8339 | 10.9445 | 4.6711, 6.0218, 6.8088 |
| from_ipod | learned | 4.9736 | 9.4600 | 5.2933, 4.8977, 4.7298 |
| from_ipod | fixed_grid | 5.1586 | 9.8748 | 5.8467, 4.9717, 4.6575 |
| from_ipod | global | 5.4993 | 10.2016 | 5.4342, 5.6902, 5.3735 |
| from_ipod | bias | 8.5210 | 13.8889 | 8.3623, 8.2083, 8.9925 |
| from_ipod | historical_plain | 6.1399 | 10.7483 | 5.8650, 6.3605, 6.1943 |
| from_ipod | capture_plain | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | capture_mixture | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | training_only_graph | 4.9736 | 9.4600 | 5.2933, 4.8977, 4.7298 |

### Таблица 2

| Protocol | Method | Mean at80% | Nominal inference params | Nominal training params | Max fit+selection MiB |
| --- | --- | --- | --- | --- | --- |
| mixed | learned | 3.6342 | 924932 | 990727 | 127.56 |
| mixed | fixed_grid | 3.5503 | 924932 | 990468 | 112.66 |
| mixed | global | 3.7410 | 924932 | 990468 | 112.66 |
| mixed | bias | 5.8001 | 924932 | 990468 | 112.13 |
| from_SLR | learned | 5.8116 | 924932 | 990727 | 123.23 |
| from_SLR | fixed_grid | 5.2196 | 924932 | 990468 | 108.73 |
| from_SLR | global | 5.2712 | 924932 | 990468 | 108.73 |
| from_SLR | bias | 6.2313 | 924932 | 990468 | 108.70 |
| from_ipod | learned | 4.8452 | 924932 | 990727 | 125.23 |
| from_ipod | fixed_grid | 5.0826 | 924932 | 990468 | 110.74 |
| from_ipod | global | 5.3890 | 924932 | 990468 | 110.74 |
| from_ipod | bias | 8.4898 | 924932 | 990468 | 110.70 |
