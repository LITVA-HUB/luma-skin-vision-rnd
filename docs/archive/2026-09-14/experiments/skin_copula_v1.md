# skin_copula_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Ранги, copula и абсолютные гистограммы. Удаление абсолютного цвета ухудшает точность.

[Полная папка артефактов](../../../../docs/benchmarks/skin_copula_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_copula_v1/report.md)

SHA-256 отчёта: `f11c8385e0ec441de745830326a8e412c477b7bb404536968fb1356ed5401097`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_copula_protocol_v1.md](../../../../docs/research/skin_copula_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_copula_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_copula_audit.py](../modules/scripts__skin_copula_audit.md)
- [scripts/skin_copula_data.py](../modules/scripts__skin_copula_data.md)
- [scripts/skin_copula_feature_audit.py](../modules/scripts__skin_copula_feature_audit.md)
- [scripts/skin_copula_model.py](../modules/scripts__skin_copula_model.md)
- [scripts/skin_copula_train.py](../modules/scripts__skin_copula_train.md)
- [tests/test_skin_copula.py](../tests/tests__test_skin_copula.md)
- [tests/test_skin_copula_model.py](../tests/tests__test_skin_copula_model.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_copula_v1/audit.json) | {"status": "PASS"} | `bfb53e52368c61d2fe5abc99b15b15ed63e5ec6487e44a2b90dc9f760eaa2cd3` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_copula_v1/audit_mixed.json) | {"status": "PASS"} | `2789358fa8ebbb049333ef6e7eaaaa15e8b654572c08c56cc0ff78ace2b576d0` |
| [feature_audit.json](../../../../docs/benchmarks/skin_copula_v1/feature_audit.json) | {"status": "PASS"} | `92fd720dbd185ecc3cc708607b66a52ec74cb1a15b11593f3dedd5ff9282f6e3` |
| [feature_receipt.json](../../../../docs/benchmarks/skin_copula_v1/feature_receipt.json) | {"scope": "Only source TRAIN/VALIDATION cached real JPEG pixels; no new labels/pixels/endpoints"} | `15d621fce3f11d4ba6288bf1f6a21b3092468c89aeebda9c6b810aab446767ab` |
| [source_lock.json](../../../../docs/benchmarks/skin_copula_v1/source_lock.json) | {} | `87c405b730af47cd8afaf338b64bf0743cf435692417d45af7df602840ed4cfa` |
| [summary.json](../../../../docs/benchmarks/skin_copula_v1/summary.json) | {"scope": "Adaptive source research; native instrument skin color; seed score averaging is not ensembling"} | `df56ca386113df9c66b6fac7eff474274390b790f1cae7fb7b660011de719db3` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Method | MeanDeltaE00 | p95 | Seed means |
| --- | --- | --- | --- | --- |
| mixed | none | 3.4801 | 6.9917 | 3.5003, 3.5000, 3.4399 |
| mixed | rgb_hist | 3.6507 | 7.6661 | 3.6433, 3.6643, 3.6444 |
| mixed | copula | 3.6460 | 7.6904 | 3.6209, 3.6419, 3.6752 |
| mixed | rank_only | 5.0899 | 10.8992 | 5.1248, 5.0504, 5.0944 |
| mixed | historical_plain | 3.4855 | 7.1819 | 3.4851, 3.5127, 3.4587 |
| mixed | capture_plain | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | capture_mixture | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | training_only_graph | 3.6394 | 7.1259 | 3.5921, 3.6626, 3.6634 |
| from_SLR | none | 5.7770 | 11.2446 | 6.1093, 5.3233, 5.8983 |
| from_SLR | rgb_hist | 5.6347 | 10.5217 | 5.5596, 5.5686, 5.7759 |
| from_SLR | copula | 6.0339 | 12.2945 | 6.1230, 5.7561, 6.2225 |
| from_SLR | rank_only | 8.8349 | 18.5104 | 8.9509, 9.3180, 8.2359 |
| from_SLR | historical_plain | 5.6063 | 10.9750 | 5.3482, 5.4913, 5.9794 |
| from_SLR | capture_plain | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | capture_mixture | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | training_only_graph | 5.8339 | 10.9445 | 4.6711, 6.0218, 6.8088 |
| from_ipod | none | 5.9897 | 10.4261 | 5.8611, 5.8138, 6.2943 |
| from_ipod | rgb_hist | 7.9333 | 14.6224 | 8.1890, 8.0095, 7.6014 |
| from_ipod | copula | 5.5702 | 11.0199 | 5.3783, 6.1041, 5.2283 |
| from_ipod | rank_only | 6.1987 | 13.4579 | 6.1335, 6.5183, 5.9444 |
| from_ipod | historical_plain | 6.1399 | 10.7483 | 5.8650, 6.3605, 6.1943 |
| from_ipod | capture_plain | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | capture_mixture | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | training_only_graph | 4.9736 | 9.4600 | 5.2933, 4.8977, 4.7298 |

### Таблица 2

| Protocol | Method | Mean at80% | Nominal inference params | Nominal training params | Max fit+selection MiB |
| --- | --- | --- | --- | --- | --- |
| mixed | none | 3.4601 | 1072644 | 1072644 | 112.68 |
| mixed | rgb_hist | 3.6186 | 1072644 | 1072644 | 112.68 |
| mixed | copula | 3.7053 | 1072644 | 1072644 | 112.68 |
| mixed | rank_only | 5.3884 | 1072644 | 1072644 | 112.68 |
| from_SLR | none | 5.8585 | 1072644 | 1072644 | 107.51 |
| from_SLR | rgb_hist | 5.6146 | 1072644 | 1072644 | 107.51 |
| from_SLR | copula | 6.2065 | 1072644 | 1072644 | 107.51 |
| from_SLR | rank_only | 8.2677 | 1072644 | 1072644 | 107.51 |
| from_ipod | none | 5.9122 | 1072644 | 1072644 | 110.46 |
| from_ipod | rgb_hist | 8.3218 | 1072644 | 1072644 | 110.46 |
| from_ipod | copula | 5.5179 | 1072644 | 1072644 | 110.46 |
| from_ipod | rank_only | 5.7064 | 1072644 | 1072644 | 110.46 |
