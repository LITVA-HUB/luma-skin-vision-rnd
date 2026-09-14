# phone_v1_alias

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Документированное исправление HDF5 имён: 79/88 пригодных эталонов. Это повторный, уже раскрытый тест.

[Полная папка артефактов](../../../../docs/benchmarks/phone_v1_alias)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/phone_v1_alias_report.md)

SHA-256 отчёта: `daa816d64d748baf6abcf83d037fc074bf62a297e0ec5e4c54d02067151ecd34`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [phone_v1_alias_report.md](../../../../docs/benchmarks/phone_v1_alias_report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.


## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [alias_repair_lock.json](../../../../docs/benchmarks/phone_v1_alias/alias_repair_lock.json) | {"status": "Additive format repair frozen before repaired errors; original results already observed"} | `9acf3ef47412fe022406a7b81cd94a59fa43cc62aa34427ddbf0e1b5db0fdeaa` |
| [family_summary.json](../../../../docs/benchmarks/phone_v1_alias/family_summary.json) | {} | `57fd042b7c4b047a9710bca78494fa7070475feeb2490969284134b5edae5c17` |
| [independent_verification.json](../../../../docs/benchmarks/phone_v1_alias/independent_verification.json) | {} | `def1d5f8e540a7c485d42740dd43517c9776af8c0a67496fae89939a46817389` |
| [preparation.json](../../../../docs/benchmarks/phone_v1_alias/preparation.json) | {} | `a0b7d0b145552a7e43d55aa9531f0bf381e12bd9f6e47ff4d0a75658c86cbdb8` |
| [results.json](../../../../docs/benchmarks/phone_v1_alias/results.json) | {"status": "MEASURED CUSTOM SOURCE-TO-PHONE TRANSFER; NOT SKIN/JPEG/HEIC VALIDATION"} | `683cd401cb62809277be9ea2f299044c163874ae2c052c4fc266b983e96f3eb6` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Family (means across3 seeds; no ensemble) | Pooled meanВ° | Pooled risk80В° | Samsung risk80В° | Oppo risk80В° |
| --- | --- | --- | --- | --- |
| v2_direct | 4.838 | 4.498 | 3.803 | 5.210 |
| v2_sog | 4.367 | 3.951 | 4.064 | 3.817 |
| gw_ridge1 | 4.430 | 4.311 | 4.630 | 3.995 |
| direct_hgb7 | 4.658 | 4.576 | 3.971 | 5.272 |
| v5_point | 4.899 | untrained | untrained | untrained |
| v5_posterior_random | 4.740 | 5.039 | 4.387 | 5.838 |
| v5_action_random | 4.733 | 4.902 | 4.232 | 5.792 |
| v5_transport_random | 4.947 | 5.106 | 4.448 | 5.914 |
| v5_transport_policy | 4.775 | 4.907 | 4.173 | 5.893 |
| v5_transport_gradient | 4.734 | 5.051 | 4.526 | 5.549 |
| gray_world | 6.240 | 5.717 | 5.779 | 5.372 |
| max_rgb | 13.907 | 12.744 | 13.678 | 11.495 |
| shades_gray | 4.806 | 4.766 | 4.835 | 4.880 |
| gray_edge | 4.900 | 5.134 | 4.768 | 5.707 |

### Таблица 2

| Comparator | Full mean95% intervalВ° | Risk80 95% intervalВ° |
| --- | --- | --- |
| v2_direct | [-1.3245567509095824, 0.39106533345816485] | [-1.531731187086911, 0.3219256208148086] |
| gw_ridge1 | [-0.4894383691072508, 0.36501096920149434] | [-1.0090260667291566, 0.236300173496546] |

### Таблица 3

| V2 SoG nominal coverage | Mean accepted errorВ° | Accepted/scorable | Accepted/planned |
| --- | --- | --- | --- |
| 100% | 4.367 | 79/79 | 79/88 |
| 95% | 4.228 | 75/79 | 75/88 |
| 90% | 4.077 | 71/79 | 71/88 |
| 80% | 3.951 | 63/79 | 63/88 |
| 70% | 3.742 | 55/79 | 55/88 |
| 60% | 3.658 | 47/79 | 47/88 |

### Таблица 4

| Method | Accepted/scorable | Actual coverage% | MeanВ° |
| --- | --- | --- | --- |
| ccv2_direct_large_g0_s17 | 50/79 | 63.3 | 4.321 |
| ccv2_direct_large_g0_s29 | 69/79 | 87.3 | 4.573 |
| ccv2_direct_large_g0_s43 | 53/79 | 67.1 | 4.604 |
| ccv2_sog_large_g0_s17 | 36/79 | 45.6 | 3.384 |
| ccv2_sog_large_g0_s29 | 50/79 | 63.3 | 3.368 |
| ccv2_sog_large_g0_s43 | 45/79 | 57.0 | 3.945 |
| gw_ridge1 | 39/79 | 49.4 | 3.772 |
| direct_hgb7 | 44/79 | 55.7 | 4.312 |
