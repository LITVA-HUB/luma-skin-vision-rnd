# phone_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Первый фиксированный Samsung/Oppo тест; строгий HDF5 loader ограничивал достижимое покрытие.

[Полная папка артефактов](../../../../docs/benchmarks/phone_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/phone_v1_report.md)

SHA-256 отчёта: `6403016002954cf00cd320bf5e0ae270a79f63d9df839baa2ac66c669b813e24`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [phone_v1_alias_report.md](../../../../docs/benchmarks/phone_v1_alias_report.md)
- [phone_v1_report.md](../../../../docs/benchmarks/phone_v1_report.md)
- [phone_loader_findings.md](../../../../docs/data/phone_loader_findings.md)
- [smartphone_benchmark_plan.md](../../../../docs/data/smartphone_benchmark_plan.md)
- [phone_alias_protocol_v1_2.md](../../../../docs/research/phone_alias_protocol_v1_2.md)
- [phone_benchmark_protocol_v1.md](../../../../docs/research/phone_benchmark_protocol_v1.md)
- [phone_loader_repair_v1_1.md](../../../../docs/research/phone_loader_repair_v1_1.md)
- [phone_reference_protocol_v1.md](../../../../docs/research/phone_reference_protocol_v1.md)
- [cc_v5_phone_progress.md](../../../../docs/skolkovo/cc_v5_phone_progress.md)
- [phone_component_evidence_2026_09_11.md](../../../../docs/skolkovo/phone_component_evidence_2026_09_11.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/cc_phone_alias.py](../modules/scripts__cc_phone_alias.md)
- [scripts/cc_phone_alias_report.py](../modules/scripts__cc_phone_alias_report.md)
- [scripts/cc_phone_benchmark.py](../modules/scripts__cc_phone_benchmark.md)
- [scripts/cc_phone_benchmark_v1_1.py](../modules/scripts__cc_phone_benchmark_v1_1.md)
- [scripts/cc_phone_data.py](../modules/scripts__cc_phone_data.md)
- [scripts/cc_phone_loader_audit.py](../modules/scripts__cc_phone_loader_audit.md)
- [scripts/cc_phone_prepare.py](../modules/scripts__cc_phone_prepare.md)
- [scripts/cc_phone_report.py](../modules/scripts__cc_phone_report.md)
- [scripts/cc_phone_resume.py](../modules/scripts__cc_phone_resume.md)
- [tests/test_cc_phone_alias.py](../tests/tests__test_cc_phone_alias.md)
- [tests/test_cc_phone_benchmark.py](../tests/tests__test_cc_phone_benchmark.md)
- [tests/test_cc_phone_data.py](../tests/tests__test_cc_phone_data.md)
- [tests/test_cc_phone_loader_audit.py](../tests/tests__test_cc_phone_loader_audit.md)
- [tests/test_cc_phone_prepare.py](../tests/tests__test_cc_phone_prepare.md)
- [tests/test_cc_phone_resume.py](../tests/tests__test_cc_phone_resume.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [family_summary.json](../../../../docs/benchmarks/phone_v1/family_summary.json) | {} | `9938c8abcbca7154b97ead1dc36596ea76258d5fbf2bdf74fd415e698c91c491` |
| [independent_verification.json](../../../../docs/benchmarks/phone_v1/independent_verification.json) | {} | `8a4c057b421f7193cb55fcbee9ecb6f0d47dc8982cf893cc6d512ea44593b230` |
| [loader_preflight_v1_1.json](../../../../docs/benchmarks/phone_v1/loader_preflight_v1_1.json) | {} | `570abf4c6184d0f8b11d816e248183325593e9e1f6199c44a628e5a17504e7b8` |
| [loader_repair_v1_1.json](../../../../docs/benchmarks/phone_v1/loader_repair_v1_1.json) | {} | `8d7e55eb0bd973463e62b8e48b2b081eea2abdea1bd0d2fa9c34a39be5adae96` |
| [method_lock.json](../../../../docs/benchmarks/phone_v1/method_lock.json) | {"status": "FROZEN BEFORE RESERVED PHONE PIXEL/GT DECODING"} | `f47778497a76782da288fe6740d8ef12388fa080ead9a08db618b7a6de393021` |
| [preparation.json](../../../../docs/benchmarks/phone_v1/preparation.json) | {} | `b5f074719b9773d55406aaf440ce85b7278d268d8ce9d30925a8c53f865a044c` |
| [results.json](../../../../docs/benchmarks/phone_v1/results.json) | {"status": "MEASURED CUSTOM SOURCE-TO-PHONE TRANSFER; NOT SKIN/JPEG/HEIC VALIDATION"} | `fe405f92d81e1ba794da942bdbd54d4104261ee8898fc490df70afcf93a01236` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Family (mean of3 seeds for CNNs; no ensemble) | Pooled meanР’В° | Pooled risk80Р’В° | Samsung risk80Р’В° | Oppo risk80Р’В° |
| --- | --- | --- | --- | --- |
| v2_direct | 5.137 | 4.380 | 3.782 | 5.178 |
| v2_sog | 4.662 | 3.822 | 3.899 | 3.795 |
| gw_ridge1 | 5.798 | 4.445 | 4.481 | 4.242 |
| direct_hgb7 | 5.386 | 4.599 | 4.056 | 5.030 |
| v5_point | 5.168 | untrained | untrained | untrained |
| v5_posterior_random | 4.949 | 4.861 | 4.285 | 5.442 |
| v5_action_random | 4.913 | 4.731 | 4.092 | 5.358 |
| v5_transport_random | 5.159 | 4.971 | 4.332 | 5.556 |
| v5_transport_policy | 4.993 | 4.807 | 4.252 | 5.457 |
| v5_transport_gradient | 4.962 | 4.927 | 4.567 | 5.267 |
| gray_world | 6.568 | 5.834 | 5.849 | 5.948 |
| max_rgb | 13.843 | 12.656 | 13.124 | 12.171 |
| shades_gray | 5.147 | 4.915 | 4.693 | 5.146 |
| gray_edge | 5.263 | 5.125 | 4.543 | 5.710 |

### Таблица 2

| V2 SoG fixed coverage | Mean errorР’В° | Accepted/scorable | Accepted/planned |
| --- | --- | --- | --- |
| 100% | 4.328 | 70/72 | 70/88 |
| 95% | 4.284 | 68/72 | 68/88 |
| 90% | 4.037 | 64/72 | 64/88 |
| 80% | 3.822 | 57/72 | 57/88 |
| 70% | 3.714 | 50/72 | 50/88 |
| 60% | 3.716 | 43/72 | 43/88 |

### Таблица 3

| Method | Accepted/scorable | Actual coverage% | MeanР’В° |
| --- | --- | --- | --- |
| ccv2_direct_large_g0_s17 | 44/72 | 61.1 | 4.348 |
| ccv2_direct_large_g0_s29 | 60/72 | 83.3 | 4.418 |
| ccv2_direct_large_g0_s43 | 46/72 | 63.9 | 4.386 |
| ccv2_sog_large_g0_s17 | 31/72 | 43.1 | 3.695 |
| ccv2_sog_large_g0_s29 | 46/72 | 63.9 | 3.385 |
| ccv2_sog_large_g0_s43 | 39/72 | 54.2 | 3.566 |
| gw_ridge1 | 34/72 | 47.2 | 3.949 |
| direct_hgb7 | 37/72 | 51.4 | 4.486 |
