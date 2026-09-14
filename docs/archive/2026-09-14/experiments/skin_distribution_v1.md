# skin_distribution_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Условные распределения цвета и решения по ожидаемой ошибке; универсального выигрыша над прямой регрессией нет.

[Полная папка артефактов](../../../../docs/benchmarks/skin_distribution_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_distribution_v1/report.md)

SHA-256 отчёта: `97b794556bcdff8f181610aa9e3ca0ae23f4a7cf3d0fe885baf04d194875db78`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_distribution_provenance_2026_09_11.md](../../../../docs/ip/skin_distribution_provenance_2026_09_11.md)
- [skin_distribution_integration_diagnostic_v1.md](../../../../docs/research/skin_distribution_integration_diagnostic_v1.md)
- [skin_distribution_next_decision.md](../../../../docs/research/skin_distribution_next_decision.md)
- [skin_distribution_protocol_v1.md](../../../../docs/research/skin_distribution_protocol_v1.md)
- [skin_distribution_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_distribution_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_distribution_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_distribution_audit.py](../modules/scripts__skin_distribution_audit.md)
- [scripts/skin_distribution_integration.py](../modules/scripts__skin_distribution_integration.md)
- [scripts/skin_distribution_model.py](../modules/scripts__skin_distribution_model.md)
- [scripts/skin_distribution_report.py](../modules/scripts__skin_distribution_report.md)
- [scripts/skin_distribution_train.py](../modules/scripts__skin_distribution_train.md)
- [tests/test_skin_distribution_model.py](../tests/tests__test_skin_distribution_model.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_distribution_v1/audit.json) | {"status": "PASS"} | `2b688ab0a266bbb04cded3a1715367b1aa90db828d4f5a55cd14e916d0e4da32` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_distribution_v1/audit_mixed.json) | {"status": "PASS"} | `59f60d1ead72980285fa440bd94fdb86020a17069548cca0d50b8ee949635739` |
| [integration_diagnostic.json](../../../../docs/benchmarks/skin_distribution_v1/integration_diagnostic.json) | {"scope": "POSTHOC numeric integration sensitivity; no refit or primary replacement"} | `620e83c7da8a6d55193befe61ad5480e5f6d311be20bb6f23b51514416e3d995` |
| [integration_lock.json](../../../../docs/benchmarks/skin_distribution_v1/integration_lock.json) | {"scope": "posthoc integration only, all18density fits"} | `094c7efa4d4e2861ad5093b58a0b473e36fb76e0075f2f6507096f64b8c261e5` |
| [source_lock.json](../../../../docs/benchmarks/skin_distribution_v1/source_lock.json) | {} | `90a92f4e209d69ffc18f3bf17f4826eac317bf75b515d9b4bec1d00744d3d9d6` |
| [summary.json](../../../../docs/benchmarks/skin_distribution_v1/summary.json) | {"scope": "REPRODUCED SOURCE ONLY; repeated validation; no confirmatory test"} | `257b4c10c11e58e2702b8948d65ab1b097212b25b81037da14d3ad2a647e3b04` |
| [verification.json](../../../../docs/benchmarks/skin_distribution_v1/verification.json) | {"passed": 322, "seconds": 34.33, "scope": "Full existing suite including4new density tests; experimental script replays audited separately"} | `7a2558168d8c68ce3211ef70319d87d0b39fc6ee5c6c5a677711917c4678cbe1` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Training / decision | Mean | Median | p95 | At80% |
| --- | --- | --- | --- | --- | --- |
| mixed | mse_mode/mean | 3.4406 | 2.9776 | 6.9316 | 3.4407 |
| mixed | mse/mean | 3.4694 | 2.9208 | 7.2176 | 3.5295 |
| mixed | gaussian/mean | 3.4605 | 2.9191 | 7.5348 | 3.2709 |
| mixed | gaussian/decision3 | 3.4605 | 2.9191 | 7.5348 | 3.2709 |
| mixed | mdn4/mean | 3.5484 | 3.0789 | 7.3575 | 3.4376 |
| mixed | mdn4/decision3 | 3.7181 | 3.1801 | 8.2025 | 3.5810 |
| mixed | mdn4/decision2 | 3.5766 | 3.0898 | 7.4244 | 3.4507 |
| from_SLR | mse_mode/mean | 5.0193 | 4.8340 | 9.7825 | 5.1087 |
| from_SLR | mse/mean | 4.9838 | 4.6078 | 9.4551 | 5.0986 |
| from_SLR | gaussian/mean | 5.6571 | 5.4041 | 11.2796 | 5.4955 |
| from_SLR | gaussian/decision3 | 5.6571 | 5.4041 | 11.2796 | 5.4955 |
| from_SLR | mdn4/mean | 5.4148 | 4.8055 | 11.0887 | 5.2370 |
| from_SLR | mdn4/decision3 | 5.4146 | 4.8235 | 10.7878 | 5.3228 |
| from_SLR | mdn4/decision2 | 5.4057 | 4.8141 | 10.8585 | 5.2748 |
| from_ipod | mse_mode/mean | 5.9635 | 5.5527 | 11.0896 | 5.7914 |
| from_ipod | mse/mean | 6.3885 | 6.1364 | 11.5979 | 6.6570 |
| from_ipod | gaussian/mean | 5.7474 | 5.4467 | 10.2485 | 5.9326 |
| from_ipod | gaussian/decision3 | 5.7474 | 5.4467 | 10.2485 | 5.9326 |
| from_ipod | mdn4/mean | 6.4005 | 5.9157 | 10.9951 | 6.8924 |
| from_ipod | mdn4/decision3 | 6.8962 | 6.6337 | 11.6709 | 7.2382 |
| from_ipod | mdn4/decision2 | 6.7023 | 6.2623 | 11.3796 | 7.0791 |

### Таблица 2

| Protocol | Candidate vs control | Image mean difference | Patient95% interval | All3seeds improve |
| --- | --- | --- | --- | --- |
| mixed | mse/mean vs mse_mode/mean | 0.0288 | [-0.0705, 0.1130] | False |
| mixed | gaussian/mean vs mse_mode/mean | 0.0199 | [-0.0889, 0.1109] | False |
| mixed | mdn4/mean vs mse_mode/mean | 0.1078 | [0.0298, 0.1733] | False |
| mixed | mdn4/decision3 vs gaussian/decision3 | 0.2576 | [0.1405, 0.3642] | False |
| mixed | mdn4/decision3 vs mdn4/mean | 0.1697 | [0.1175, 0.2194] | False |
| from_SLR | mse/mean vs mse_mode/mean | -0.0355 | [-0.1798, 0.1745] | False |
| from_SLR | gaussian/mean vs mse_mode/mean | 0.6378 | [-0.4168, 1.2731] | False |
| from_SLR | mdn4/mean vs mse_mode/mean | 0.3955 | [-0.9054, 1.1268] | False |
| from_SLR | mdn4/decision3 vs gaussian/decision3 | -0.2425 | [-0.6360, 0.2521] | True |
| from_SLR | mdn4/decision3 vs mdn4/mean | -0.0002 | [-0.3280, 0.1824] | False |
| from_ipod | mse/mean vs mse_mode/mean | 0.4250 | [-0.5884, 1.3422] | False |
| from_ipod | gaussian/mean vs mse_mode/mean | -0.2161 | [-0.7385, 0.4679] | False |
| from_ipod | mdn4/mean vs mse_mode/mean | 0.4370 | [-0.6224, 1.4260] | False |
| from_ipod | mdn4/decision3 vs gaussian/decision3 | 1.1488 | [0.2460, 1.8868] | False |
| from_ipod | mdn4/decision3 vs mdn4/mean | 0.4957 | [0.1298, 0.9288] | False |

### Таблица 3

| Protocol | Density | Mean1024 | Mean4096 | At80%4096 |
| --- | --- | --- | --- | --- |
| mixed | gaussian | 3.4605 | 3.4605 | 3.2593 |
| mixed | mdn4 | 3.6015 | 3.5993 | 3.4788 |
| from_SLR | gaussian | 5.6571 | 5.6571 | 5.4955 |
| from_SLR | mdn4 | 5.3926 | 5.3897 | 5.2960 |
| from_ipod | gaussian | 5.7474 | 5.7474 | 5.9347 |
| from_ipod | mdn4 | 6.8491 | 6.8491 | 7.1584 |
