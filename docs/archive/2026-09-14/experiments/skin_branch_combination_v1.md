# skin_branch_combination_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Равновесные пары уже обученных моделей: source gains при удвоенной стоимости, не новый независимый результат.

[Полная папка артефактов](../../../../docs/benchmarks/skin_branch_combination_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_branch_combination_v1/report.md)

SHA-256 отчёта: `d011f8d4be1140fb12e3ff3150b043ca88dec2793309197aefc8200a92826114`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_branch_combination_protocol_v1.md](../../../../docs/research/skin_branch_combination_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_branch_combination_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_branch_combination.py](../modules/scripts__skin_branch_combination.md)
- [scripts/skin_branch_combination_curves.py](../modules/scripts__skin_branch_combination_curves.md)
- [scripts/skin_branch_combination_profile.py](../modules/scripts__skin_branch_combination_profile.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [curve_audit.json](../../../../docs/benchmarks/skin_branch_combination_v1/curve_audit.json) | {"status": "PASS"} | `3cd3a537a6050c09cb1ff349c99feb7ac68e97d9e9914be7c846ac29a1aa08b5` |
| [profile.json](../../../../docs/benchmarks/skin_branch_combination_v1/profile.json) | {"scope": "Batch1 CUDA event; prepared1x64x18tokens to averaged nativeLab; noJPEG/features/CPU/risk/localization"} | `fe64a6a6cf6cd8769a0e426cc81922ef33e9b0aec6e0f92c32083476cc3e6ec3` |
| [protocol_lock.json](../../../../docs/benchmarks/skin_branch_combination_v1/protocol_lock.json) | {} | `0304e04872e73720ca76741439a900bbbb7e3ff5655ae0f74afba1b6bd2c3861` |
| [summary.json](../../../../docs/benchmarks/skin_branch_combination_v1/summary.json) | {"scope": "Adaptive SOURCE exploration; fixed equal averaging, uncalibrated disagreement; not independent test"} | `41832b54eb327d37b43cae0a193395e619338d1a1e0f45faf2f1e5e31318b3b3` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Pair | MeanDeltaE00 | At80% | p95 | Seed means |
| --- | --- | --- | --- | --- | --- |
| mixed | mixture_graph_always | 3.4043 | 3.4411 | 6.8769 | 3.3035, 3.4913, 3.4181 |
| mixed | mixture_graph_drop | 3.3694 | 3.3123 | 6.7629 | 3.3232, 3.3843, 3.4007 |
| mixed | mixture_conv_drop | 3.3855 | 3.3626 | 6.9525 | 3.3383, 3.3942, 3.4240 |
| mixed | mixture_plain | 3.3680 | 3.2749 | 6.7946 | 3.3497, 3.3820, 3.3723 |
| mixed | plain_graph_always | 3.4136 | 3.3561 | 6.9739 | 3.3620, 3.4608, 3.4181 |
| from_SLR | mixture_graph_always | 5.0697 | 5.3043 | 9.9307 | 4.4972, 5.3626, 5.3494 |
| from_SLR | mixture_graph_drop | 5.0123 | 5.2217 | 10.0444 | 5.0950, 4.8850, 5.0570 |
| from_SLR | mixture_conv_drop | 5.2698 | 5.3972 | 10.4060 | 5.2046, 5.4771, 5.1278 |
| from_SLR | mixture_plain | 5.1702 | 5.1650 | 9.9795 | 5.2431, 5.2800, 4.9874 |
| from_SLR | plain_graph_always | 5.2237 | 5.4085 | 10.2161 | 4.8164, 5.6897, 5.1650 |
| from_ipod | mixture_graph_always | 5.2806 | 5.4846 | 10.0112 | 5.4262, 5.4201, 4.9955 |
| from_ipod | mixture_graph_drop | 5.7810 | 5.8445 | 10.4380 | 5.7219, 5.8938, 5.7272 |
| from_ipod | mixture_conv_drop | 5.7114 | 5.7515 | 10.2254 | 5.4546, 5.9353, 5.7442 |
| from_ipod | mixture_plain | 5.6523 | 5.6836 | 10.1399 | 5.4673, 5.9243, 5.5652 |
| from_ipod | plain_graph_always | 5.1284 | 5.3470 | 9.4521 | 5.2316, 5.1516, 5.0021 |
