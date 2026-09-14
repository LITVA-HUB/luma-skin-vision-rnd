# chromaseed_fast_kernel_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_fast_kernel_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/report.md)

SHA-256 отчёта: `97eeef0a7a415906d62b63ef88ef1ce0489f5a3176623bb78c00653473e9a6db`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_fast_kernel_model_card.md](../../../../docs/architecture/chromaseed_fast_kernel_model_card.md)
- [chromaseed_fast_kernel_next_decision.md](../../../../docs/research/chromaseed_fast_kernel_next_decision.md)
- [chromaseed_fast_kernel_v1_protocol.md](../../../../docs/research/chromaseed_fast_kernel_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_fast_kernel.py](../modules/scripts__chromaseed_fast_kernel.md)
- [scripts/chromaseed_fast_kernel_audit.py](../modules/scripts__chromaseed_fast_kernel_audit.md)
- [scripts/chromaseed_fast_kernel_report.py](../modules/scripts__chromaseed_fast_kernel_report.md)
- [scripts/chromaseed_fast_kernel_runtime.py](../modules/scripts__chromaseed_fast_kernel_runtime.md)
- [scripts/chromaseed_fast_kernel_train.py](../modules/scripts__chromaseed_fast_kernel_train.md)
- [tests/test_chromaseed_fast_kernel.py](../tests/tests__test_chromaseed_fast_kernel.md)
- [tests/test_chromaseed_fast_kernel_protocol.py](../tests/tests__test_chromaseed_fast_kernel_protocol.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/audit.json) | {"passed": true} | `ff0afb49967c0d90fed9e2030f8aae6078e2e2280bc74e55695a981027753f7a` |
| [condensed_exact.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/condensed_exact.json) | {"passed": true} | `f06f481d486501bc03314d14f6fd54ccf348c5ea0e0708b80d1758588f5ef34e` |
| [condensed_source_lock.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/condensed_source_lock.json) | {} | `204f8c1352e56e5656d89ed3dc258c63f95965d30f42aea347fbee7868f99b8f` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/runtime.json) | {} | `35ae1ed2056afaaf1776e8432050726827956cd17c59fde0ac40087fd36e1497` |
| [runtime_real_checkpoint.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/runtime_real_checkpoint.json) | {} | `6e3af72b39d30e4e0e9cd9eedb06ae20ba70d8b89e90bf2ba3c112d02da7217b` |
| [runtime_scaling_checkpoint.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/runtime_scaling_checkpoint.json) | {} | `832ae50aa641828d3067d96b8a2aee9aca42803d6ea98dc42c9a2a619ceb4a97` |
| [selections.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/selections.json) | {} | `237ef7a98d2cd1cadf38494fe572b56482482dc58a48d2c66833125fd284ea74` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/source_lock.json) | {} | `b7936dca21e8474e3e8745edc8d9af566c432f0654e4a26e902ad26e88d1da21` |
| [summary.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/summary.json) | {} | `c7724726c0c12626f30237879b648a22ed24f353e61b4e4dc26dfbf26c0ad1d1` |
| [verification.json](../../../../docs/benchmarks/chromaseed_fast_kernel_v1/verification.json) | {"passed": true} | `fa44f82e30750b488660e6524c9a37ad69144224e26898ff64c9306999b300ea` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Роль / опоры | Полная матрица, мс | Только столбцы, мс | Столбцы + точная медиана SciPy, мс |
| --- | --- | --- | --- |
| mixed / 64 | 63.35 | 60.00 | 10.67 |
| mixed / 128 | 68.08 | 62.60 | 15.11 |
| mixed / 256 | 82.29 | 78.14 | 30.79 |
| slr_to_ipod / 64 | 13.41 | 12.99 | 4.05 |
| slr_to_ipod / 128 | 16.05 | 16.12 | 7.22 |
| slr_to_ipod / 256 | 27.46 | 28.49 | 19.62 |
| ipod_to_slr / 64 | 49.15 | 45.19 | 8.97 |
| ipod_to_slr / 128 | 53.03 | 49.08 | 13.02 |
| ipod_to_slr / 256 | 67.96 | 67.47 | 28.41 |

### Таблица 2

| Метод / опоры | Смешанные, 6 чел. | SLR → iPod, 16 чел. | iPod → SLR, 8 чел. |
| --- | --- | --- | --- |
| Полное ядро + точная медиана / 64 | 5.9542 | 8.7329 | 9.3612 |
| Полное ядро + точная медиана / 128 | 5.4387 | 8.5970 | 8.7050 |
| Полное ядро + точная медиана / 256 | 5.4003 | 8.5834 | 8.8556 |
| Столбцы + точная медиана / 64 | 5.9542 | 8.7329 | 9.3612 |
| Столбцы + точная медиана / 128 | 5.4387 | 8.5970 | 8.7050 |
| Столбцы + точная медиана / 256 | 5.4003 | 8.5834 | 8.8556 |
| Столбцы + 1024 пары / 64 | 5.9480 | 8.6882 | 8.8541 |
| Столбцы + 1024 пары / 128 | 5.9161 | 8.5910 | 8.6936 |
| Столбцы + 1024 пары / 256 | 5.4021 | 8.5742 | 8.8810 |
| Столбцы + 4096 пар / 64 | 5.9616 | 8.6969 | 9.0320 |
| Столбцы + 4096 пар / 128 | 5.4538 | 8.6025 | 8.7789 |
| Столбцы + 4096 пар / 256 | 5.4028 | 8.5867 | 8.8971 |
| Столбцы + 16384 пары / 64 | 5.9645 | 8.7015 | 9.3191 |
| Столбцы + 16384 пары / 128 | 5.4508 | 8.5873 | 8.6821 |
| Столбцы + 16384 пары / 256 | 5.4044 | 8.5830 | 8.9065 |

### Таблица 3

| Роль | Опоры | Метод | Внешняя ошибка ΔE00 |
| --- | --- | --- | --- |
| mixed | 64 | column_pairs1024 | 5.9480 |
| slr_to_ipod | 128 | column_pairs1024 | 8.5910 |
| ipod_to_slr | 128 | column_pairs1024 | 8.6936 |

### Таблица 4

| N | Метод | Fit, мс | Пик выделений, МиБ |
| --- | --- | --- | --- |
| 1024 | Полное ядро + точная медиана | 124.89 | 26.31 |
| 1024 | Столбцы + точная медиана | 111.62 | 26.31 |
| 1024 | Столбцы + 4096 пар | 11.37 | 3.82 |
| 1024 | Столбцы + точная медиана SciPy | 17.15 | 4.81 |
| 4096 | Полное ядро + точная медиана | 1950.28 | 385.35 |
| 4096 | Столбцы + точная медиана | 1693.95 | 321.21 |
| 4096 | Столбцы + 4096 пар | 31.47 | 13.76 |
| 4096 | Столбцы + точная медиана SciPy | 143.18 | 73.24 |
| 8192 | Полное ядро + точная медиана | 7791.84 | 1538.57 |
| 8192 | Столбцы + точная медиана | 6888.08 | 1282.41 |
| 8192 | Столбцы + 4096 пар | 64.01 | 27.01 |
| 8192 | Столбцы + точная медиана SciPy | 529.78 | 290.47 |
