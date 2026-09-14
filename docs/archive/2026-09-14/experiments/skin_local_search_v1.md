# skin_local_search_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/skin_local_search_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_local_search_v1/report.md)

SHA-256 отчёта: `4546be1150fe320fe7600b6654c2c489db0798607f4df9c96e0ef2cdf8f6a268`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_local_search_provenance.md](../../../../docs/ip/skin_local_search_provenance.md)
- [skin_local_search_compact_protocol.md](../../../../docs/research/skin_local_search_compact_protocol.md)
- [skin_local_search_next_decision.md](../../../../docs/research/skin_local_search_next_decision.md)
- [skin_local_search_precision_protocol.md](../../../../docs/research/skin_local_search_precision_protocol.md)
- [skin_local_search_v1_protocol.md](../../../../docs/research/skin_local_search_v1_protocol.md)
- [ablations.md](../../../../docs/benchmarks/skin_local_search_v1/ablations.md)
- [report.md](../../../../docs/benchmarks/skin_local_search_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/skin_local_search_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_local_search_compact.py](../modules/scripts__skin_local_search_compact.md)
- [scripts/skin_local_search_core.py](../modules/scripts__skin_local_search_core.md)
- [scripts/skin_local_search_precision.py](../modules/scripts__skin_local_search_precision.md)
- [scripts/skin_local_search_report.py](../modules/scripts__skin_local_search_report.md)
- [scripts/skin_local_search_supplement.py](../modules/scripts__skin_local_search_supplement.md)
- [scripts/skin_local_search_train.py](../modules/scripts__skin_local_search_train.md)
- [tests/test_skin_local_search_compact.py](../tests/tests__test_skin_local_search_compact.md)
- [tests/test_skin_local_search_core.py](../tests/tests__test_skin_local_search_core.md)
- [tests/test_skin_local_search_precision.py](../tests/tests__test_skin_local_search_precision.md)
- [tests/test_skin_local_search_protocol.py](../tests/tests__test_skin_local_search_protocol.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [compact_inner_summary.json](../../../../docs/benchmarks/skin_local_search_v1/compact_inner_summary.json) | {"scope": "secondary adaptive prefix compression; original TRAIN only"} | `cfb2a11ebc85bc3629d8a22a7551481baa2412c1cb6688561cd4766a8e5fe449` |
| [historical_reference_audit.json](../../../../docs/benchmarks/skin_local_search_v1/historical_reference_audit.json) | {"scope": "archived encoder reference on exactly same mixed cohort; different representation and historical training recipe, no new encoder training"} | `ff66c9c7eb99d12abf9c4ba708d3f002f0932d4341662b5bcad443cf0506b32a` |
| [independent_audit.json](../../../../docs/benchmarks/skin_local_search_v1/independent_audit.json) | {"status": "COMPLETE"} | `dbdc6c343b95c40fe80229914878c31c0a997113e919306a2e17b37a98958cc4` |
| [runtime_device_audit.json](../../../../docs/benchmarks/skin_local_search_v1/runtime_device_audit.json) | {"status": "COMPLETE"} | `b6e41e1d06903807417d2261948c712a37104a61705d486174ce29ce2d0dfbd5` |
| [source_lock.json](../../../../docs/benchmarks/skin_local_search_v1/source_lock.json) | {} | `9665568430b03e66cc5bb5b6b564a7bffd1e9a9e4fb726660c27e31116c57ea5` |
| [summary.json](../../../../docs/benchmarks/skin_local_search_v1/summary.json) | {} | `18326ef0fb655dec7cf6ffc5c34b53ce6e91973ab40bdbec2decf4784e46fcb0` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Протокол | Метод | Внутренний ΔE00 | Отложенные люди ΔE00 | Среднее по фото | Числовые байты | Обучение, с |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | ridge | 4.9629 | 6.5389 | 6.7605 | 756 | 0.0068 |
| mixed | krr | 4.4409 | 5.3984 | 5.5814 | 114820 | 0.0356 |
| mixed | random_rbf | 4.6058 | 5.7521 | 5.9810 | 10996 | 0.0338 |
| mixed | guided_rbf | 4.4360 | 5.8013 | 6.0149 | 10996 | 0.0757 |
| mixed | mlp | 4.3927 | 5.7856 | 5.9978 | 10996 | 0.4478 |
| slr_to_ipod | ridge | 5.2976 | 10.8739 | 11.0679 | 756 | 0.0025 |
| slr_to_ipod | krr | 5.0565 | 8.5845 | 8.8322 | 50704 | 0.0078 |
| slr_to_ipod | random_rbf | 5.1481 | 10.3636 | 10.5661 | 10996 | 0.0102 |
| slr_to_ipod | guided_rbf | 5.3455 | 10.0280 | 10.2701 | 10996 | 0.0621 |
| slr_to_ipod | mlp | 4.9140 | 9.4810 | 9.7263 | 10996 | 0.4465 |
| ipod_to_slr | ridge | 4.7992 | 7.9557 | 7.9977 | 756 | 0.0057 |
| ipod_to_slr | krr | 4.3558 | 8.9125 | 8.9911 | 100624 | 0.0245 |
| ipod_to_slr | random_rbf | 4.5974 | 8.0618 | 8.0995 | 10996 | 0.0260 |
| ipod_to_slr | guided_rbf | 4.4527 | 7.7193 | 7.7475 | 10996 | 0.0688 |
| ipod_to_slr | mlp | 4.5235 | 9.7309 | 9.8504 | 10996 | 0.5265 |

### Таблица 2

| Протокол | Метод | Файл NPZ, байт | CPU batch1 p50, мкс | CPU batch1 p95, мкс | Peak CUDA allocated, MiB |
| --- | --- | --- | --- | --- | --- |
| mixed | ridge | 2254 | 6.50 | 6.70 | 16.97 |
| mixed | krr | 116800 | 25.20 | 25.80 | 314.74 |
| mixed | random_rbf | 13004 | 16.00 | 20.11 | 328.61 |
| mixed | guided_rbf | 13004 | 15.93 | 16.23 | 328.61 |
| mixed | mlp | 13470 | 11.17 | 11.67 | 17.69 |
| slr_to_ipod | ridge | 2254 | 6.50 | 6.60 | 16.57 |
| slr_to_ipod | krr | 52684 | 17.60 | 18.30 | 74.15 |
| slr_to_ipod | random_rbf | 13004 | 15.83 | 16.45 | 153.32 |
| slr_to_ipod | guided_rbf | 13004 | 15.70 | 21.64 | 153.32 |
| slr_to_ipod | mlp | 13470 | 11.10 | 11.27 | 17.62 |
| ipod_to_slr | ridge | 2254 | 6.50 | 6.60 | 16.88 |
| ipod_to_slr | krr | 102604 | 23.40 | 24.90 | 245.92 |
| ipod_to_slr | random_rbf | 13004 | 15.70 | 16.50 | 290.29 |
| ipod_to_slr | guided_rbf | 13004 | 15.70 | 16.07 | 290.29 |
| ipod_to_slr | mlp | 13470 | 11.10 | 11.30 | 17.67 |
