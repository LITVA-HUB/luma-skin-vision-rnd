# chromaseed_camera_support_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_camera_support_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_camera_support_v1/report.md)

SHA-256 отчёта: `dbf3d6c414a8e26b5a36eb77be551c8b04a43442dad8b7ba64a8fc24bd451f8f`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_camera_support_next_decision.md](../../../../docs/research/chromaseed_camera_support_next_decision.md)
- [chromaseed_camera_support_v1_protocol.md](../../../../docs/research/chromaseed_camera_support_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_camera_support_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_camera_support_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_camera_support.py](../modules/scripts__chromaseed_camera_support.md)
- [scripts/chromaseed_camera_support_audit.py](../modules/scripts__chromaseed_camera_support_audit.md)
- [scripts/chromaseed_camera_support_report.py](../modules/scripts__chromaseed_camera_support_report.md)
- [scripts/chromaseed_camera_support_run.py](../modules/scripts__chromaseed_camera_support_run.md)
- [tests/test_chromaseed_camera_support.py](../tests/tests__test_chromaseed_camera_support.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_camera_support_v1/audit.json) | {"passed": true} | `d308280c2b9b2fe98997c6ddbd4f911edd1ebff1a8bd87cd498469644f80b9e0` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_camera_support_v1/source_lock.json) | {} | `59ca583d3fad34e16d0ae003b2d9d89df453bc730b0040d557407c86f2cb10a5` |
| [summary.json](../../../../docs/benchmarks/chromaseed_camera_support_v1/summary.json) | {} | `39c4cfb9a8609d497e84aa08954448bf89e610e863f4a19f37a9acd442faef45` |
| [verification.json](../../../../docs/benchmarks/chromaseed_camera_support_v1/verification.json) | {"passed": true} | `7b5befd9d660a471bb7f36af974397010d643d9cf49c91d845b6f5c51f94fa5f` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Input | Method | Person AUC | Balanced accuracy, threshold0 | Correct SLR people | Correct iPod people | Fold AUCs |
| --- | --- | --- | --- | --- | --- | --- |
| All 36 color features | linear | 1.0000 | 1.0000 | 8/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| All 36 color features | rbf | 1.0000 | 0.9375 | 7/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| Mean RGB only | linear | 0.6484 | 0.6875 | 4/8 | 14/16 | 1.0000, 0.4667, 0.8000 |
| Mean RGB only | rbf | 0.9844 | 0.7500 | 4/8 | 16/16 | 1.0000, 1.0000, 1.0000 |
| Instrument Lab (oracle) | linear | 0.5859 | 0.6250 | 5/8 | 10/16 | 0.7778, 0.4000, 0.6000 |
| Instrument Lab (oracle) | rbf | 0.5781 | 0.5938 | 4/8 | 11/16 | 0.6667, 0.4000, 0.5000 |
| Color residual after Lab (oracle) | linear | 1.0000 | 0.9688 | 8/8 | 15/16 | 1.0000, 1.0000, 1.0000 |
| Color residual after Lab (oracle) | rbf | 1.0000 | 0.9688 | 8/8 | 15/16 | 1.0000, 1.0000, 1.0000 |

### Таблица 2

| Caliper | Pairs | Matched people | Unmatched SLR / iPod | Matched images | Mean pair DeltaE00 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 8 / 16 | 0 | — |
| 2 | 2 | 4 | 6 / 14 | 176 | 1.9005 |
| 3 | 2 | 4 | 6 / 14 | 176 | 1.9005 |
| 5 | 4 | 8 | 4 / 12 | 323 | 2.7038 |
| 10 | 8 | 16 | 0 / 8 | 640 | 5.0722 |

### Таблица 3

| Caliper | Input | Method | Matched-person AUC | Balanced accuracy | SLR score above paired iPod (ties half) |
| --- | --- | --- | --- | --- | --- |
| 1 | All 36 color features | linear | — | — | — |
| 1 | All 36 color features | rbf | — | — | — |
| 1 | Mean RGB only | linear | — | — | — |
| 1 | Mean RGB only | rbf | — | — | — |
| 1 | Instrument Lab (oracle) | linear | — | — | — |
| 1 | Instrument Lab (oracle) | rbf | — | — | — |
| 1 | Color residual after Lab (oracle) | linear | — | — | — |
| 1 | Color residual after Lab (oracle) | rbf | — | — | — |
| 2 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 2 | All 36 color features | rbf | 1.0000 | 0.7500 | 1.0000 |
| 2 | Mean RGB only | linear | 0.5000 | 0.5000 | 0.5000 |
| 2 | Mean RGB only | rbf | 1.0000 | 0.5000 | 1.0000 |
| 2 | Instrument Lab (oracle) | linear | 0.2500 | 0.5000 | 0.0000 |
| 2 | Instrument Lab (oracle) | rbf | 0.2500 | 0.5000 | 0.0000 |
| 2 | Color residual after Lab (oracle) | linear | 1.0000 | 0.7500 | 1.0000 |
| 2 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 3 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 3 | All 36 color features | rbf | 1.0000 | 0.7500 | 1.0000 |
| 3 | Mean RGB only | linear | 0.5000 | 0.5000 | 0.5000 |
| 3 | Mean RGB only | rbf | 1.0000 | 0.5000 | 1.0000 |
| 3 | Instrument Lab (oracle) | linear | 0.2500 | 0.5000 | 0.0000 |
| 3 | Instrument Lab (oracle) | rbf | 0.2500 | 0.5000 | 0.0000 |
| 3 | Color residual after Lab (oracle) | linear | 1.0000 | 0.7500 | 1.0000 |
| 3 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 5 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 5 | All 36 color features | rbf | 1.0000 | 0.8750 | 1.0000 |
| 5 | Mean RGB only | linear | 0.6875 | 0.6250 | 0.7500 |
| 5 | Mean RGB only | rbf | 1.0000 | 0.7500 | 1.0000 |
| 5 | Instrument Lab (oracle) | linear | 0.3125 | 0.5000 | 0.0000 |
| 5 | Instrument Lab (oracle) | rbf | 0.3750 | 0.5000 | 0.0000 |
| 5 | Color residual after Lab (oracle) | linear | 1.0000 | 0.8750 | 1.0000 |
| 5 | Color residual after Lab (oracle) | rbf | 1.0000 | 1.0000 | 1.0000 |
| 10 | All 36 color features | linear | 1.0000 | 1.0000 | 1.0000 |
| 10 | All 36 color features | rbf | 1.0000 | 0.9375 | 1.0000 |
| 10 | Mean RGB only | linear | 0.6719 | 0.6250 | 0.6250 |
| 10 | Mean RGB only | rbf | 0.9844 | 0.7500 | 1.0000 |
| 10 | Instrument Lab (oracle) | linear | 0.5156 | 0.5625 | 0.5000 |
| 10 | Instrument Lab (oracle) | rbf | 0.5312 | 0.5625 | 0.2500 |
| 10 | Color residual after Lab (oracle) | linear | 1.0000 | 0.9375 | 1.0000 |
| 10 | Color residual after Lab (oracle) | rbf | 1.0000 | 0.9375 | 1.0000 |

### Таблица 4

| Role | Space | Fit reference mean / p95 | Query mean / p95 | Query mass above fit radius |
| --- | --- | --- | --- | --- |
| Mixed cameras | color36 | 0.2422 / 0.5030 | 0.2413 / 0.4245 | 1.52% |
| Mixed cameras | native_lab | 1.5436 / 3.0313 | 1.6295 / 3.2837 | 10.61% |
| SLR → iPod | color36 | 0.2251 / 0.3964 | 0.3685 / 0.7860 | 28.31% |
| SLR → iPod | native_lab | 2.4792 / 6.7486 | 2.5410 / 6.2025 | 3.98% |
| iPod → SLR | color36 | 0.2819 / 0.5171 | 0.8236 / 2.3322 | 54.20% |
| iPod → SLR | native_lab | 1.6669 / 3.5974 | 1.8722 / 3.9588 | 9.34% |

### Таблица 5

| Camera | People / rows | Lab 5th percentile | Lab median | Lab 95th percentile |
| --- | --- | --- | --- | --- |
| SLR | 8 / 323 | (32.67, 3.33, 9.00) | (50.00, 9.00, 18.00) | (67.00, 13.33, 24.33) |
| ipod | 16 / 643 | (32.00, 1.00, 6.67) | (60.00, 7.67, 16.33) | (72.00, 14.00, 23.00) |
