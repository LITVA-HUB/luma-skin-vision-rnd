# chromaseed_architecture_scale_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_architecture_scale_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/report.md)

SHA-256 отчёта: `58af438fe7f541dee3d9659b7e20403d727c37d5b830cdcc8a7b493c0911a60d`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_architecture_scale_model_card.md](../../../../docs/architecture/chromaseed_architecture_scale_model_card.md)
- [chromaseed_architecture_scale_interim_2026-09-14.md](../../../../docs/research/chromaseed_architecture_scale_interim_2026-09-14.md)
- [chromaseed_architecture_scale_next_decision.md](../../../../docs/research/chromaseed_architecture_scale_next_decision.md)
- [chromaseed_architecture_scale_v1_protocol.md](../../../../docs/research/chromaseed_architecture_scale_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_architecture_scale.py](../modules/scripts__chromaseed_architecture_scale.md)
- [scripts/chromaseed_architecture_scale_audit.py](../modules/scripts__chromaseed_architecture_scale_audit.md)
- [scripts/chromaseed_architecture_scale_inner_review.py](../modules/scripts__chromaseed_architecture_scale_inner_review.md)
- [scripts/chromaseed_architecture_scale_plot.py](../modules/scripts__chromaseed_architecture_scale_plot.md)
- [scripts/chromaseed_architecture_scale_report.py](../modules/scripts__chromaseed_architecture_scale_report.md)
- [scripts/chromaseed_architecture_scale_run.py](../modules/scripts__chromaseed_architecture_scale_run.md)
- [scripts/chromaseed_architecture_scale_runtime.py](../modules/scripts__chromaseed_architecture_scale_runtime.md)
- [scripts/chromaseed_architecture_scale_support.py](../modules/scripts__chromaseed_architecture_scale_support.md)
- [scripts/chromaseed_architecture_scale_support_verify.py](../modules/scripts__chromaseed_architecture_scale_support_verify.md)
- [tests/test_chromaseed_architecture_scale.py](../tests/tests__test_chromaseed_architecture_scale.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/audit.json) | {"passed": true, "seconds": 283.6915474999696} | `d51ea15018e52c7bfbbccada1e1e543c28740afc5f58355af1aef903ebd843ae` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/runtime.json) | {"passed": true, "seconds": 2976.8032815000042} | `32fea9942fb1acab30305fc4204a2ecda912aec6159821f3c15bfc473ba73ada` |
| [summary.json](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/summary.json) | {} | `c8ba60b3720e765da9fff1ec683863ec5270663dba236e4755f314ff91c566b1` |
| [verification.json](../../../../docs/benchmarks/chromaseed_architecture_scale_v1/verification.json) | {"passed": true} | `ce2a0f51c2a395aced8b3dad51e736441d13fccdeb9932e721e2f54f0734c2e9` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Модель | Параметры | Весовые данные, МБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |
| --- | --- | --- | --- | --- | --- | --- |
| patch_small | 17,374 | 0.070 | 5.6368 | 8.0045 | 8.8981 | 50.0 |
| patch5m | 4,962,566 | 19.851 | 5.6064 | 8.2801 | 8.9704 | 954.6 |
| soft_small | 15,246 | 0.061 | 5.6941 | 8.2958 | 8.5366 | 156.9 |
| soft5m | 4,846,822 | 19.388 | 5.6625 | 7.9768 | 8.8114 | 3094.9 |
| dynamic_small | 15,246 | 0.061 | 5.6927 | 8.1191 | 8.6256 | 190.6 |
| dynamic5m | 4,846,822 | 19.388 | 5.6444 | 8.1598 | 9.0015 | 3198.5 |
| pool5m | 4,851,846 | 19.408 | 5.4039 | 8.4105 | 8.5660 | 484.1 |
| np — прежний контроль | — | — | 5.7705 | 8.2943 | 8.5895 | — |
| we — прежний контроль | — | — | 5.4778 | 8.3811 | 8.3853 | — |

### Таблица 2

| Роль | Выбранный вариант | Шаги | Скорость | Внутренняя ошибка | Внешняя ошибка |
| --- | --- | --- | --- | --- | --- |
| mixed | pool5m | 2048 | 1e-05 | 4.0697 | 5.4039 |
| slr_to_ipod | pool5m | 2048 | 1e-05 | 4.6340 | 8.4105 |
| ipod_to_slr | patch5m | 2048 | 1e-05 | 4.3342 | 8.9704 |

### Таблица 3

| Семейство | Роль | Большая минус малая, ΔE00 | Условный интервал 95% | Людей с улучшением |
| --- | --- | --- | --- | --- |
| patch | mixed | -0.0304 | [-0.1588; +0.0802] | 3/6 |
| patch | slr_to_ipod | +0.2756 | [+0.0458; +0.5392] | 5/16 |
| patch | ipod_to_slr | +0.0723 | [-0.0848; +0.2293] | 4/8 |
| soft | mixed | -0.0316 | [-0.0942; +0.0035] | 3/6 |
| soft | slr_to_ipod | -0.3190 | [-0.6546; -0.0120] | 12/16 |
| soft | ipod_to_slr | +0.2748 | [+0.0768; +0.5013] | 1/8 |
| dynamic | mixed | -0.0483 | [-0.4100; +0.3481] | 3/6 |
| dynamic | slr_to_ipod | +0.0407 | [-0.0579; +0.1386] | 5/16 |
| dynamic | ipod_to_slr | +0.3758 | [+0.1747; +0.5746] | 1/8 |

### Таблица 4

| Модель | Роль | Ошибки после каждого прохода | Полная сборка трёх запусков, с |
| --- | --- | --- | --- |
| patch_small | mixed | 5.6368 | 3.902 |
| patch_small | slr_to_ipod | 8.0045 | 3.199 |
| patch_small | ipod_to_slr | 8.8981 | 3.217 |
| patch5m | mixed | 5.6064 | 154.462 |
| patch5m | slr_to_ipod | 8.2801 | 11.012 |
| patch5m | ipod_to_slr | 8.9704 | 156.408 |
| soft_small | mixed | 5.7503 → 5.7306 → 5.7118 → 5.6941 | 2.463 |
| soft_small | slr_to_ipod | 8.2947 → 8.2951 → 8.2954 → 8.2958 | 1.338 |
| soft_small | ipod_to_slr | 8.5626 → 8.5474 → 8.5396 → 8.5366 | 2.427 |
| soft5m | mixed | 5.7429 → 5.7148 → 5.6879 → 5.6625 | 111.452 |
| soft5m | slr_to_ipod | 8.2330 → 8.1231 → 8.0117 → 7.9768 | 440.825 |
| soft5m | ipod_to_slr | 8.5553 → 8.5574 → 8.6317 → 8.8114 | 440.680 |
| dynamic_small | mixed | 5.7498 → 5.7298 → 5.7107 → 5.6927 | 2.761 |
| dynamic_small | slr_to_ipod | 8.3239 → 8.2699 → 8.1889 → 8.1191 | 6.847 |
| dynamic_small | ipod_to_slr | 8.5673 → 8.5492 → 8.5635 → 8.6256 | 6.838 |
| dynamic5m | mixed | 5.7077 → 5.6486 → 5.6149 → 5.6444 | 440.926 |
| dynamic5m | slr_to_ipod | 8.2885 → 8.2626 → 8.2145 → 8.1598 | 442.774 |
| dynamic5m | ipod_to_slr | 8.6535 → 8.7245 → 8.8294 → 9.0015 | 440.565 |
| pool5m | mixed | 5.4039 | 110.186 |
| pool5m | slr_to_ipod | 8.4105 | 110.339 |
| pool5m | ipod_to_slr | 8.5660 | 28.474 |
