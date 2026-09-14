# chromaseed_widen_early_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_widen_early_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_widen_early_v1/report.md)

SHA-256 отчёта: `b498c0c9db93185a5b9ca4adfde6b9e8dd5169901f5b0a56b60248cf6ed0f0d2`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_widen_early_model_card.md](../../../../docs/architecture/chromaseed_widen_early_model_card.md)
- [chromaseed_widen_early_next_decision.md](../../../../docs/research/chromaseed_widen_early_next_decision.md)
- [chromaseed_widen_early_v1_protocol.md](../../../../docs/research/chromaseed_widen_early_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_widen_early_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_widen_early_audit.py](../modules/scripts__chromaseed_widen_early_audit.md)
- [scripts/chromaseed_widen_early_fit.py](../modules/scripts__chromaseed_widen_early_fit.md)
- [scripts/chromaseed_widen_early_report.py](../modules/scripts__chromaseed_widen_early_report.md)
- [scripts/chromaseed_widen_early_run.py](../modules/scripts__chromaseed_widen_early_run.md)
- [scripts/chromaseed_widen_early_runtime.py](../modules/scripts__chromaseed_widen_early_runtime.md)
- [tests/test_chromaseed_widen_early.py](../tests/tests__test_chromaseed_widen_early.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_widen_early_v1/audit.json) | {"passed": true, "seconds": 184.32708809996257} | `834c213218a2e56d718ddd348fcc70dbf3e7b6c9c1bd7653cd80c802e0f04ad0` |
| [checks.json](../../../../docs/benchmarks/chromaseed_widen_early_v1/checks.json) | {"passed": true} | `eb742e18ffcad0ebbe2e072a21524fe11824a11ff8bcdd015a39e938afd55cdd` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_widen_early_v1/runtime.json) | {"passed": true, "seconds": 375.9794387000147} | `67f0de4197da72229fe83025a89c0612092287eb53f94b6d3c0f6cd11a459570` |
| [summary.json](../../../../docs/benchmarks/chromaseed_widen_early_v1/summary.json) | {} | `e5fd8bf65bc2edea211e68795a39f075cf67d4756cabfc3fb6f059c3dc376777` |
| [verification.json](../../../../docs/benchmarks/chromaseed_widen_early_v1/verification.json) | {"passed": true} | `bed48664e75da3b095e6c2dd5c3d256a80d867e45a35dd174aeb8e43f1301ecf` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Вариант | Параметры | Модель, КБ | Смешанная | SLR → iPod | iPod → SLR | CPU, мкс |
| --- | --- | --- | --- | --- | --- | --- |
| np | 643 | 2.89 | 5.7705 | 8.2943 | 8.5895 | 6.4 |
| tiny | 1,179 | 5.17 | 5.5107 | 8.4355 | 8.8056 | 25.6 |
| WIDE m31 | 30,915 | 124.12 | 5.5340 | 8.3989 | 8.7331 | 33.9 |
| WE m31 | 30,915 | 124.12 | 5.5486 | 8.3686 | 8.6296 | 33.7 |
| WIDE m61 | 60,611 | 242.90 | 5.5066 | 8.4086 | 8.6773 | 37.2 |
| WE m61 | 60,611 | 242.90 | 5.5239 | 8.3969 | 8.3853 | 36.6 |
| WIDE m111 | 110,979 | 444.37 | 5.4739 | 8.2647 | 8.6865 | 47.4 |
| WE m111 | 110,979 | 444.37 | 5.4778 | 8.3083 | 8.3106 | 48.1 |
| WIDE m832 | 832,259 | 3329.49 | 5.3223 | 8.6801 | 8.3243 | 112.8 |
| WE m832 | 832,259 | 3329.49 | 5.5121 | 8.3811 | 8.2939 | 118.7 |

### Таблица 2

| Роль | Модель | Шаги | Скорость | Ошибка WIDE policy | Ошибка WE policy |
| --- | --- | --- | --- | --- | --- |
| mixed | m111 | 2048 | 3e-05 | 5.4739 | 5.4778 |
| slr_to_ipod | m832 | 2048 | 1e-05 | 8.4086 | 8.3811 |
| ipod_to_slr | m61 | 2048 | 3e-05 | 8.6865 | 8.3853 |

### Таблица 3

| Модель WE | Шаги: смешанная / SLR→iPod / iPod→SLR | Скорости | Полная сборка трёх запусков, с |
| --- | --- | --- | --- |
| m31 | 2048 / 2048 / 2048 | 3e-05 / 3e-05 / 3e-05 | 3.425 / 3.440 / 3.401 |
| m61 | 2048 / 2048 / 2048 | 3e-05 / 3e-05 / 3e-05 | 3.619 / 3.640 / 3.635 |
| m111 | 2048 / 2048 / 2048 | 3e-05 / 3e-05 / 3e-05 | 6.191 / 6.211 / 6.182 |
| m832 | 2048 / 2048 / 2048 | 1e-05 / 1e-05 / 1e-05 | 27.206 / 27.375 / 27.259 |
