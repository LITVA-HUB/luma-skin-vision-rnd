# chromaseed_weak_ridge_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_weak_ridge_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/report.md)

SHA-256 отчёта: `951f3016e11db75dc2b5674b9ffaae93cd329698ce9cd3458a0ea7d382371c88`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_weak_ridge_model_card.md](../../../../docs/architecture/chromaseed_weak_ridge_model_card.md)
- [chromaseed_weak_ridge_next_decision.md](../../../../docs/research/chromaseed_weak_ridge_next_decision.md)
- [chromaseed_weak_ridge_v1_protocol.md](../../../../docs/research/chromaseed_weak_ridge_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_weak_ridge.py](../modules/scripts__chromaseed_weak_ridge.md)
- [scripts/chromaseed_weak_ridge_audit.py](../modules/scripts__chromaseed_weak_ridge_audit.md)
- [scripts/chromaseed_weak_ridge_report.py](../modules/scripts__chromaseed_weak_ridge_report.md)
- [scripts/chromaseed_weak_ridge_runtime.py](../modules/scripts__chromaseed_weak_ridge_runtime.md)
- [scripts/chromaseed_weak_ridge_train.py](../modules/scripts__chromaseed_weak_ridge_train.md)
- [tests/test_chromaseed_weak_ridge.py](../tests/tests__test_chromaseed_weak_ridge.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/audit.json) | {"passed": true, "seconds": 19.861407899996266} | `d1aa61fb4cc5fc759face8bf1669dabfa9472b9bade8d813a4f872fa57fdaa04` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/runtime.json) | {} | `73af1593483785e0732c79782fb3517d56304952c808433589f548d22e5a2f3e` |
| [selections.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/selections.json) | {} | `c455b980d26f39ec6f5230819aaf06eead19332b9ff44d455029827fbf13cef4` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/source_lock.json) | {} | `c16e607e05cc9750ccc3275c16d482c2fb863c693d920ab366b0e0200c1d33e8` |
| [summary.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/summary.json) | {} | `69d2555ea1182703c3ffe40eb2a65fc1d6ce2d7b41c2c4bcf576daf1f8dc4f52` |
| [verification.json](../../../../docs/benchmarks/chromaseed_weak_ridge_v1/verification.json) | {"passed": true} | `1a3aea3400503e0e10c69821c9f4aa82021fc52274df06666af2925ce1b46080` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Метод | mixed | SLR→iPod | iPod→SLR |
| --- | --- | --- | --- |
| norm_mse | 5.5447 | 9.5273 | 8.9189 |
| constant_de2 | 5.4995 | 9.5260 | 8.6863 |
| local_de2 | 5.4844 | 9.6353 | 8.7300 |
| local_irls | 5.5616 | 9.4134 | 8.5290 |
| midpoint_irls | 5.5666 | 9.5237 | 8.5144 |

### Таблица 2

| Роль / метод | alpha | Ширина | Запрошено шагов | Выполнено решений | Fit, мс | Ответ, мкс |
| --- | --- | --- | --- | --- | --- | --- |
| mixed / norm_mse | 0.01 | 2 | 0 | 0 | 16.22 | 16.43 |
| mixed / constant_de2 | 0.01 | 2 | 1 | 1 | 22.21 | 16.43 |
| mixed / local_de2 | 0.01 | 2 | 1 | 1 | 22.37 | 16.40 |
| mixed / local_irls | 0.01 | 2 | 1 | 1 | 22.94 | 16.43 |
| mixed / midpoint_irls | 0.01 | 2 | 1 | 1 | 25.89 | 16.43 |
| slr_to_ipod / norm_mse | 0.01 | 2 | 0 | 0 | 8.37 | 16.37 |
| slr_to_ipod / constant_de2 | 0.01 | 2 | 1 | 1 | 12.14 | 16.37 |
| slr_to_ipod / local_de2 | 0.01 | 2 | 1 | 1 | 12.15 | 16.37 |
| slr_to_ipod / local_irls | 0.001 | 2 | 4 | 4 | 20.05 | 16.40 |
| slr_to_ipod / midpoint_irls | 0.003 | 2 | 16 | 5 | 30.04 | 16.53 |
| ipod_to_slr / norm_mse | 0.03 | 1 | 0 | 0 | 14.56 | 16.50 |
| ipod_to_slr / constant_de2 | 0.03 | 1 | 1 | 1 | 20.30 | 16.50 |
| ipod_to_slr / local_de2 | 0.03 | 1 | 1 | 1 | 20.37 | 16.43 |
| ipod_to_slr / local_irls | 0.03 | 1 | 1 | 1 | 20.47 | 16.50 |
| ipod_to_slr / midpoint_irls | 0.03 | 1 | 16 | 16 | 111.33 | 16.47 |
