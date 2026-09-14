# chromaseed_perceptual_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_perceptual_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_perceptual_v1/report.md)

SHA-256 отчёта: `a643ad13cefe228708291f0d7c39d6cbacc5ed95d15fbf21c2d746f92230becd`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_perceptual_model_card.md](../../../../docs/architecture/chromaseed_perceptual_model_card.md)
- [chromaseed_perceptual_next_decision.md](../../../../docs/research/chromaseed_perceptual_next_decision.md)
- [chromaseed_perceptual_v1_protocol.md](../../../../docs/research/chromaseed_perceptual_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_perceptual_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_perceptual_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_perceptual.py](../modules/scripts__chromaseed_perceptual.md)
- [scripts/chromaseed_perceptual_audit.py](../modules/scripts__chromaseed_perceptual_audit.md)
- [scripts/chromaseed_perceptual_reference.py](../modules/scripts__chromaseed_perceptual_reference.md)
- [scripts/chromaseed_perceptual_report.py](../modules/scripts__chromaseed_perceptual_report.md)
- [scripts/chromaseed_perceptual_runtime.py](../modules/scripts__chromaseed_perceptual_runtime.md)
- [scripts/chromaseed_perceptual_train.py](../modules/scripts__chromaseed_perceptual_train.md)
- [tests/test_chromaseed_perceptual.py](../tests/tests__test_chromaseed_perceptual.md)
- [tests/test_chromaseed_perceptual_protocol.py](../tests/tests__test_chromaseed_perceptual_protocol.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/audit.json) | {"passed": true, "seconds": 8.571513400005642} | `bc420ad7367b35b39b54b2cd738d169b16279168cec695df2ef63b5edb3a7555` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/runtime.json) | {} | `11e3f14bf510add7cd7a51a619e8bad618bd467a4b4731bf79109dc1e442c93c` |
| [selections.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/selections.json) | {} | `7f526a4f17e2afb53f4a81bd436b958bac58d703ee368afa3bdf76a9b95efcfb` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/source_lock.json) | {} | `9ce75fddcfc3241b271318da51eead680d78a9e51877b346aed64716b5550e92` |
| [summary.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/summary.json) | {} | `048f56900588229e0735b1d243cd61577662c5988f8584758c8a70878225df89` |
| [verification.json](../../../../docs/benchmarks/chromaseed_perceptual_v1/verification.json) | {"passed": true} | `4b7124a59a3d45cadd39b061fce4c0d3e01f56e1a209f538b918c1f0b99a3794` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Метод | mixed,6 чел. | SLR→iPod,16 чел. | iPod→SLR,8 чел. |
| --- | --- | --- | --- |
| Нормированная ошибка | 5.4387 | 8.5970 | 8.7050 |
| Общие цветовые веса | 5.3901 | 8.6548 | 8.3869 |
| Веса каждого оттенка | 5.3842 | 8.7874 | 8.4135 |
| Коррекция с фиксированной геометрией | 5.4387 | 8.5970 | 8.7050 |
| Коррекция с обновлением геометрии | 5.4387 | 8.5970 | 8.7050 |

### Таблица 2

| Роль / метод | Fit, мс | Ответ, мкс | Выбранные шаги |
| --- | --- | --- | --- |
| mixed / norm_mse | 16.39 | 16.57 | 0 |
| mixed / constant_de2 | 22.58 | 16.47 | 1 |
| mixed / local_de2 | 22.72 | 16.53 | 1 |
| mixed / local_irls | 16.27 | 16.50 | 0 |
| mixed / midpoint_irls | 16.91 | 16.60 | 0 |
| slr_to_ipod / norm_mse | 8.91 | 16.47 | 0 |
| slr_to_ipod / constant_de2 | 12.39 | 16.40 | 1 |
| slr_to_ipod / local_de2 | 12.30 | 16.40 | 1 |
| slr_to_ipod / local_irls | 8.53 | 16.43 | 0 |
| slr_to_ipod / midpoint_irls | 8.48 | 16.40 | 0 |
| ipod_to_slr / norm_mse | 14.82 | 16.37 | 0 |
| ipod_to_slr / constant_de2 | 20.50 | 16.43 | 1 |
| ipod_to_slr / local_de2 | 20.50 | 16.30 | 1 |
| ipod_to_slr / local_irls | 14.65 | 16.47 | 0 |
| ipod_to_slr / midpoint_irls | 14.69 | 16.37 | 0 |

### Таблица 3

| Роль / метод | Fit, мс | Реально выполнено коррекций |
| --- | --- | --- |
| mixed / local_irls | 82.90 | 16 |
| mixed / midpoint_irls | 46.96 | 4 |
| slr_to_ipod / local_irls | 51.69 | 16 |
| slr_to_ipod / midpoint_irls | 73.75 | 16 |
| ipod_to_slr / local_irls | 75.71 | 16 |
| ipod_to_slr / midpoint_irls | 112.78 | 16 |
