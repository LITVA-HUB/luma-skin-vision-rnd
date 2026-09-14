# chromaseed_neural_blocks_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Эквивалентность не принята; отрицательная диагностика сохранена.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_neural_blocks_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/report.md)

SHA-256 отчёта: `2d109f9dea908a26da8abd17ab41cda7c32ab2013e90ed02d28c48faee2e7785`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_neural_blocks_model_card.md](../../../../docs/architecture/chromaseed_neural_blocks_model_card.md)
- [chromaseed_neural_blocks_failure_diagnostic.md](../../../../docs/research/chromaseed_neural_blocks_failure_diagnostic.md)
- [chromaseed_neural_blocks_next_decision.md](../../../../docs/research/chromaseed_neural_blocks_next_decision.md)
- [chromaseed_neural_blocks_v1_protocol.md](../../../../docs/research/chromaseed_neural_blocks_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_neural_blocks_audit.py](../modules/scripts__chromaseed_neural_blocks_audit.md)
- [scripts/chromaseed_neural_blocks_diagnose.py](../modules/scripts__chromaseed_neural_blocks_diagnose.md)
- [scripts/chromaseed_neural_blocks_fit.py](../modules/scripts__chromaseed_neural_blocks_fit.md)
- [scripts/chromaseed_neural_blocks_probe.py](../modules/scripts__chromaseed_neural_blocks_probe.md)
- [scripts/chromaseed_neural_blocks_report.py](../modules/scripts__chromaseed_neural_blocks_report.md)
- [scripts/chromaseed_neural_blocks_run.py](../modules/scripts__chromaseed_neural_blocks_run.md)
- [scripts/chromaseed_neural_blocks_runtime.py](../modules/scripts__chromaseed_neural_blocks_runtime.md)
- [tests/test_chromaseed_neural_blocks.py](../tests/tests__test_chromaseed_neural_blocks.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/audit.json) | {"passed": true, "equivalence_accepted": false, "primary_exit_code": 1, "seconds": 66.0108176000067} | `d6a1aa259f1239d589d01cf97315a59ec8f87c4afb4526c90bcda38815140c2b` |
| [first_step_probe.json](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/first_step_probe.json) | {} | `2a28ee256d92c577411f855ba7bb4010b40283488e8317731e450c10a5fad249` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/runtime.json) | {"seconds": 28.311318599968217, "equivalence_accepted": false} | `8b83aa31fba5623eb8e69f22136cb3f7fba358eb0a174b256e5a03905f0bf099` |
| [summary.json](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/summary.json) | {"equivalence_accepted": false, "primary_exit_code": 1} | `5e5b510b6c963db52a705bb0516021e3960920b1644d7bd9e3cd0d0a9143d393` |
| [verification.json](../../../../docs/benchmarks/chromaseed_neural_blocks_v1/verification.json) | {"passed": true, "equivalence_accepted": false, "primary_exit_code": 1} | `671fb2cd20d5e914bbddf3f5f9abafd05a73ea6ecee57c244e769a1cd47f8f2c` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Семейство | Шаги | Побитово /18 | В допуске /18 | Максимальный разрыв весов | Максимальный разрыв native Lab |
| --- | --- | --- | --- | --- | --- |
| local2 | 512 | 9 | 18 | 2.08616e-07 | 4.59164e-06 |
| local2 | 2048 | 9 | 18 | 6.85453e-07 | 1.14126e-05 |
| local2 | 8192 | 9 | 11 | 0.121841 | 1.81464 |
| local4 | 512 | 12 | 18 | 1.78814e-07 | 3.77337e-06 |
| local4 | 2048 | 12 | 16 | 0.0107988 | 0.21541 |
| local4 | 8192 | 12 | 14 | 0.200071 | 3.35671 |
| blind4 | 512 | 0 | 18 | 8.9407e-08 | 3.2787e-06 |
| blind4 | 2048 | 0 | 18 | 7.89762e-07 | 1.12609e-05 |
| blind4 | 8192 | 0 | 12 | 0.0393863 | 0.567923 |

### Таблица 2

| Сценарий | Семейство / J | Полное обучение, мс | Только блоки, мс | Ускорение |
| --- | --- | --- | --- | --- |
| mixed | local2 / 1 | 101.27 | 103.62 | 0.977× |
| mixed | local2 / 2 | 102.44 | 102.83 | 0.996× |
| mixed | local4 / 1 | 100.16 | 99.16 | 1.010× |
| mixed | local4 / 4 | 97.97 | 99.78 | 0.982× |
| mixed | blind4 / 2 | 273.03 | 255.37 | 1.069× |
| mixed | blind4 / 4 | 271.37 | 255.35 | 1.063× |
| slr_to_ipod | local2 / 1 | 284.38 | 266.30 | 1.068× |
| slr_to_ipod | local2 / 2 | 281.31 | 288.15 | 0.976× |
| slr_to_ipod | local4 / 3 | 289.63 | 282.43 | 1.025× |
| slr_to_ipod | local4 / 4 | 283.76 | 288.26 | 0.984× |
| slr_to_ipod | blind4 / 3 | 269.11 | 250.66 | 1.074× |
| slr_to_ipod | blind4 / 4 | 270.48 | 251.09 | 1.077× |
| ipod_to_slr | local2 / 1 | 285.06 | 268.35 | 1.062× |
| ipod_to_slr | local2 / 2 | 283.46 | 285.17 | 0.994× |
| ipod_to_slr | local4 / 2 | 286.84 | 285.71 | 1.004× |
| ipod_to_slr | local4 / 4 | 286.83 | 290.76 | 0.986× |
| ipod_to_slr | blind4 / 2 | 275.31 | 254.07 | 1.084× |
| ipod_to_slr | blind4 / 4 | 272.50 | 254.62 | 1.070× |
