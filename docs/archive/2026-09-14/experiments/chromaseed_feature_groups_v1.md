# chromaseed_feature_groups_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_feature_groups_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_feature_groups_v1/report.md)

SHA-256 отчёта: `f1782f3b596e34b5a40671f0ca6cdd23b5e50b64aea9ecbd00c7900b496e490d`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_feature_groups_model_card.md](../../../../docs/architecture/chromaseed_feature_groups_model_card.md)
- [chromaseed_feature_groups_count_erratum.md](../../../../docs/research/chromaseed_feature_groups_count_erratum.md)
- [chromaseed_feature_groups_next_decision.md](../../../../docs/research/chromaseed_feature_groups_next_decision.md)
- [chromaseed_feature_groups_v1_protocol.md](../../../../docs/research/chromaseed_feature_groups_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_feature_groups_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_feature_groups_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_feature_groups.py](../modules/scripts__chromaseed_feature_groups.md)
- [scripts/chromaseed_feature_groups_audit.py](../modules/scripts__chromaseed_feature_groups_audit.md)
- [scripts/chromaseed_feature_groups_numpy.py](../modules/scripts__chromaseed_feature_groups_numpy.md)
- [scripts/chromaseed_feature_groups_report.py](../modules/scripts__chromaseed_feature_groups_report.md)
- [scripts/chromaseed_feature_groups_runtime.py](../modules/scripts__chromaseed_feature_groups_runtime.md)
- [scripts/chromaseed_feature_groups_train.py](../modules/scripts__chromaseed_feature_groups_train.md)
- [scripts/chromaseed_feature_groups_verify.py](../modules/scripts__chromaseed_feature_groups_verify.md)
- [tests/test_chromaseed_feature_groups.py](../tests/tests__test_chromaseed_feature_groups.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_feature_groups_v1/audit.json) | {"passed": true} | `0979a00eebe0866cd6768d0f7f0769aae29e7ec5fdcd9681ca54bb5a9471e381` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_feature_groups_v1/runtime.json) | {"scope": "291 actual one-row consumers,20 warmups and3 passes;96 settings x4 complete fits, including warmups. One CPU thread. Fit includes normalizers, weights, exact width, landmarks, gate when active, perceptual metric and readout. X control includes its projection. Excludes process imports/I/O/grid/image decoding and skin extraction. Numeric/cache/archive storage reported separately; cache excludes temporaries, Python/process overhead and caller payload."} | `ffad66a44101cc51965b918a8dc9264ff0045cc52b3b0dba01acb5228e52c4d6` |
| [summary.json](../../../../docs/benchmarks/chromaseed_feature_groups_v1/summary.json) | {} | `e83326dd5a5f69b6642fba599a001d2ac09816d8d426ff922abb08efd1cf2ddc` |
| [verification.json](../../../../docs/benchmarks/chromaseed_feature_groups_v1/verification.json) | {"passed": true} | `459e6554bfba7ce22c63f7377556648632ba07406d20e1d50a42351d8bbf8908` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Вход | Mixed ΔE00 | SLR → iPod | iPod → SLR | Веса mixed, Б | Обучение mixed, мс | Ответ mixed, мкс |
| --- | --- | --- | --- | --- | --- | --- |
| raw36 | 5.272642 | 8.654805 | 8.386889 | 21,973 | 34.72 | 12.3 |
| mean3 | 6.511623 | 6.468986 | 7.901494 | 4,684 | 27.89 | 12.6 |
| median3 | 6.517289 | 6.533613 | 7.642070 | 4,684 | 26.25 | 12.6 |
| central9 | 5.903517 | 6.843411 | 7.725162 | 7,834 | 29.87 | 12.9 |
| mean_std6 | 5.824761 | 8.372741 | 9.124801 | 6,259 | 29.69 | 12.9 |
| quant27 | 5.399296 | 8.146156 | 10.368963 | 17,284 | 31.39 | 13.1 |
| no_corr33 | 5.346335 | 8.440965 | 10.227152 | 20,434 | 31.82 | 13.3 |
| projected16 | 5.206762 | 9.147710 | 9.479015 | 14,181 | 30.66 | 14.0 |

### Таблица 2

| Роль | Семья | Политика | Группа / alpha | Внутренняя ΔE00 | Внешняя ΔE00 | Разница с raw |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | norm_static | quality | raw36 / 0.1 | 4.4554 | 5.4387 | +0.0000 |
| mixed | norm_static | compact | central9 / 0.1 | 4.4982 | 5.8836 | +0.4449 |
| mixed | norm_joint_soft | quality | projected16 / 0.1 | 4.4307 | 5.2488 | -0.0824 |
| mixed | norm_joint_soft | compact | central9 / 0.1 | 4.4612 | 5.8603 | +0.5292 |
| mixed | perceptual_static | quality | raw36 / 0.1 | 4.4485 | 5.3901 | +0.0000 |
| mixed | perceptual_static | compact | central9 / 0.1 | 4.4808 | 5.9199 | +0.5299 |
| mixed | perceptual_joint_soft | quality | central9 / 0.1 | 4.4402 | 5.9035 | +0.6309 |
| mixed | perceptual_joint_soft | compact | central9 / 0.1 | 4.4402 | 5.9035 | +0.6309 |
| slr_to_ipod | norm_static | quality | projected16 / 0.1 | 5.0090 | 9.1419 | +0.5449 |
| slr_to_ipod | norm_static | compact | mean_std6 / 0.1 | 5.1019 | 8.4276 | -0.1694 |
| slr_to_ipod | norm_joint_soft | quality | projected16 / 0.1 | 5.0090 | 9.1419 | +0.5449 |
| slr_to_ipod | norm_joint_soft | compact | mean_std6 / 0.1 | 5.1019 | 8.4276 | -0.1694 |
| slr_to_ipod | perceptual_static | quality | projected16 / 0.1 | 5.1414 | 9.1477 | +0.4929 |
| slr_to_ipod | perceptual_static | compact | no_corr33 / 0.1 | 5.1518 | 8.4410 | -0.2138 |
| slr_to_ipod | perceptual_joint_soft | quality | projected16 / 0.1 | 5.1414 | 9.1477 | +0.4929 |
| slr_to_ipod | perceptual_joint_soft | compact | no_corr33 / 0.1 | 5.1518 | 8.4410 | -0.2138 |
| ipod_to_slr | norm_static | quality | raw36 / 0.1 | 4.3719 | 8.7050 | +0.0000 |
| ipod_to_slr | norm_static | compact | quant27 / 0.1 | 4.4104 | 10.0573 | +1.3523 |
| ipod_to_slr | norm_joint_soft | quality | raw36 / 0.1 | 4.3719 | 8.7050 | +0.0000 |
| ipod_to_slr | norm_joint_soft | compact | quant27 / 0.1 | 4.4104 | 10.0573 | +1.3523 |
| ipod_to_slr | perceptual_static | quality | raw36 / 0.1 | 4.3321 | 8.3869 | +0.0000 |
| ipod_to_slr | perceptual_static | compact | no_corr33 / 0.1 | 4.3731 | 10.2272 | +1.8403 |
| ipod_to_slr | perceptual_joint_soft | quality | raw36 / 0.1 | 4.3321 | 8.3869 | +0.0000 |
| ipod_to_slr | perceptual_joint_soft | compact | no_corr33 / 0.1 | 4.3731 | 10.2272 | +1.8403 |

### Таблица 3

| Роль | Группа | Разница ΔE00 | Описательный диапазон 95% | Людей с улучшением |
| --- | --- | --- | --- | --- |
| mixed | mean3 | +1.2390 | [-0.0197, +3.1151] | 2/6 |
| mixed | median3 | +1.2446 | [+0.1259, +2.9490] | 1/6 |
| mixed | central9 | +0.6309 | [+0.0454, +1.3777] | 2/6 |
| slr_to_ipod | mean3 | -2.1858 | [-3.3093, -0.8075] | 14/16 |
| slr_to_ipod | median3 | -2.1212 | [-3.1517, -0.8573] | 14/16 |
| slr_to_ipod | central9 | -1.8114 | [-2.4899, -1.0866] | 15/16 |
| ipod_to_slr | mean3 | -0.4854 | [-1.6277, +0.5224] | 4/8 |
| ipod_to_slr | median3 | -0.7448 | [-2.8799, +0.8215] | 4/8 |
| ipod_to_slr | central9 | -0.6617 | [-2.5927, +0.8308] | 4/8 |
