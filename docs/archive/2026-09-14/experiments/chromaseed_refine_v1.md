# chromaseed_refine_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_refine_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_refine_v1/report.md)

SHA-256 отчёта: `26fe17234b126821072da110497cb1250bbc216fd1f0f983d62eb73f8d41daf7`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_refine_model_card.md](../../../../docs/architecture/chromaseed_refine_model_card.md)
- [chromaseed_refine_next_decision.md](../../../../docs/research/chromaseed_refine_next_decision.md)
- [chromaseed_refine_precision_protocol.md](../../../../docs/research/chromaseed_refine_precision_protocol.md)
- [chromaseed_refine_v1_protocol.md](../../../../docs/research/chromaseed_refine_v1_protocol.md)
- [precision.md](../../../../docs/benchmarks/chromaseed_refine_v1/precision.md)
- [report.md](../../../../docs/benchmarks/chromaseed_refine_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_refine_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_refine.py](../modules/scripts__chromaseed_refine.md)
- [scripts/chromaseed_refine_audit.py](../modules/scripts__chromaseed_refine_audit.md)
- [scripts/chromaseed_refine_numpy.py](../modules/scripts__chromaseed_refine_numpy.md)
- [scripts/chromaseed_refine_precision.py](../modules/scripts__chromaseed_refine_precision.md)
- [scripts/chromaseed_refine_precision_diagnosis.py](../modules/scripts__chromaseed_refine_precision_diagnosis.md)
- [scripts/chromaseed_refine_report.py](../modules/scripts__chromaseed_refine_report.md)
- [scripts/chromaseed_refine_train.py](../modules/scripts__chromaseed_refine_train.md)
- [tests/test_chromaseed_refine.py](../tests/tests__test_chromaseed_refine.md)
- [tests/test_chromaseed_refine_numpy.py](../tests/tests__test_chromaseed_refine_numpy.md)
- [tests/test_chromaseed_refine_precision.py](../tests/tests__test_chromaseed_refine_precision.md)
- [tests/test_chromaseed_refine_training.py](../tests/tests__test_chromaseed_refine_training.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_refine_v1/audit.json) | {"passed": true} | `16217bfddfc049d0ff17ea6ee2228718ecddb1d53fb5b88ade7f86aad8fe3fbb` |
| [precision.json](../../../../docs/benchmarks/chromaseed_refine_v1/precision.json) | {} | `bc4184d2c39a2cb44b6c6d973c3ccfb9c313e8ad4a71af401309d696a04bf17d` |
| [precision_exit_diagnosis.json](../../../../docs/benchmarks/chromaseed_refine_v1/precision_exit_diagnosis.json) | {} | `fbe6260d242bdfb2fe5d4568fd23746da527fbb292f132d83207c4f447d7521e` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_refine_v1/runtime.json) | {"scope": "NumPy float32 batch-one CPU; includes normalization, anchor, gates and real loop exit; excludes image/feature extraction and face detection"} | `9b907ee82e42a4fa2f0b107fd6d2b4c1a2b1a9d5b1894e395b78f7156bb5c684` |
| [selections.json](../../../../docs/benchmarks/chromaseed_refine_v1/selections.json) | {} | `86e7a96c30391fc5e9957d056bcd8ff65898819f14550e48875b2f5426ea22a6` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_refine_v1/source_lock.json) | {} | `ebf9c90bc32142ede9a0a40d1d77cb4037b327435faf81444f140e8ad66fc7d5` |
| [summary.json](../../../../docs/benchmarks/chromaseed_refine_v1/summary.json) | {} | `9e67d93d6b6dde7efb8d7c272b491f74429cf7ba7792051eef3ac8ea001de7e0` |
| [verification.json](../../../../docs/benchmarks/chromaseed_refine_v1/verification.json) | {} | `274cd01d70def6ea9cd3d42c379098cf52bd32b22dd6d9826036d91c9ff4cc93` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Модель | Смешанные камеры, 6 человек | SLR → iPod, 16 человек | iPod → SLR, 8 человек | FP32, байт |
| --- | --- | --- | --- | --- |
| Statistics MLP | 5.647 | 11.515 | 9.008 | 71,572 |
| Patch MLP | 5.472 | 11.214 | 8.830 | 67,828 |
| Refine all 64 | 5.421 | 10.443 | 9.162 | 59,316 |
| Refine top 16 | 5.396 | 10.353 | 9.941 | 59,316 |
| Refine dynamic | 5.355 | 9.909 | 9.812 | 59,316 |
| Предыдущий krr | 5.398 | 8.584 | 8.913 | 114,820* |
| Предыдущий guided_rbf | 5.801 | 10.028 | 7.719 | 10,996* |
| Предыдущий random_rbf | 5.752 | 10.364 | 8.062 | 10,996* |
| Предыдущий mlp | 5.786 | 9.481 | 9.731 | 10,996* |

### Таблица 2

| Протокол | Модель | Шаги / LR | ΔE, 4 прохода или статическая | ΔE, выбранный выход | Проходов | CPU, мкс, фикс. → адапт. | Обучение 3 моделей, с |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | Statistics MLP | 512 / 0.001 | 5.647 | 5.647 | 1.00 | 18.7 → 18.8 | 0.18 |
| mixed | Patch MLP | 512 / 0.001 | 5.472 | 5.472 | 1.00 | 42.4 → 42.4 | 0.52 |
| mixed | Refine all 64 | 512 / 0.001 | 5.367 | 5.421 | 3.12 | 144.1 → 149.0 | 1.20 |
| mixed | Refine top 16 | 512 / 0.001 | 5.323 | 5.396 | 3.11 | 150.2 → 153.5 | 1.22 |
| mixed | Refine dynamic | 512 / 0.001 | 5.360 | 5.355 | 3.73 | 152.1 → 157.3 | 1.26 |
| slr_to_ipod | Statistics MLP | 512 / 0.0003 | 11.515 | 11.515 | 1.00 | 18.8 → 18.7 | 0.17 |
| slr_to_ipod | Patch MLP | 512 / 0.0003 | 11.214 | 11.214 | 1.00 | 42.3 → 42.2 | 0.51 |
| slr_to_ipod | Refine all 64 | 512 / 0.001 | 10.225 | 10.443 | 2.98 | 144.6 → 130.1 | 1.21 |
| slr_to_ipod | Refine top 16 | 512 / 0.003 | 10.103 | 10.353 | 2.97 | 149.0 → 134.1 | 1.22 |
| slr_to_ipod | Refine dynamic | 512 / 0.001 | 9.885 | 9.909 | 3.96 | 152.8 → 159.0 | 1.25 |
| ipod_to_slr | Statistics MLP | 512 / 0.001 | 9.008 | 9.008 | 1.00 | 19.0 → 19.0 | 0.16 |
| ipod_to_slr | Patch MLP | 512 / 0.001 | 8.830 | 8.830 | 1.00 | 42.3 → 42.3 | 0.52 |
| ipod_to_slr | Refine all 64 | 512 / 0.001 | 9.266 | 9.162 | 3.15 | 144.3 → 149.0 | 1.21 |
| ipod_to_slr | Refine top 16 | 512 / 0.0003 | 9.994 | 9.941 | 3.66 | 148.8 → 155.0 | 1.22 |
| ipod_to_slr | Refine dynamic | 512 / 0.001 | 9.870 | 9.812 | 3.24 | 153.7 → 150.9 | 1.25 |

### Таблица 3

| Протокол | Seed | Число связей, min–max | Доля проходов только с 4 | Разных чисел связей по входам, проходы 1/2/3/4 |
| --- | --- | --- | --- | --- |
| mixed | 17 | 4–64 | 51.0% | 51 / 48 / 49 / 50 |
| mixed | 29 | 4–64 | 60.3% | 50 / 44 / 46 / 44 |
| mixed | 43 | 4–64 | 50.3% | 47 / 42 / 40 / 42 |
| slr_to_ipod | 17 | 4–64 | 40.7% | 60 / 60 / 59 / 60 |
| slr_to_ipod | 29 | 4–64 | 46.3% | 61 / 60 / 59 / 60 |
| slr_to_ipod | 43 | 4–64 | 71.2% | 56 / 55 / 44 / 45 |
| ipod_to_slr | 17 | 4–64 | 45.3% | 20 / 28 / 26 / 26 |
| ipod_to_slr | 29 | 4–64 | 57.3% | 33 / 41 / 41 / 43 |
| ipod_to_slr | 43 | 4–64 | 49.3% | 47 / 47 / 48 / 37 |

### Таблица 4

| Протокол | A − B | Δ среднего, меньше 0 лучше A | Описательный 95% интервал | Людей лучше A |
| --- | --- | --- | --- | --- |
| mixed | patch_mlp − stats_mlp | -0.175 | [-0.445, 0.064] | 4/6 |
| mixed | recur_soft − patch_mlp | -0.051 | [-0.183, 0.127] | 5/6 |
| mixed | recur_dynamic − patch_mlp | -0.117 | [-0.242, 0.011] | 5/6 |
| mixed | recur_dynamic − recur_soft | -0.066 | [-0.163, 0.029] | 3/6 |
| mixed | recur_dynamic − recur_top16 | -0.041 | [-0.127, 0.063] | 4/6 |
| slr_to_ipod | patch_mlp − stats_mlp | -0.301 | [-0.447, -0.156] | 12/16 |
| slr_to_ipod | recur_soft − patch_mlp | -0.771 | [-1.309, -0.210] | 10/16 |
| slr_to_ipod | recur_dynamic − patch_mlp | -1.305 | [-1.909, -0.704] | 13/16 |
| slr_to_ipod | recur_dynamic − recur_soft | -0.534 | [-0.671, -0.365] | 15/16 |
| slr_to_ipod | recur_dynamic − recur_top16 | -0.444 | [-0.590, -0.270] | 15/16 |
| ipod_to_slr | patch_mlp − stats_mlp | -0.179 | [-0.368, 0.009] | 7/8 |
| ipod_to_slr | recur_soft − patch_mlp | 0.333 | [0.054, 0.613] | 1/8 |
| ipod_to_slr | recur_dynamic − patch_mlp | 0.982 | [0.451, 1.572] | 1/8 |
| ipod_to_slr | recur_dynamic − recur_soft | 0.650 | [0.303, 1.053] | 0/8 |
| ipod_to_slr | recur_dynamic − recur_top16 | -0.129 | [-0.399, 0.136] | 4/8 |
