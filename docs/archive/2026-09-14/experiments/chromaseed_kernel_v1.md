# chromaseed_kernel_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_kernel_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_kernel_v1/report.md)

SHA-256 отчёта: `dfeb87839c222fd844592e17cda924da1d3df449e9a60b450c74d85d30c8a9fe`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_kernel_model_card.md](../../../../docs/architecture/chromaseed_kernel_model_card.md)
- [chromaseed_kernel_next_decision.md](../../../../docs/research/chromaseed_kernel_next_decision.md)
- [chromaseed_kernel_v1_protocol.md](../../../../docs/research/chromaseed_kernel_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_kernel_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_kernel_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_kernel.py](../modules/scripts__chromaseed_kernel.md)
- [scripts/chromaseed_kernel_audit.py](../modules/scripts__chromaseed_kernel_audit.md)
- [scripts/chromaseed_kernel_bank.py](../modules/scripts__chromaseed_kernel_bank.md)
- [scripts/chromaseed_kernel_replay_timing.py](../modules/scripts__chromaseed_kernel_replay_timing.md)
- [scripts/chromaseed_kernel_report.py](../modules/scripts__chromaseed_kernel_report.md)
- [scripts/chromaseed_kernel_runtime.py](../modules/scripts__chromaseed_kernel_runtime.md)
- [scripts/chromaseed_kernel_train.py](../modules/scripts__chromaseed_kernel_train.md)
- [tests/test_chromaseed_kernel.py](../tests/tests__test_chromaseed_kernel.md)
- [tests/test_chromaseed_kernel_bank.py](../tests/tests__test_chromaseed_kernel_bank.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_kernel_v1/audit.json) | {"passed": true} | `133a62f46abeb52e9159cb4df6e0913fbe547f2ec65f0df8fadec719e6ad75b2` |
| [complete_workflow_runtime.json](../../../../docs/benchmarks/chromaseed_kernel_v1/complete_workflow_runtime.json) | {"scope": "Full train.py run subprocess: interpreter/import/CUDA startup, source/data/legacy hashing, cache load, all 9 inner banks, choices, 3 final banks, selected model export, outer evaluation and persistence. Filesystem cache and driver may be warm. Excludes this wrapper's post-run comparisons, independent audit, plots and runtime benchmarks. This is a repeated timing verification of the same experiment, not 4428 new hypotheses or fresh evidence."} | `a3839fa06bcb85c5f10ae45c80576d375eca39028ca03d67e0d4a41526ba5c8c` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_kernel_v1/runtime.json) | {"scope": "Warm batch-one CPU model calls include normalization, kernel evaluation, projections/blends and actual adaptive control. Excludes file load, image/face feature extraction and app/network time. Adaptive initialization is reported separately. Payload bytes are not peak runtime RAM. Measurements are implementation-specific."} | `66b3238fc722e862af04ad63a15c45a32f6c956393d2bb2b4bb5e20b41a02104` |
| [selections.json](../../../../docs/benchmarks/chromaseed_kernel_v1/selections.json) | {} | `46a092bd5096fd349427e1baaeada61fd1155b68bca89820d8e81e9f6b442419` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_kernel_v1/source_lock.json) | {} | `27adebf6f28ea0e3eecd9006ef43fc94b37e9482bdb73e6d4398703b48e6bb4f` |
| [summary.json](../../../../docs/benchmarks/chromaseed_kernel_v1/summary.json) | {} | `05cf81ab2a3695159e0b6b607622bba58850df370ef226aa1ef42be046b880a5` |
| [verification.json](../../../../docs/benchmarks/chromaseed_kernel_v1/verification.json) | {"passed": true} | `807d9f25fe074085ce70854a26d4416b2370293b1290e679a4d72d8af542b4c9` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Модель / опоры | Смешанные, 6 чел. | SLR → iPod, 16 чел. | iPod → SLR, 8 чел. | Числовые байты |
| --- | --- | --- | --- | --- |
| Полное ядро | 5.3984 | 8.5845 | 8.9125 | 50704–114820 |
| Nyström random / 16 | 6.4421 | 8.0875 | 7.9628 | 2812 |
| Nyström random / 32 | 6.1130 | 8.3131 | 8.7687 | 5308 |
| Nyström random / 64 | 5.9661 | 8.7795 | 8.9210 | 10300 |
| Nyström random / 128 | 5.9244 | 8.6500 | 8.8552 | 20284 |
| Nyström greedy / 16 | 6.6118 | 7.8799 | 10.1415 | 2812 |
| Nyström greedy / 32 | 6.1294 | 8.1435 | 9.6113 | 5308 |
| Nyström greedy / 64 | 5.9787 | 8.8389 | 9.3628 | 10300 |
| Nyström greedy / 128 | 5.4595 | 8.5738 | 8.8269 | 20284 |
| Nyström RPCholesky / 16 | 6.6635 | 8.2602 | 9.2168 | 2812 |
| Nyström RPCholesky / 32 | 6.1052 | 8.2780 | 9.5021 | 5308 |
| Nyström RPCholesky / 64 | 5.9542 | 8.7329 | 9.3612 | 10300 |
| Nyström RPCholesky / 128 | 5.4387 | 8.5970 | 8.7050 | 20284 |
| Проекция RPCholesky / 16 | 7.0673 | 8.4562 | 8.9720 | 2812 |
| Проекция RPCholesky / 32 | 6.1933 | 8.2376 | 9.1980 | 5308 |
| Проекция RPCholesky / 64 | 5.9641 | 8.6956 | 9.5680 | 10300 |
| Проекция RPCholesky / 128 | 5.4365 | 8.5968 | 8.7289 | 20284 |
| Проекция с ранним выходом / 128 | 5.4365 | 8.5968 | 8.7289 | 27508 |
| Nyström64 + guided RBF | 5.8216 | 8.9573 | 8.2285 | 21300 |
| Проекция128 + guided RBF | 5.5741 | 8.5968 | 8.3238 | 20288–31284 |
| Прежний guided_rbf | 5.8013 | 10.0280 | 7.7193 | 10996 |
| Прежний mlp | 5.7856 | 9.4810 | 9.7309 | 10996 |

### Таблица 2

| Роль | Разность ΔE00 | Описательный 95% интервал |
| --- | --- | --- |
| Смешанные камеры (6 человек) | +0.0381 | [+0.0076; +0.0635] |
| SLR → iPod (16 человек) | +0.0123 | [-0.0018; +0.0283] |
| iPod → SLR (8 человек) | -0.1836 | [-0.3768; -0.0037] |

### Таблица 3

| Модель, смешанная роль | Медиана, мкс | p95, мкс |
| --- | --- | --- |
| Полное ядро | 33.50 | 37.32 |
| Nyström RPCholesky | 16.50 | 20.18 |
| Проекция RPCholesky | 16.57 | 16.78 |
| Проекция с ранним выходом | 96.00 | 100.08 |
| Проекция128 + guided RBF | 34.73 | 36.79 |
| Прежний krr (FP32) | 24.50 | 25.02 |
| Прежний guided_rbf (FP32) | 16.13 | 16.50 |

### Таблица 4

| Модель | Полное решение | Медиана fit, мс |
| --- | --- | --- |
| Полное ядро | cpu | 107.40 |
| Полное ядро | cuda | 97.07 |
| Nyström random | учитель не требуется | 66.03 |
| Nyström greedy | учитель не требуется | 66.88 |
| Nyström RPCholesky | учитель не требуется | 67.77 |
| Проекция RPCholesky | cpu | 110.13 |
| Проекция RPCholesky | cuda | 102.87 |
