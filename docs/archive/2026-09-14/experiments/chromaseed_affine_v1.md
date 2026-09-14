# chromaseed_affine_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_affine_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_affine_v1/report.md)

SHA-256 отчёта: `cdbb1fc7e73140a1b34fc9b5d60c29cc82eb82ade03f524ad4b2df99b5cc61bc`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_affine_model_card.md](../../../../docs/architecture/chromaseed_affine_model_card.md)
- [chromaseed_affine_next_decision.md](../../../../docs/research/chromaseed_affine_next_decision.md)
- [chromaseed_affine_v1_protocol.md](../../../../docs/research/chromaseed_affine_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_affine_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_affine_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_affine.py](../modules/scripts__chromaseed_affine.md)
- [scripts/chromaseed_affine_audit.py](../modules/scripts__chromaseed_affine_audit.md)
- [scripts/chromaseed_affine_reference.py](../modules/scripts__chromaseed_affine_reference.md)
- [scripts/chromaseed_affine_report.py](../modules/scripts__chromaseed_affine_report.md)
- [scripts/chromaseed_affine_runtime.py](../modules/scripts__chromaseed_affine_runtime.md)
- [scripts/chromaseed_affine_train.py](../modules/scripts__chromaseed_affine_train.md)
- [scripts/chromaseed_affine_verify.py](../modules/scripts__chromaseed_affine_verify.md)
- [tests/test_chromaseed_affine.py](../tests/tests__test_chromaseed_affine.md)
- [tests/test_chromaseed_affine_reference.py](../tests/tests__test_chromaseed_affine_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_affine_v1/audit.json) | {"passed": true, "scope": "All banks/OOF/direct predictions, frozen policies and all final standalone outputs/metrics. Independent QR/SVD refits cover72 selected new instances plus12 positive probes, not every grid fit. Shared frozen splits/CIEDE2000; normalizers checked on original fit rows, basis pinned to audited G. Bootstrap is descriptive for fixed historically reused predictions, not fresh accuracy confidence."} | `023672302eb5b1d89d2459764f9e3f0adf778b03422dbb2b96468fab92996132` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_affine_v1/runtime.json) | {"scope": "111 actual standalone batch-one consumers,20 warmups/3 passes.112 exact full fits including warmups, weights/normalization/bandwidth/landmarks/gate/augmentation/readout. Excludes imports, I/O, search and image feature preparation. CPU one thread, numeric payload and cached arrays exclude library/process memory. Constant control uses a checked NumPy-only broadcast consumer."} | `98d8ec0c6617e4e6fd825fdc2bfaf5a8d7e40d71d6cf366568da19b59f314377` |
| [summary.json](../../../../docs/benchmarks/chromaseed_affine_v1/summary.json) | {} | `05aaf6ef42c5abb533092dec6d46c4d49e889b112c01cecd824fde4fa1cde76a` |
| [verification.json](../../../../docs/benchmarks/chromaseed_affine_v1/verification.json) | {"passed": true} | `91a348711bbe95dba917a6b31c34e5c78d94c6113737a33f0a93ba4fcc5ec646` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Семейство / политика | Mixed | SLR → iPod | iPod → SLR |
| --- | --- | --- | --- |
| norm_static / clean | 5.438652 | 8.597000 | 8.705018 |
| norm_static / guarded | 5.446757 | 8.671774 | 8.700098 |
| norm_joint_soft / clean | 5.331149 | 8.597000 | 8.705018 |
| norm_joint_soft / guarded | 5.338888 | 8.671774 | 8.700098 |
| perceptual_static / clean | 5.390066 | 8.654805 | 8.386889 |
| perceptual_static / guarded | 5.397213 | 8.743031 | 8.390257 |
| perceptual_joint_soft / clean | 5.272642 | 8.654805 | 8.386889 |
| perceptual_joint_soft / guarded | 5.279610 | 8.743031 | 8.390257 |
| g_norm_base / reference | 5.438652 | 8.597000 | 8.705018 |
| g_norm_soft / reference | 5.349017 | 8.597000 | 8.705018 |
| g_perceptual_base / reference | 5.390066 | 8.654805 | 8.386889 |
| g_perceptual_soft / reference | 5.295118 | 8.654805 | 8.386889 |
| constant / reference | 13.656917 | 12.267125 | 10.599715 |

### Таблица 2

| Совместная перцепционная модель | Mixed clean | Mixed стресс4/255 | SLR→iPod clean | SLR→iPod стресс4/255 |
| --- | --- | --- | --- | --- |
| clean | 5.272642 | 6.278089 | 8.654805 | 9.234482 |
| guarded | 5.279610 | 6.232160 | 8.743031 | 9.309668 |

### Таблица 3

| Mixed, seed17 | Полное обучение, мс | Ответ, мкс¹ | Веса, байт |
| --- | --- | --- | --- |
| norm_static / clean | 17.865 | 9.3 | 20284 |
| norm_static / guarded | 43.162 | 9.3 | 20284 |
| norm_joint_soft / clean | 32.018 | 11.9 | 21973 |
| norm_joint_soft / guarded | 83.152 | 11.8 | 21973 |
| perceptual_static / clean | 20.363 | 9.3 | 20284 |
| perceptual_static / guarded | 46.034 | 9.3 | 20284 |
| perceptual_joint_soft / clean | 34.600 | 11.8 | 21973 |
| perceptual_joint_soft / guarded | 88.481 | 11.9 | 21973 |
