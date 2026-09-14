# chromaseed_projection_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_projection_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_projection_v1/report.md)

SHA-256 отчёта: `29ce674e5bcb89eddea47012f424cd54d9e83b468e4f421ddca6f398d2628d64`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_projection_model_card.md](../../../../docs/architecture/chromaseed_projection_model_card.md)
- [chromaseed_projection_next_decision.md](../../../../docs/research/chromaseed_projection_next_decision.md)
- [chromaseed_projection_v1_protocol.md](../../../../docs/research/chromaseed_projection_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_projection_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_projection_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_projection.py](../modules/scripts__chromaseed_projection.md)
- [scripts/chromaseed_projection_audit.py](../modules/scripts__chromaseed_projection_audit.md)
- [scripts/chromaseed_projection_numpy.py](../modules/scripts__chromaseed_projection_numpy.md)
- [scripts/chromaseed_projection_reference.py](../modules/scripts__chromaseed_projection_reference.md)
- [scripts/chromaseed_projection_report.py](../modules/scripts__chromaseed_projection_report.md)
- [scripts/chromaseed_projection_runtime.py](../modules/scripts__chromaseed_projection_runtime.md)
- [scripts/chromaseed_projection_train.py](../modules/scripts__chromaseed_projection_train.md)
- [scripts/chromaseed_projection_verify.py](../modules/scripts__chromaseed_projection_verify.md)
- [tests/test_chromaseed_projection.py](../tests/tests__test_chromaseed_projection.md)
- [tests/test_chromaseed_projection_reference.py](../tests/tests__test_chromaseed_projection_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_projection_v1/audit.json) | {"passed": true, "scope": "Independent weighted-design SVD covariance/projection, exhaustive widths and dense pivot paths, all saved-payload OOF and standalone final outputs/metrics/policies.72 selected and12 positive refits recompute projection/width/landmarks/gate/color metric via SVD/dense/QR; not every grid coefficient refitted. Frozen split/CIEDE2000 reused. Bootstrap is descriptive for fixed historical predictions."} | `2872d0e2455fa602aad8ef559f055c4f4bc2246fa939494ecfea8f055068d3de` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_projection_v1/runtime.json) | {"scope": "Actual111 batch-one NumPy consumers and112 exact full fits including warmups. Full fit includes original weights/moments/covariance/projection/exact width/landmarks/gate/readout. No imports/I/O/search/image preparation. Cached arrays exclude process, transient and retained caller objects. CPU one thread."} | `7620b044dabfcc85f03ebe0c02cdb9b4befda976078fc344dc150de8bd6625cb` |
| [summary.json](../../../../docs/benchmarks/chromaseed_projection_v1/summary.json) | {} | `2ccb9553db67758aed546e167390b78742ed8b819bfc910322f71387127d9e43` |
| [verification.json](../../../../docs/benchmarks/chromaseed_projection_v1/verification.json) | {"passed": true} | `537014c2d803f27bb9a72f93d260c2b0d1a830a9732fc4037bfe3dc202a98153` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Семейство / политика | Mixed | SLR→iPod | iPod→SLR |
| --- | --- | --- | --- |
| norm_static / raw A clean | 5.438652 | 8.597000 | 8.705018 |
| norm_static / quality | 5.452487 | 9.136583 | 8.906827 |
| norm_static / compact | 5.452487 | 10.053216 | 8.863696 |
| norm_joint_soft / raw A clean | 5.331149 | 8.597000 | 8.705018 |
| norm_joint_soft / quality | 5.249471 | 9.136583 | 8.906827 |
| norm_joint_soft / compact | 5.248772 | 10.053216 | 8.863696 |
| perceptual_static / raw A clean | 5.390066 | 8.654805 | 8.386889 |
| perceptual_static / quality | 5.390065 | 9.173536 | 8.770553 |
| perceptual_static / compact | 5.408278 | 11.029335 | 8.770553 |
| perceptual_joint_soft / raw A clean | 5.272642 | 8.654805 | 8.386889 |
| perceptual_joint_soft / quality | 5.223838 | 9.173536 | 8.770553 |
| perceptual_joint_soft / compact | 5.206762 | 11.029335 | 8.770553 |
| g_norm_soft / reference | 5.349017 | 8.597000 | 8.705018 |
| g_perceptual_soft / reference | 5.295118 | 8.654805 | 8.386889 |
| a_norm_joint_guarded / reference | 5.338888 | 8.671774 | 8.700098 |
| a_perceptual_joint_guarded / reference | 5.279610 | 8.743031 | 8.390257 |
| constant / reference | 13.656917 | 12.267125 | 10.599715 |

### Таблица 2

| Совместная перцепционная модель | Проекция | Веса, байт | Обычная ΔE00 | Стресс4/255 |
| --- | --- | --- | --- | --- |
| mixed / quality | d36_t05 | 27301 | 5.223838 | 6.296910 |
| mixed / compact | d16_t05 | 14181 | 5.206762 | 6.271458 |
| slr_to_ipod / quality | d36_t05 | 25612 | 9.173536 | 9.942061 |
| slr_to_ipod / compact | d8_t0 | 7244 | 11.029335 | 12.341334 |
| ipod_to_slr / quality | d8_t1 | 7244 | 8.770553 | 9.632104 |
| ipod_to_slr / compact | d8_t1 | 7244 | 8.770553 | 9.632104 |

### Таблица 3

| Mixed | Проекция | Полное обучение, мс¹ | Ответ, мкс² | Массивы исполнителя, байт |
| --- | --- | --- | --- | --- |
| norm_static/quality | d16_t1 | 13.991 | 11.0 | 25688 |
| norm_static/compact | d16_t1 | 14.745 | 11.0 | 25688 |
| norm_joint_soft/quality | d36_t05 | 31.070 | 13.8 | 55296 |
| norm_joint_soft/compact | d16_t05 | 28.034 | 13.5 | 29056 |
| perceptual_static/quality | d36_t1 | 17.981 | 11.3 | 51928 |
| perceptual_static/compact | d16_t1 | 16.437 | 11.0 | 25688 |
| perceptual_joint_soft/quality | d36_t05 | 32.326 | 13.8 | 55296 |
| perceptual_joint_soft/compact | d16_t05 | 30.559 | 13.5 | 29056 |
| g_perceptual_soft | raw control | — | 12.2 | 44640 |
| a_perceptual_joint_guarded | raw control | — | 12.3 | 44640 |
