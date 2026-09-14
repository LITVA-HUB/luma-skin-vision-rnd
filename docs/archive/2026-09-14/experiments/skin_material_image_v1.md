# skin_material_image_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Физический спектральный декодер на реальных изображениях: диапазон представимых цветов не объясняет основной провал.

[Полная папка артефактов](../../../../docs/benchmarks/skin_material_image_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_material_image_v1/report.md)

SHA-256 отчёта: `f21f569ca8c28ca49371a229688a73f0b73c2850a33f26d791c129cce48f4da3`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_material_image_protocol_v1.md](../../../../docs/research/skin_material_image_protocol_v1.md)
- [skin_material_image_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_material_image_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_material_image_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.


## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_material_image_v1/audit.json) | {"status": "PASS"} | `5f9b6cb0c4eccf27244de06923abc45171c42ffb6b50c4f33643b4f80ab82b16` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_material_image_v1/audit_mixed.json) | {"status": "PASS"} | `aa96c9759acbb4dc9d7ae88a41264121e1bdb7bb2290b6509f8a682e368d69ad` |
| [feasibility_audit.json](../../../../docs/benchmarks/skin_material_image_v1/feasibility_audit.json) | {"status": "PASS"} | `a00a92c184dda5a3873740542b2371578515173c80f91c685984626220db95c1` |
| [feasibility_train.json](../../../../docs/benchmarks/skin_material_image_v1/feasibility_train.json) | {"scope": "POST-HOC TRAIN known-target oracle; no image accuracy, no global infeasibility proof"} | `a46f4856ab4e73eab1237864de091555967f09ee7d0b37585d33cdeb9d1db6fa` |
| [prior_receipt.json](../../../../docs/benchmarks/skin_material_image_v1/prior_receipt.json) | {"scope": "ISSA TRAIN-only prior, modelled D65/10degree color interface; MSKCC labels unchanged"} | `379d4887bf46745292f2de71d79acd80f36688c58d356a03412261474b1903ce` |
| [source_lock.json](../../../../docs/benchmarks/skin_material_image_v1/source_lock.json) | {} | `a3fb3fad67ad722bf867a89da48b22dd239bbbe03e7787757b46bf6c4e297868` |
| [summary.json](../../../../docs/benchmarks/skin_material_image_v1/summary.json) | {"scope": "45fixed source image fits; averages of separate seed scores, not ensembles"} | `2dd957d81996a1e136568f43bd01031e965c1bdeba85bf07ac14ea052187bb15` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Arm | Mean DeltaE00 | Mean seed p95 | Mean at80% | Seed means |
| --- | --- | --- | --- | --- | --- |
| mixed | direct | 3.4406 | 6.9303 | 3.4407 | 3.3785, 3.4698, 3.4735 |
| mixed | tangent | 3.5112 | 7.3823 | 3.5393 | 3.5084, 3.4928, 3.5323 |
| mixed | material | 3.5141 | 7.2257 | 3.4992 | 3.5291, 3.5002, 3.5131 |
| mixed | tangent_residual | 3.4510 | 7.3864 | 3.4953 | 3.3890, 3.4773, 3.4866 |
| mixed | material_residual | 3.4529 | 7.2327 | 3.4787 | 3.4448, 3.4397, 3.4742 |
| from_SLR | direct | 5.0193 | 9.7826 | 5.1088 | 4.9977, 5.2416, 4.8187 |
| from_SLR | tangent | 4.9531 | 9.1153 | 5.0204 | 5.0433, 4.8575, 4.9584 |
| from_SLR | material | 4.9083 | 9.1436 | 4.9778 | 4.9087, 4.9035, 4.9126 |
| from_SLR | tangent_residual | 5.0311 | 9.7962 | 5.3053 | 4.9601, 5.2824, 4.8510 |
| from_SLR | material_residual | 5.0100 | 9.8046 | 5.3498 | 4.8415, 5.3424, 4.8461 |
| from_ipod | direct | 5.9635 | 11.0899 | 5.7914 | 5.7634, 6.4147, 5.7124 |
| from_ipod | tangent | 5.4509 | 10.1641 | 5.2781 | 5.6321, 5.1510, 5.5695 |
| from_ipod | material | 6.1311 | 10.5748 | 6.4299 | 6.3720, 6.2309, 5.7906 |
| from_ipod | tangent_residual | 6.0128 | 10.4162 | 5.9151 | 5.6676, 6.0342, 6.3367 |
| from_ipod | material_residual | 6.1110 | 10.5066 | 6.2489 | 5.6713, 6.4799, 6.1817 |

### Таблица 2

| Protocol | Method | Mean DeltaE00 |
| --- | --- | --- |
| mixed | plain_mse | 3.4771 |
| mixed | mixture_mse | 3.4406 |
| mixed | training_only_graph | 3.6394 |
| from_SLR | plain_mse | 5.8301 |
| from_SLR | mixture_mse | 5.0193 |
| from_SLR | training_only_graph | 5.8339 |
| from_ipod | plain_mse | 5.5089 |
| from_ipod | mixture_mse | 5.9635 |
| from_ipod | training_only_graph | 4.9736 |

### Таблица 3

| Protocol | Arm vs control | Image-mean difference | Patient-mean95% interval | All3seeds improve |
| --- | --- | --- | --- | --- |
| mixed | material vs tangent | 0.0030 | [-0.0403,0.0431] | False |
| mixed | material_residual vs tangent_residual | 0.0019 | [-0.0434,0.0403] | False |
| mixed | material_residual vs direct | 0.0123 | [-0.0802,0.1022] | False |
| from_SLR | material vs tangent | -0.0448 | [-0.1765,0.0644] | False |
| from_SLR | material_residual vs tangent_residual | -0.0211 | [-0.1402,0.1384] | False |
| from_SLR | material_residual vs direct | -0.0094 | [-0.4700,0.2464] | False |
| from_ipod | material vs tangent | 0.6803 | [0.0927,1.0750] | False |
| from_ipod | material_residual vs tangent_residual | 0.0981 | [0.0519,0.1348] | False |
| from_ipod | material_residual vs direct | 0.1475 | [-0.7704,0.6300] | False |
