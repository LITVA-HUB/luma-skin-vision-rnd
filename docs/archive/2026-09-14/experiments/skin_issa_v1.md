# skin_issa_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Измеренные спектры кожи: oracle compression с полным спектром на входе, не точность фотографии.

[Полная папка артефактов](../../../../docs/benchmarks/skin_issa_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_issa_v1/report.md)

SHA-256 отчёта: `7e86e3b0cf50144124edbd45e7f87ad6b10efbdc5c226da811cdf4b49740e234`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_issa_material_audit_amendment_v1.md](../../../../docs/research/skin_issa_material_audit_amendment_v1.md)
- [skin_issa_material_protocol_v1.md](../../../../docs/research/skin_issa_material_protocol_v1.md)
- [skin_issa_next_decision.md](../../../../docs/research/skin_issa_next_decision.md)
- [skin_issa_material_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_issa_material_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_issa_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_issa_audit.py](../modules/scripts__skin_issa_audit.md)
- [scripts/skin_issa_color.py](../modules/scripts__skin_issa_color.md)
- [scripts/skin_issa_data.py](../modules/scripts__skin_issa_data.md)
- [scripts/skin_issa_material.py](../modules/scripts__skin_issa_material.md)
- [scripts/skin_issa_overlap.py](../modules/scripts__skin_issa_overlap.md)
- [scripts/skin_issa_report.py](../modules/scripts__skin_issa_report.md)
- [scripts/skin_issa_verify.py](../modules/scripts__skin_issa_verify.md)
- [tests/test_skin_issa_color.py](../tests/tests__test_skin_issa_color.md)
- [tests/test_skin_issa_data.py](../tests/tests__test_skin_issa_data.md)
- [tests/test_skin_issa_material.py](../tests/tests__test_skin_issa_material.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [independent_audit.json](../../../../docs/benchmarks/skin_issa_v1/independent_audit.json) | {"status": "PASS"} | `64129495ada194b5ecd7ead4fe11d5befed374527d0fd334d3fc0456570960a5` |
| [material_fit.json](../../../../docs/benchmarks/skin_issa_v1/material_fit.json) | {"seconds": 0.015634600014891475} | `6db9b7ec21a962ac07b307ab6a0dbbca8f79bd3c49ee132b40d1a54d409f1c40` |
| [material_lock.json](../../../../docs/benchmarks/skin_issa_v1/material_lock.json) | {} | `05074afc3868e557b7f91c2bbf79a6eb1e568130793295a25a251ce3a815a122` |
| [material_results.json](../../../../docs/benchmarks/skin_issa_v1/material_results.json) | {"scope": "ORACLE full-measured-spectrum compression; NOT camera/RGB/phone skin accuracy"} | `a3c0027a78e3914820d1ae0cbfd860189f1df36d1ded2ab06de40f85c0153900` |
| [metadata.json](../../../../docs/benchmarks/skin_issa_v1/metadata.json) | {"scope": "metadata only, numeric participant endpoints not decoded"} | `a1084472781472cdc89441c4f24482444e1855a10b73d4f7fc9e96d4207755db` |
| [overlap_sensitivity.json](../../../../docs/benchmarks/skin_issa_v1/overlap_sensitivity.json) | {"scope": "POST-HOC descriptive exclusion; original models/split/results unchanged; not an independent confirmation"} | `2ba325d74f041ce3f29d19819ef960e77da88cfaa6f77846b3ff80edafa08ac0` |
| [split_lock.json](../../../../docs/benchmarks/skin_issa_v1/split_lock.json) | {} | `bda5edc37c7fdd67cc54fc39a18771fd18f84daf6dd35a5c0d10bd0db3f98ffb` |
| [train_audit.json](../../../../docs/benchmarks/skin_issa_v1/train_audit.json) | {"scope": "Original measured spectra and cached colorimetry; no camera input or accuracy measured"} | `c410207c702981cfedf87619e1883a9f54d314a7b060de2a682c8e7a4a03ac10` |
| [validation_audit.json](../../../../docs/benchmarks/skin_issa_v1/validation_audit.json) | {"scope": "Original measured spectra and cached colorimetry; no camera input or accuracy measured"} | `0a02624fadab932354c1502d639a7325a747a666bbe108aa70fe06f7eba63477` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Representation | Latent width | Mean DeltaE00 | Median | p95 | Worst source mean | Mean spectral relative L2 |
| --- | --- | --- | --- | --- | --- | --- |
| mean | 0 | 7.0840 | 5.7777 | 17.5594 | 15.4591 | 25.237% |
| reflectance | 2 | 2.3959 | 2.1301 | 5.5471 | 3.5630 | 3.182% |
| reflectance | 3 | 0.4495 | 0.3602 | 1.0564 | 0.6944 | 1.765% |
| reflectance | 4 | 0.4207 | 0.3308 | 1.0115 | 0.7692 | 1.154% |
| reflectance | 6 | 0.2039 | 0.1374 | 0.6062 | 0.3478 | 0.539% |
| reflectance | 8 | 0.0567 | 0.0490 | 0.1332 | 0.0675 | 0.351% |
| density | 2 | 2.3533 | 1.9902 | 5.5246 | 4.5389 | 4.433% |
| density | 3 | 0.6834 | 0.5786 | 1.5333 | 0.8750 | 2.018% |
| density | 4 | 0.3810 | 0.3024 | 1.0357 | 0.5315 | 1.161% |
| density | 6 | 0.2063 | 0.1500 | 0.5790 | 0.2775 | 0.556% |
| density | 8 | 0.0236 | 0.0170 | 0.0611 | 0.0706 | 0.409% |
| logit | 2 | 2.5323 | 2.2303 | 5.7065 | 4.3523 | 3.631% |
| logit | 3 | 0.6224 | 0.5431 | 1.3795 | 0.8551 | 1.921% |
| logit | 4 | 0.3893 | 0.3125 | 0.9624 | 0.5778 | 1.112% |
| logit | 6 | 0.2045 | 0.1478 | 0.5725 | 0.2885 | 0.536% |
| logit | 8 | 0.0259 | 0.0192 | 0.0675 | 0.0593 | 0.395% |

### Таблица 2

| Representation | Width | Mean spectral relative L2 without linked labels |
| --- | --- | --- |
| mean | 0 | 25.257% |
| reflectance | 2 | 3.184% |
| reflectance | 3 | 1.761% |
| reflectance | 4 | 1.150% |
| reflectance | 6 | 0.535% |
| reflectance | 8 | 0.350% |
| density | 2 | 4.429% |
| density | 3 | 2.016% |
| density | 4 | 1.157% |
| density | 6 | 0.553% |
| density | 8 | 0.406% |
| logit | 2 | 3.629% |
| logit | 3 | 1.919% |
| logit | 4 | 1.108% |
| logit | 6 | 0.532% |
| logit | 8 | 0.392% |
