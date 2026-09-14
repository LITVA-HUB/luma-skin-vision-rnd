# skin_mskcc_selective_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Основной независимый тест: 400 изображений / 10 новых людей. 4,457 ΔE00; обычная fusion сильнее.

[Полная папка артефактов](../../../../docs/benchmarks/skin_mskcc_selective_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_mskcc_selective_v1/report.md)

SHA-256 отчёта: `de871913f11b04b272703e24eae266b39816da44efdc95a0e442a2ee0b4976b8`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_mskcc_selective_protocol_v1.md](../../../../docs/research/skin_mskcc_selective_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_mskcc_selective_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_mskcc_selective_audit.py](../modules/scripts__skin_mskcc_selective_audit.md)
- [scripts/skin_mskcc_selective_core.py](../modules/scripts__skin_mskcc_selective_core.md)
- [scripts/skin_mskcc_selective_data.py](../modules/scripts__skin_mskcc_selective_data.md)
- [scripts/skin_mskcc_selective_metrics.py](../modules/scripts__skin_mskcc_selective_metrics.md)
- [scripts/skin_mskcc_selective_profile.py](../modules/scripts__skin_mskcc_selective_profile.md)
- [scripts/skin_mskcc_selective_report.py](../modules/scripts__skin_mskcc_selective_report.md)
- [scripts/skin_mskcc_selective_run.py](../modules/scripts__skin_mskcc_selective_run.md)
- [tests/test_skin_mskcc_selective.py](../tests/tests__test_skin_mskcc_selective.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [calibration.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/calibration.json) | {} | `145965bcf62a61396a7533601462bead711ad5689655902b1788b0a347e2c4b9` |
| [calibration_data.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/calibration_data.json) | {} | `41e82ed8bceafe67a38a2a1c52a8ef7a6af9f6a89f3a8775ead7c2f766b2f4ea` |
| [final_lock.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/final_lock.json) | {} | `8d30596335c20d6d1e57e4acf1994c16983580f76740fe7d2ea36d8feae94d6e` |
| [head_search.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/head_search.json) | {"scope": "SOURCE VALIDATION selection only; no calibration/test used"} | `a0308b32f21201a05400faa5fd057d8bd06c0975c85ceda4be6d43c9c7670495` |
| [oof.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/oof.json) | {} | `7791b2afc1eb56e06f2124624d0b8660e51310d323f4ecef4b1d3383569792da` |
| [oof_audit.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/oof_audit.json) | {"status": "PASS"} | `c3a4d0ecdea8c01531254543c17085f831946e7d9b801266403cfea60385f68d` |
| [precalibration_lock.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/precalibration_lock.json) | {} | `9062bab77366f203684a07051e5da13df92e0fdd349d52ae49b427644af33b7d` |
| [profile.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/profile.json) | {"scope": "Frozen three-seed color estimator, batch1 prepared64x18 descriptors; no witness extraction, density, risk head, JPEG preprocessing or localization included in GPU timing"} | `b44255fd5c5eeea6dcb0cdb33846ee3b2eeb50fc9dad4d0f15487cb03e4401d3` |
| [test_audit.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/test_audit.json) | {"status": "PASS"} | `7b61dbdcdef58430262e4f27f5492688e8aef9f4cce39eefea09ab2197565e67` |
| [test_data.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/test_data.json) | {} | `f731cc383a90cd1c2c23890e1aee37b93db0ebfd8e49698ec105c359943d12ea` |
| [test_results.json](../../../../docs/benchmarks/skin_mskcc_selective_v1/test_results.json) | {} | `5a0ef8824bb929c72f5c5198c2f0c1033db810ee1625632f8ad45240e15a9398` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Accepted | C+ mean | Proposed mean | Proposed median | Proposed p95 | Proposed error >10 |
| --- | --- | --- | --- | --- | --- |
| 100% | 4.4570 | 4.4570 | 4.0012 | 9.2631 | 3.75% |
| 95% | 4.4452 | 4.3176 | 3.9625 | 8.8808 | 3.16% |
| 90% | 4.4401 | 4.2439 | 3.9323 | 8.5672 | 2.78% |
| 80% | 4.3328 | 4.1591 | 3.8834 | 8.0966 | 1.88% |
| 70% | 4.3037 | 4.1219 | 3.8834 | 8.2565 | 1.79% |
| 60% | 4.1801 | 4.1287 | 3.9117 | 7.9873 | 1.67% |

### Таблица 2

| Locally reproduced method | Mean | Median | p95 | Mean at 80% |
| --- | --- | --- | --- | --- |
| fusion_ensemble | 4.3005 | 3.9637 | 8.6024 | 4.1447 |
| fusion_s43 | 4.3162 | 4.0815 | 8.3378 | 4.2093 |
| shared_tight_ensemble | 4.4188 | 3.9121 | 9.2333 | 4.2269 |
| fusion_s29 | 4.4334 | 3.9627 | 8.8522 | 4.2993 |
| cnn_ensemble | 4.4481 | 4.1044 | 8.5979 | 4.3071 |
| votes_huber3_ensemble | 4.4521 | 4.0130 | 9.2261 | 4.2718 |
| votes_mean_ensemble | 4.4570 | 4.0012 | 9.2631 | 4.2770 |
| votes_huber3_s17 | 4.4595 | 4.1001 | 8.9004 | 4.2941 |
| shared_tight_s17 | 4.4599 | 4.0937 | 8.8132 | 4.2946 |
| votes_mean_s17 | 4.4667 | 4.1037 | 9.0071 | 4.3013 |
| votes_huber3_s29 | 4.4936 | 3.9777 | 9.7681 | 4.2959 |
| fusion_s17 | 4.4957 | 4.1849 | 9.5850 | 4.2456 |
| shared_tight_s43 | 4.4962 | 3.9614 | 9.3187 | 4.3156 |
| votes_mean_s29 | 4.4975 | 4.0066 | 9.8370 | 4.3029 |
| votes_huber3_s43 | 4.5072 | 3.9585 | 9.4560 | 4.3237 |
| votes_mean_s43 | 4.5096 | 3.9778 | 9.4858 | 4.3260 |
| global_mlp_s43 | 4.5555 | 4.1031 | 9.4958 | 4.2454 |
| global_mlp_ensemble | 4.5804 | 4.1064 | 10.1316 | 4.2545 |
| local_mean_s43 | 4.5838 | 4.2564 | 8.9917 | 4.4063 |
| local_mean_ensemble | 4.5982 | 4.2837 | 8.9240 | 4.4223 |
| local_tight_s29 | 4.6381 | 4.4280 | 8.9129 | 4.4273 |
| local_mean_s17 | 4.6529 | 4.3302 | 9.0916 | 4.4651 |
| local_tight_ensemble | 4.6698 | 4.5064 | 8.9706 | 4.4646 |
| local_tight_s17 | 4.6804 | 4.4419 | 8.9172 | 4.5008 |
| local_mean_s29 | 4.6968 | 4.2749 | 8.9710 | 4.5460 |
| cnn_s43 | 4.7007 | 4.5379 | 8.8919 | 4.6795 |
| shared_tight_s29 | 4.7291 | 4.4028 | 9.8329 | 4.4994 |
| global_mlp_s17 | 4.7497 | 4.3450 | 10.0382 | 4.4513 |
| control_median_mlp | 4.7605 | 4.3288 | 9.5769 | 4.5118 |
| global_mlp_s29 | 4.7695 | 4.1751 | 10.6908 | 4.4036 |
| control_color_mlp | 4.7807 | 4.1292 | 10.3167 | 4.4641 |
| local_tight_s43 | 4.8109 | 4.5436 | 9.2429 | 4.5765 |
| cnn_s29 | 4.8151 | 4.2954 | 9.3407 | 4.7328 |
| control_median_poly2 | 5.0092 | 4.6852 | 9.7760 | 4.7677 |
| cnn_s17 | 5.0164 | 4.6641 | 9.8988 | 4.6545 |
| control_color_ridge | 5.2225 | 4.7188 | 10.3730 | 4.7300 |
| control_median_ridge | 5.5123 | 5.2385 | 10.1516 | 5.1947 |
| control_hist_ridge | 8.0536 | 4.8284 | 27.3041 | 6.9742 |
| control_color_hist_ridge | 8.2722 | 4.4510 | 25.7718 | 7.1413 |
| control_color_hist_mlp | 9.1778 | 7.0650 | 26.9583 | 8.4183 |
| control_hist_mlp | 10.2095 | 7.3326 | 29.9781 | 9.5529 |

### Таблица 3

| Version | C+ mean at 80% | Proposed mean at 80% | Difference |
| --- | --- | --- | --- |
| single17 | 4.3463 | 4.2382 | -0.1081 |
| single29 | 4.0655 | 4.0396 | -0.0259 |
| single43 | 4.4517 | 4.2370 | -0.2148 |
| ensemble | 4.3328 | 4.1591 | -0.1736 |

### Таблица 4

| Primary model stratum | Images | Mean | p95 |
| --- | --- | --- | --- |
| device: SLR | 152 | 3.8599 | 7.4467 |
| device: ipod | 248 | 4.8230 | 10.2868 |
| image_type: clinical: close-up | 98 | 4.5059 | 9.5339 |
| image_type: dermoscopic | 302 | 4.4411 | 9.1208 |
