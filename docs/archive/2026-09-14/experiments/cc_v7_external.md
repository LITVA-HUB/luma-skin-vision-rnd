# cc_v7_external

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

29 заранее зафиксированных методов; новые изображения внешних камер, с оговоркой о сценах и истории камер.

[Полная папка артефактов](../../../../docs/benchmarks/cc_v7_external)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/cc_v7_external/report.md)

SHA-256 отчёта: `c156381e338da039049e977c57f6aa5468dd2f62b3695eadcecb48141c93f1b7`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [cc_v7_external_methods_protocol.md](../../../../docs/research/cc_v7_external_methods_protocol.md)
- [report.md](../../../../docs/benchmarks/cc_v7_external/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/cc_v7_external_audit.py](../modules/scripts__cc_v7_external_audit.md)
- [scripts/cc_v7_external_benchmark.py](../modules/scripts__cc_v7_external_benchmark.md)
- [scripts/cc_v7_external_controls.py](../modules/scripts__cc_v7_external_controls.md)
- [scripts/cc_v7_external_data.py](../modules/scripts__cc_v7_external_data.md)
- [scripts/cc_v7_external_lock.py](../modules/scripts__cc_v7_external_lock.md)
- [scripts/cc_v7_external_population.py](../modules/scripts__cc_v7_external_population.md)
- [scripts/cc_v7_external_report.py](../modules/scripts__cc_v7_external_report.md)
- [tests/test_cc_v7_external.py](../tests/tests__test_cc_v7_external.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [independent_metric_audit.json](../../../../docs/benchmarks/cc_v7_external/independent_metric_audit.json) | {"status": "PASS: independent scalar atan2 metrics, all rankings, curves and source thresholds"} | `4da21e7a4db678334ba177ed57f6f4592b50db414390ebdfb10420a1d9ad95f5` |
| [method_lock.json](../../../../docs/benchmarks/cc_v7_external/method_lock.json) | {"status": "ALL29 METHODS FROZEN BEFORE NEW TARGET DECODE"} | `2828472f40348ebb9785c8d294e4353fd6fc0d33956191756c074d8912e3948d` |
| [prediction_smoke.json](../../../../docs/benchmarks/cc_v7_external/prediction_smoke.json) | {"scope": "Three TRAIN inputs only, no GT; every architecture/payload kind"} | `ff5049da9c1a143589cf6d90636edf4c958fca7b972fbc9789cd0e5ce1df67d1` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Family | Models | Known full ° | Unseen full ° | Unseen risk80 ° | Unseen p95 ° | Source80 threshold target coverage |
| --- | --- | --- | --- | --- | --- | --- |
| direct_hgb7 | 1 | 2.2431 | 6.6570 | 6.7645 | 14.1021 | 54.57% |
| fourier_ridge_combined | 1 | 2.6330 | 4.3552 | 3.7135 | 10.9136 | 33.44% |
| fourier_ridge_raw | 1 | 2.6330 | 4.3552 | 3.6233 | 10.9136 | 28.39% |
| gray_edge | 1 | 3.8866 | 6.9723 | 5.9718 | 18.6961 | 37.22% |
| gray_world | 1 | 4.5922 | 6.3911 | 5.8960 | 16.2521 | 43.22% |
| gw_ridge1 | 1 | 2.9798 | 4.7642 | 4.2349 | 12.5935 | 40.38% |
| max_rgb | 1 | 5.6572 | 11.2831 | 9.6008 | 21.0070 | 39.75% |
| shades_gray | 1 | 3.5735 | 6.3484 | 5.7798 | 15.8396 | 36.59% |
| v2_direct | 3 | 2.1959 | 5.7125 | 5.6778 | 12.1937 | 59.41% |
| v2_sog | 3 | 2.8212 | 5.0868 | 4.0576 | 14.2511 | 47.74% |
| v7_canonical_teacher_native | 3 | 2.2270 | 5.7573 | 5.7084 | 12.5398 | 60.57% |
| v7_canonical_teacher_sensor | 3 | 2.6017 | 5.5013 | 5.3135 | 12.6369 | 41.43% |
| v7_gt_native | 3 | 2.1898 | 5.6224 | 5.6985 | 12.1891 | 62.99% |
| v7_gt_sensor | 3 | 2.6167 | 5.0057 | 4.7682 | 12.7812 | 53.73% |
| v7_raw_teacher_sensor | 3 | 2.4999 | 4.9761 | 4.7481 | 12.2134 | 40.59% |

### Таблица 2

| Family | 100% | 95% | 90% | 80% | 70% | 60% |
| --- | --- | --- | --- | --- | --- | --- |
| direct_hgb7 | 6.6570 | 6.6717 | 6.7076 | 6.7645 | 6.8360 | 7.0518 |
| fourier_ridge_combined | 4.3552 | 4.1138 | 4.0200 | 3.7135 | 3.4221 | 3.2805 |
| fourier_ridge_raw | 4.3552 | 4.1410 | 3.9090 | 3.6233 | 3.4107 | 3.2632 |
| gray_edge | 6.9723 | 6.5327 | 6.2793 | 5.9718 | 5.3250 | 5.0314 |
| gray_world | 6.3911 | 6.3543 | 6.2296 | 5.8960 | 5.6698 | 5.4065 |
| gw_ridge1 | 4.7642 | 4.5697 | 4.4321 | 4.2349 | 4.0659 | 3.9667 |
| max_rgb | 11.2831 | 10.8083 | 10.3527 | 9.6008 | 8.8480 | 7.8177 |
| shades_gray | 6.3484 | 6.1238 | 6.0461 | 5.7798 | 5.4415 | 5.1174 |
| v2_direct | 5.7125 | 5.7256 | 5.6847 | 5.6778 | 5.7246 | 5.7754 |
| v2_sog | 5.0868 | 4.7563 | 4.4618 | 4.0576 | 3.8484 | 3.6032 |
| v7_canonical_teacher_native | 5.7573 | 5.6943 | 5.6929 | 5.7084 | 5.8307 | 5.9133 |
| v7_canonical_teacher_sensor | 5.5013 | 5.4848 | 5.4387 | 5.3135 | 5.2273 | 5.1017 |
| v7_gt_native | 5.6224 | 5.6176 | 5.6739 | 5.6985 | 5.7229 | 5.8018 |
| v7_gt_sensor | 5.0057 | 4.8973 | 4.8284 | 4.7682 | 4.6843 | 4.4635 |
| v7_raw_teacher_sensor | 4.9761 | 4.8883 | 4.8349 | 4.7481 | 4.6559 | 4.5937 |

### Таблица 3

| Unseen primary camera | Family | Full ° | Risk80 ° |
| --- | --- | --- | --- |
| Canon_5DSR | direct_hgb7 | 4.5641 | 4.6809 |
| Canon_5DSR | fourier_ridge_combined | 4.1018 | 3.5485 |
| Canon_5DSR | fourier_ridge_raw | 4.1018 | 3.6181 |
| Canon_5DSR | gray_edge | 8.0749 | 6.7676 |
| Canon_5DSR | gray_world | 5.7641 | 5.6575 |
| Canon_5DSR | gw_ridge1 | 4.7662 | 4.0711 |
| Canon_5DSR | max_rgb | 13.8731 | 12.3932 |
| Canon_5DSR | shades_gray | 7.1604 | 6.3223 |
| Canon_5DSR | v2_direct | 4.5723 | 4.3968 |
| Canon_5DSR | v2_sog | 5.3147 | 4.2168 |
| Canon_5DSR | v7_canonical_teacher_native | 4.5505 | 4.3379 |
| Canon_5DSR | v7_canonical_teacher_sensor | 5.1126 | 4.9779 |
| Canon_5DSR | v7_gt_native | 4.3803 | 4.2839 |
| Canon_5DSR | v7_gt_sensor | 4.7330 | 4.5468 |
| Canon_5DSR | v7_raw_teacher_sensor | 4.6572 | 4.3862 |
| Nikon_D810 | direct_hgb7 | 6.9860 | 7.1427 |
| Nikon_D810 | fourier_ridge_combined | 4.2714 | 3.6632 |
| Nikon_D810 | fourier_ridge_raw | 4.2714 | 3.2162 |
| Nikon_D810 | gray_edge | 7.6139 | 6.8050 |
| Nikon_D810 | gray_world | 7.3717 | 6.6684 |
| Nikon_D810 | gw_ridge1 | 5.1989 | 4.7584 |
| Nikon_D810 | max_rgb | 10.4361 | 9.1479 |
| Nikon_D810 | shades_gray | 6.8381 | 6.8086 |
| Nikon_D810 | v2_direct | 5.5107 | 5.3762 |
| Nikon_D810 | v2_sog | 5.7882 | 4.7116 |
| Nikon_D810 | v7_canonical_teacher_native | 5.4558 | 5.2684 |
| Nikon_D810 | v7_canonical_teacher_sensor | 5.5921 | 5.2697 |
| Nikon_D810 | v7_gt_native | 5.3172 | 5.2329 |
| Nikon_D810 | v7_gt_sensor | 5.1559 | 4.7606 |
| Nikon_D810 | v7_raw_teacher_sensor | 4.9541 | 4.6085 |
| Sony_IMX135_BLCCSC | direct_hgb7 | 8.4089 | 8.1629 |
| Sony_IMX135_BLCCSC | fourier_ridge_combined | 4.7031 | 4.0047 |
| Sony_IMX135_BLCCSC | fourier_ridge_raw | 4.7031 | 4.0357 |
| Sony_IMX135_BLCCSC | gray_edge | 5.1543 | 4.4937 |
| Sony_IMX135_BLCCSC | gray_world | 5.9476 | 5.2775 |
| Sony_IMX135_BLCCSC | gw_ridge1 | 4.2848 | 3.9827 |
| Sony_IMX135_BLCCSC | max_rgb | 9.5976 | 7.9660 |
| Sony_IMX135_BLCCSC | shades_gray | 4.9907 | 4.5920 |
| Sony_IMX135_BLCCSC | v2_direct | 7.0854 | 7.1353 |
| Sony_IMX135_BLCCSC | v2_sog | 4.0864 | 3.5962 |
| Sony_IMX135_BLCCSC | v7_canonical_teacher_native | 7.3070 | 7.4871 |
| Sony_IMX135_BLCCSC | v7_canonical_teacher_sensor | 5.7941 | 5.7590 |
| Sony_IMX135_BLCCSC | v7_gt_native | 7.2116 | 7.3739 |
| Sony_IMX135_BLCCSC | v7_gt_sensor | 5.1161 | 4.9930 |
| Sony_IMX135_BLCCSC | v7_raw_teacher_sensor | 5.3224 | 5.2079 |

### Таблица 4

| Prespecified contrast A minus B | Full difference and95% proxy-cluster CI | Risk80 difference and95% CI |
| --- | --- | --- |
| v7_canonical_teacher_sensor minus v7_raw_teacher_sensor | 0.5252 / [0.3807516422907444, 0.6679193442952731] | 0.5654 / [0.38527027242776224, 0.7756663213958984] |
| v7_gt_sensor minus v7_gt_native | -0.6166 / [-0.919849082598471, -0.2994761241580536] | -0.9303 / [-1.237962592729787, -0.609738587582312] |
| v7_canonical_teacher_sensor minus v2_sog | 0.4145 / [0.049364993869361974, 0.7561906244276781] | 1.2559 / [0.9349623652208364, 1.577695655594537] |
| v7_gt_sensor minus v2_sog | -0.0810 / [-0.4184387360488526, 0.24668466184038124] | 0.7106 / [0.39278093440936035, 1.0052597021629033] |
| v7_canonical_teacher_sensor minus gw_ridge1 | 0.7371 / [0.4378916979055937, 1.0286145685060983] | 1.0786 / [0.7203505887986162, 1.4349225951182325] |
| v7_canonical_teacher_sensor minus fourier_ridge_combined | 1.1461 / [0.7095362888228725, 1.6030339309745902] | 1.6000 / [1.2285478255694045, 1.9651182481587783] |
