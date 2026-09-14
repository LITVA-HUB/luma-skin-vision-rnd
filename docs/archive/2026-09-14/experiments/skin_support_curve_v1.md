# skin_support_curve_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

6/12/18 обучающих людей при одинаковом числе обновлений: больше людей помогло, pixel adapter не убедил.

[Полная папка артефактов](../../../../docs/benchmarks/skin_support_curve_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_support_curve_v1/report.md)

SHA-256 отчёта: `7aed99e64d79d9e2d6352b813585cb3381a6918b2ac34fc6061cc3e4189de7f4`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_support_curve_provenance.md](../../../../docs/ip/skin_support_curve_provenance.md)
- [skin_support_curve_interventions_v1.md](../../../../docs/research/skin_support_curve_interventions_v1.md)
- [skin_support_curve_next_decision.md](../../../../docs/research/skin_support_curve_next_decision.md)
- [skin_support_curve_protocol_v1.md](../../../../docs/research/skin_support_curve_protocol_v1.md)
- [skin_support_curve_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_support_curve_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_support_curve_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_support_curve.py](../modules/scripts__skin_support_curve.md)
- [scripts/skin_support_curve_interventions.py](../modules/scripts__skin_support_curve_interventions.md)
- [scripts/skin_support_curve_report.py](../modules/scripts__skin_support_curve_report.md)
- [scripts/skin_support_curve_train.py](../modules/scripts__skin_support_curve_train.md)
- [scripts/skin_support_curve_verify.py](../modules/scripts__skin_support_curve_verify.md)
- [tests/test_skin_support_curve.py](../tests/tests__test_skin_support_curve.md)
- [tests/test_skin_support_curve_interventions.py](../tests/tests__test_skin_support_curve_interventions.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_support_curve_v1/audit.json) | {"status": "PASS"} | `0234f3d615d2835ac7d7959f37919c56af81fbb7760833bf97a3ca299df176c2` |
| [interventions.json](../../../../docs/benchmarks/skin_support_curve_v1/interventions.json) | {"status": "PASS", "scope": "post-fit TRAIN/internal-holdout diagnostics; no new training or selection"} | `6e0b70d8822ce573d81016ec5c4f9bd9e03fb7965246527ff4bef33c81148ec6` |
| [source_lock.json](../../../../docs/benchmarks/skin_support_curve_v1/source_lock.json) | {} | `b2c52f9f56d6ef3d62e79b6f82f83667c92d0564e83fd5a50229ee7a4caedb35` |
| [summary.json](../../../../docs/benchmarks/skin_support_curve_v1/summary.json) | {"scope": "TRAIN-only internal holdout, exploratory; not fresh external accuracy"} | `c21a0e023fc1cbee1293748d1a89e0167dd90435d0cabba3d705c9f4ee928ddc` |
| [test_receipt.json](../../../../docs/benchmarks/skin_support_curve_v1/test_receipt.json) | {"passed": 353} | `6c06d3fea0d0dee739506e212fd85d2582ffd52a67741e84b2413151de0bf1d6` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| People / model | Training mean | Holdout mean | Median | p95 | At 80% |
| --- | --- | --- | --- | --- | --- |
| 6/baseline | 3.0095 | 7.2439 | 5.7243 | 17.8575 | 6.1440 |
| 6/statistics | 2.9652 | 7.2068 | 5.6289 | 17.7882 | 6.1219 |
| 6/pixels | 2.9945 | 7.1762 | 5.6432 | 17.7771 | 6.1150 |
| 12/baseline | 3.4065 | 6.2843 | 5.1261 | 15.0596 | 5.5917 |
| 12/statistics | 3.3636 | 6.2940 | 5.1346 | 15.6009 | 5.5743 |
| 12/pixels | 3.3895 | 6.2744 | 5.1056 | 15.2808 | 5.5750 |
| 18/baseline | 3.5823 | 6.1082 | 5.1353 | 14.9960 | 5.3918 |
| 18/statistics | 3.5366 | 6.1560 | 5.1727 | 15.1242 | 5.4256 |
| 18/pixels | 3.5742 | 6.1258 | 5.1768 | 15.0828 | 5.4081 |

### Таблица 2

| People | Candidate vs control | Image difference | Patient difference | Patient interval |
| --- | --- | --- | --- | --- |
| 6 | pixels vs statistics | -0.0305 | -0.0022 | [-0.1123, 0.1230] |
| 6 | pixels vs baseline | -0.0677 | -0.0872 | [-0.1966, 0.0069] |
| 6 | statistics vs baseline | -0.0371 | -0.0850 | [-0.2956, 0.0443] |
| 12 | pixels vs statistics | -0.0196 | -0.0139 | [-0.0522, 0.0245] |
| 12 | pixels vs baseline | -0.0100 | -0.0128 | [-0.0395, 0.0173] |
| 12 | statistics vs baseline | 0.0096 | 0.0011 | [-0.0569, 0.0608] |
| 18 | pixels vs statistics | -0.0302 | -0.0176 | [-0.0702, 0.0389] |
| 18 | pixels vs baseline | 0.0176 | 0.0136 | [-0.0045, 0.0309] |
| 18 | statistics vs baseline | 0.0478 | 0.0311 | [-0.0432, 0.0953] |

### Таблица 3

| People / model | Intervention | Mean skin error | Prediction Lab RMS change |
| --- | --- | --- | --- |
| 6/statistics | zero | 7.9942 | 2.1944 |
| 6/pixels | zero | 7.6479 | 1.2816 |
| 6/pixels | shuffle | 7.1756 | 0.0373 |
| 6/pixels | mean | 7.1695 | 0.0322 |
| 12/statistics | zero | 7.6113 | 2.5550 |
| 12/pixels | zero | 6.8297 | 1.3601 |
| 12/pixels | shuffle | 6.2706 | 0.0267 |
| 12/pixels | mean | 6.2654 | 0.0214 |
| 18/statistics | zero | 7.3067 | 2.1069 |
| 18/pixels | zero | 6.6175 | 0.9890 |
| 18/pixels | shuffle | 6.1266 | 0.0175 |
| 18/pixels | mean | 6.1199 | 0.0160 |
