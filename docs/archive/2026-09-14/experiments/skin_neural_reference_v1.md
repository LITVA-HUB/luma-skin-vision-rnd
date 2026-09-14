# skin_neural_reference_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Более ёмкий neural reference head в лимите +200k; снижение TRAIN ошибки не перенеслось универсально.

[Полная папка артефактов](../../../../docs/benchmarks/skin_neural_reference_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_neural_reference_v1/report.md)

SHA-256 отчёта: `328b0eae188ab114a2287e4d3c9ebd6017ce3a47b4b37499f363e0f306510623`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_neural_reference_provenance.md](../../../../docs/ip/skin_neural_reference_provenance.md)
- [skin_neural_reference_next_decision.md](../../../../docs/research/skin_neural_reference_next_decision.md)
- [skin_neural_reference_protocol_v1.md](../../../../docs/research/skin_neural_reference_protocol_v1.md)
- [skin_neural_reference_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_neural_reference_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_neural_reference_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_neural_reference.py](../modules/scripts__skin_neural_reference.md)
- [scripts/skin_neural_reference_report.py](../modules/scripts__skin_neural_reference_report.md)
- [scripts/skin_neural_reference_train.py](../modules/scripts__skin_neural_reference_train.md)
- [scripts/skin_neural_reference_verify.py](../modules/scripts__skin_neural_reference_verify.md)
- [tests/test_skin_neural_reference.py](../tests/tests__test_skin_neural_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_neural_reference_v1/audit.json) | {"status": "PASS", "scope": "Exploratory source evidence only; no independent TEST/CAL accessed"} | `346f08ee064a566a12eac1154ece267d030c443470c5bf3e57041e33113f7ca5` |
| [results.json](../../../../docs/benchmarks/skin_neural_reference_v1/results.json) | {"status": "COMPLETE"} | `63e9bf81dd1ce46820a8e04d9d5d2edddd70065f92dec8f1926f21c96b3a8e7a` |
| [source_lock.json](../../../../docs/benchmarks/skin_neural_reference_v1/source_lock.json) | {} | `668c46e53f9ee9dcf8437c81b761dc0521addf03a4ba4e98046ed770b03368b4` |
| [summary.json](../../../../docs/benchmarks/skin_neural_reference_v1/summary.json) | {} | `7b205e11838256f667fe8c0608f9159e361e855e7b0e34f4ba5e938d1de5febc` |
| [test_receipt.json](../../../../docs/benchmarks/skin_neural_reference_v1/test_receipt.json) | {"passed": 373, "seconds": 33.44} | `8df368d3a2e46bf24687e863c02eefab9db51fde1741f2bd06ddf71f3eb3aabe` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Training / domain | System | Mean ± seed SD | Median | p95 | Error >10 | Mean at80% |
| --- | --- | --- | --- | --- | --- | --- |
| mixed /known | base | 3.4406 ±0.0540 | 2.9776 | 6.9316 | 0.38% | 3.2403 |
| mixed /known | residual | 3.6198 ±0.0207 | 3.1246 | 7.6979 | 1.26% | 3.4049 |
| mixed /known | mean | 3.5332 ±0.0549 | 3.0605 | 7.2908 | 0.51% | 3.3346 |
| mixed /known | affine | 3.5182 ±0.0533 | 3.0376 | 7.4892 | 0.51% | 3.3266 |
| from_SLR /known | base | 3.3198 ±0.0202 | 3.1374 | 6.1355 | 0.00% | 3.1820 |
| from_SLR /known | residual | 3.3596 ±0.0741 | 3.1503 | 6.5073 | 0.00% | 3.1591 |
| from_SLR /known | mean | 3.3578 ±0.0238 | 3.1987 | 6.0581 | 0.00% | 3.2178 |
| from_SLR /known | affine | 3.3037 ±0.0072 | 3.0117 | 6.2863 | 0.00% | 3.1675 |
| from_SLR /unseen | base | 5.0193 ±0.2123 | 4.8340 | 9.7825 | 4.55% | 5.0384 |
| from_SLR /unseen | residual | 5.4398 ±0.2291 | 5.1890 | 10.4599 | 7.83% | 5.5616 |
| from_SLR /unseen | mean | 5.0618 ±0.1964 | 4.8164 | 9.8460 | 5.30% | 5.0937 |
| from_SLR /unseen | affine | 4.9658 ±0.2317 | 4.7589 | 9.6942 | 4.80% | 5.0140 |
| from_ipod /known | base | 3.6138 ±0.0414 | 3.1536 | 7.2695 | 2.02% | 3.6023 |
| from_ipod /known | residual | 3.9013 ±0.0185 | 3.3312 | 8.5108 | 1.52% | 3.9031 |
| from_ipod /known | mean | 3.7195 ±0.0417 | 3.2281 | 7.6561 | 2.02% | 3.7111 |
| from_ipod /known | affine | 3.7123 ±0.0689 | 3.1770 | 7.8060 | 2.02% | 3.7196 |
| from_ipod /unseen | base | 5.9635 ±0.3915 | 5.5527 | 11.0896 | 8.59% | 6.2122 |
| from_ipod /unseen | residual | 6.4545 ±0.3539 | 6.1740 | 12.1336 | 12.12% | 7.0439 |
| from_ipod /unseen | mean | 6.0437 ±0.3787 | 5.6978 | 11.1263 | 9.60% | 6.3806 |
| from_ipod /unseen | affine | 6.4782 ±0.1610 | 5.9807 | 11.9513 | 12.88% | 6.8604 |

### Таблица 2

| Adapter | Parameters | Head train peak MiB | Batch1 median ms range | Complete checkpoint bytes | Extra reference scalars |
| --- | --- | --- | --- | --- | --- |
| residual | 1,123,092 | 75.181–79.566 | 0.725–1.119 | 4505423–4505423 | 0–0 |
| mean | 1,123,092 | 81.353–100.932 | 0.978–2.283 | 4529999–4578895 | 6137–18354 |
| affine | 1,123,092 | 84.767–110.073 | 1.424–3.042 | 4529999–4578895 | 6137–18354 |
