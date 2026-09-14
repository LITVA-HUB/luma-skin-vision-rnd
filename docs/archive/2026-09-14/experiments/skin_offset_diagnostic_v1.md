# skin_offset_diagnostic_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Привилегированные offsets по эталонам других evaluation людей; не результат без калибровки.

[Полная папка артефактов](../../../../docs/benchmarks/skin_offset_diagnostic_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_offset_diagnostic_v1/report.md)

SHA-256 отчёта: `75b4bce8f7b0087236bfb2b5ebc1c0de6e3c7ab33c124fbb1637efdfd0d1b119`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_offset_diagnostic_provenance.md](../../../../docs/ip/skin_offset_diagnostic_provenance.md)
- [skin_offset_diagnostic_next_decision.md](../../../../docs/research/skin_offset_diagnostic_next_decision.md)
- [skin_offset_diagnostic_protocol_v1.md](../../../../docs/research/skin_offset_diagnostic_protocol_v1.md)
- [skin_offset_diagnostic_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_offset_diagnostic_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_offset_diagnostic_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_offset_diagnostic.py](../modules/scripts__skin_offset_diagnostic.md)
- [scripts/skin_offset_diagnostic_report.py](../modules/scripts__skin_offset_diagnostic_report.md)
- [scripts/skin_offset_diagnostic_verify.py](../modules/scripts__skin_offset_diagnostic_verify.md)
- [tests/test_skin_offset_diagnostic.py](../tests/tests__test_skin_offset_diagnostic.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/audit.json) | {"status": "PASS"} | `1ef3c0ccf754df702ef4c5ad2aa02efd0f6b5b8bcabf48987f9d77cd2d626be3` |
| [energy_diagnostic.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/energy_diagnostic.json) | {"scope": "Descriptive squared Euclidean native-Lab identity, not DeltaE00 accuracy"} | `806e9de9a552ffb29889d9ffb9267bb80fb60d3210e16a5c6b3f360bf5dc821a` |
| [results.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/results.json) | {"scope": "PRIVILEGED REFERENCE-CALIBRATION COMPARATOR; not the single-image model"} | `ed1048c4602b698b02e80985fd04a35d7e7f1c79ac6c61b94cd558ea7d7fa080` |
| [source_lock.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/source_lock.json) | {} | `7921ef190ca9d8b7ec738fc414140b978c97fdeb650fd0e097889cc94b134ad2` |
| [summary.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/summary.json) | {"scope": "PRIVILEGED REFERENCE-CALIBRATION COMPARATOR; no production accuracy improvement"} | `6a797b9903f7e77e3ebf4524033d602f80efb88713c078e74451c87cc8601f24` |
| [test_receipt.json](../../../../docs/benchmarks/skin_offset_diagnostic_v1/test_receipt.json) | {"passed": 361} | `eae634628affd3fdaff7ed92feaef8fdd05d1f10901e6ec173db3965f7c86c37` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol / domain / model | Original mean | Privileged half | Privileged full | Original80% | Privileged full80% |
| --- | --- | --- | --- | --- | --- |
| mixed/known/image | 3.8277 | 3.6541 | 3.5990 | 3.8015 | 3.5629 |
| mixed/known/person_site | 3.8120 | 3.6729 | 3.6259 | 3.7878 | 3.5943 |
| mixed/known/site | 3.7959 | 3.6345 | 3.5792 | 3.7675 | 3.5402 |
| mixed/known/color | 3.7864 | 3.6950 | 3.6712 | 3.7591 | 3.6276 |
| mixed/known/color_ipw | 3.7976 | 3.6561 | 3.6104 | 3.7747 | 3.5805 |
| mixed/known/person_color | 3.7435 | 3.6163 | 3.5808 | 3.7264 | 3.5581 |
| from_SLR/known/image | 3.4210 | 3.5995 | 3.8244 | 3.2311 | 3.6006 |
| from_SLR/known/person_site | 3.3848 | 3.5396 | 3.7422 | 3.1936 | 3.5183 |
| from_SLR/known/site | 3.3870 | 3.5640 | 3.7848 | 3.2312 | 3.5896 |
| from_SLR/known/color | 3.4000 | 3.5397 | 3.7258 | 3.2406 | 3.5214 |
| from_SLR/known/color_ipw | 3.3841 | 3.5468 | 3.7567 | 3.2011 | 3.5429 |
| from_SLR/known/person_color | 3.4478 | 3.6366 | 3.8774 | 3.2447 | 3.6410 |
| from_SLR/unseen/image | 5.4877 | 5.7066 | 6.2298 | 5.5556 | 5.8688 |
| from_SLR/unseen/person_site | 5.7261 | 5.8415 | 6.3051 | 5.4999 | 5.8817 |
| from_SLR/unseen/site | 6.0016 | 6.1897 | 6.7522 | 5.6250 | 6.2222 |
| from_SLR/unseen/color | 5.7192 | 5.7497 | 6.1300 | 5.5271 | 5.7954 |
| from_SLR/unseen/color_ipw | 5.7416 | 5.7538 | 6.1043 | 5.7681 | 5.8750 |
| from_SLR/unseen/person_color | 6.1545 | 6.6762 | 7.4325 | 5.6200 | 6.6008 |
| from_ipod/known/image | 4.0558 | 3.7594 | 3.7239 | 4.0050 | 3.5686 |
| from_ipod/known/person_site | 3.9949 | 3.6854 | 3.6186 | 3.9592 | 3.4699 |
| from_ipod/known/site | 4.0226 | 3.7639 | 3.7186 | 3.9571 | 3.5458 |
| from_ipod/known/color | 4.0722 | 3.8166 | 3.7815 | 4.0312 | 3.6060 |
| from_ipod/known/color_ipw | 4.0460 | 3.7123 | 3.6394 | 4.0037 | 3.4730 |
| from_ipod/known/person_color | 4.0492 | 3.7343 | 3.6774 | 3.9987 | 3.5076 |
| from_ipod/unseen/image | 6.5602 | 6.1725 | 6.0229 | 6.5114 | 5.6142 |
| from_ipod/unseen/person_site | 6.0825 | 5.4695 | 5.2299 | 6.3985 | 5.0854 |
| from_ipod/unseen/site | 6.1048 | 5.6020 | 5.3826 | 6.2809 | 5.1367 |
| from_ipod/unseen/color | 6.6475 | 6.2406 | 6.1306 | 6.5616 | 5.7747 |
| from_ipod/unseen/color_ipw | 6.4344 | 6.0012 | 5.8297 | 6.4994 | 5.4604 |
| from_ipod/unseen/person_color | 6.4074 | 5.7353 | 5.4434 | 6.7609 | 5.2665 |

### Таблица 2

| Protocol / domain / model | Shared residual energy | Between-person energy | Full squared-Lab error change |
| --- | --- | --- | --- |
| mixed/known/image | 2.5555 | 2.3321 | -1.5294 |
| mixed/known/person_site | 2.1072 | 2.1105 | -1.1785 |
| mixed/known/site | 2.2859 | 2.1257 | -1.3506 |
| mixed/known/color | 1.5574 | 2.1911 | -0.5934 |
| mixed/known/color_ipw | 2.1742 | 2.1671 | -1.2206 |
| mixed/known/person_color | 1.9402 | 2.2250 | -0.9612 |
| from_SLR/known/image | 0.2956 | 4.4279 | 5.2393 |
| from_SLR/known/person_site | 0.4117 | 4.1252 | 4.7447 |
| from_SLR/known/site | 0.2967 | 4.4324 | 5.2438 |
| from_SLR/known/color | 0.4512 | 3.8753 | 4.3929 |
| from_SLR/known/color_ipw | 0.4072 | 4.1934 | 4.8345 |
| from_SLR/known/person_color | 0.3421 | 4.5778 | 5.3802 |
| from_SLR/unseen/image | 8.8848 | 14.3471 | 9.0491 |
| from_SLR/unseen/person_site | 10.6186 | 15.7607 | 9.0823 |
| from_SLR/unseen/site | 10.6270 | 20.6881 | 15.2331 |
| from_SLR/unseen/color | 11.2316 | 13.0405 | 5.0691 |
| from_SLR/unseen/color_ipw | 11.3555 | 13.2408 | 5.1955 |
| from_SLR/unseen/person_color | 6.2580 | 24.2008 | 23.9930 |
| from_ipod/known/image | 6.1965 | 0.7977 | -5.1994 |
| from_ipod/known/person_site | 5.8350 | 0.5684 | -5.1245 |
| from_ipod/known/site | 5.2940 | 0.6105 | -4.5309 |
| from_ipod/known/color | 5.2369 | 0.8107 | -4.2235 |
| from_ipod/known/color_ipw | 6.3395 | 0.5989 | -5.5910 |
| from_ipod/known/person_color | 6.2348 | 0.6882 | -5.3746 |
| from_ipod/unseen/image | 14.0537 | 3.1965 | -10.0581 |
| from_ipod/unseen/person_site | 16.6367 | 2.3505 | -13.6985 |
| from_ipod/unseen/site | 14.7500 | 2.3020 | -11.8725 |
| from_ipod/unseen/color | 16.0726 | 4.3444 | -10.6420 |
| from_ipod/unseen/color_ipw | 14.7394 | 3.2081 | -10.7292 |
| from_ipod/unseen/person_color | 19.0466 | 2.0845 | -16.4410 |
