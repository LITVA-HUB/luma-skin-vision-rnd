# skin_local_reference_transfer_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Перенос локального эталонного банка на камеры не побеждает сильные нейронные контроли.

[Полная папка артефактов](../../../../docs/benchmarks/skin_local_reference_transfer_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_local_reference_transfer_v1/report.md)

SHA-256 отчёта: `c98e3db9c04cb99387de56ccc6468f59825750bc36f5d3b09e5fcbdbb67df042`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_local_reference_transfer_protocol_v1.md](../../../../docs/research/skin_local_reference_transfer_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_local_reference_transfer_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_local_reference_transfer.py](../modules/scripts__skin_local_reference_transfer.md)
- [scripts/skin_local_reference_transfer_verify.py](../modules/scripts__skin_local_reference_transfer_verify.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_local_reference_transfer_v1/audit.json) | {"status": "PASS"} | `800804d920c527cd3341922e74411a7686ceda0c1c23296759c1c0b457f5440a` |
| [results.json](../../../../docs/benchmarks/skin_local_reference_transfer_v1/results.json) | {} | `f5cd88464b917bab942b9cc70294ae7291132ae1266d8afa3f8d7d448ac96feb` |
| [source_lock.json](../../../../docs/benchmarks/skin_local_reference_transfer_v1/source_lock.json) | {} | `f91603ff65c7af57f24726b9ddb8e1fff632a82387fb71c5a18fe5be3444bbdc` |
| [summary.json](../../../../docs/benchmarks/skin_local_reference_transfer_v1/summary.json) | {} | `764bb585c1566e2216ff3a05e6d3e2e1afd3c5563e9731346cec54de7614380d` |
| [test_receipt.json](../../../../docs/benchmarks/skin_local_reference_transfer_v1/test_receipt.json) | {"passed": 370, "seconds": 33.53} | `a98afd19b3d62bf8acf1fd2cc50c1ab8779e23c2899c7fdeb427bea14062e5ae` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| System | Mean DeltaE00 | Median | p95 | Error>10 | Mean at80% | People improved vs ridge |
| --- | --- | --- | --- | --- | --- | --- |
| global_ridge | 5.4259 | 4.7431 | 11.4364 | 7.35% | 4.9899 | 0/24 |
| global_mean | 10.9115 | 9.8601 | 22.6492 | 49.17% | 10.6179 | 2/24 |
| appearance_mean | 8.6773 | 7.7410 | 18.1422 | 29.71% | 8.3442 | 1/24 |
| appearance_affine | 4.9555 | 4.3992 | 10.1019 | 5.28% | 4.5666 | 18/24 |
| color_mean | 5.7978 | 5.3023 | 11.5360 | 10.04% | 5.5455 | 7/24 |
| color_affine | 4.5860 | 3.9255 | 10.0157 | 5.07% | 4.2031 | 21/24 |

### Таблица 2

| Training bank / domain | System | Mean DeltaE00 | p95 | Mean at80% |
| --- | --- | --- | --- | --- |
| mixed /known | global_ridge | 4.5314 | 8.9563 | 4.3159 |
| mixed /known | global_mean | 9.9879 | 17.8445 | 10.4153 |
| mixed /known | appearance_mean | 7.8084 | 15.0530 | 7.8538 |
| mixed /known | appearance_affine | 4.1525 | 8.2099 | 3.9371 |
| mixed /known | color_mean | 4.5036 | 8.6499 | 4.3367 |
| mixed /known | color_affine | 3.9343 | 7.8965 | 3.7276 |
| from_SLR /known | global_ridge | 4.0025 | 8.7965 | 3.6429 |
| from_SLR /known | global_mean | 7.5679 | 12.6286 | 7.9645 |
| from_SLR /known | appearance_mean | 5.4123 | 10.3827 | 5.5126 |
| from_SLR /known | appearance_affine | 3.8133 | 8.0388 | 3.4656 |
| from_SLR /known | color_mean | 3.9949 | 8.3978 | 3.7553 |
| from_SLR /known | color_affine | 3.6365 | 7.2549 | 3.3071 |
| from_SLR /unseen | global_ridge | 8.8835 | 17.8859 | 8.9403 |
| from_SLR /unseen | global_mean | 9.3195 | 19.3989 | 9.4112 |
| from_SLR /unseen | appearance_mean | 8.3549 | 15.4807 | 8.0280 |
| from_SLR /unseen | appearance_affine | 9.6764 | 17.7088 | 9.4151 |
| from_SLR /unseen | color_mean | 7.3523 | 16.2273 | 8.3584 |
| from_SLR /unseen | color_affine | 7.1177 | 12.9255 | 7.3607 |
| from_ipod /known | global_ridge | 4.3910 | 9.1592 | 4.4145 |
| from_ipod /known | global_mean | 10.5246 | 20.7206 | 10.1553 |
| from_ipod /known | appearance_mean | 8.9164 | 16.4992 | 8.6943 |
| from_ipod /known | appearance_affine | 4.2187 | 8.7614 | 4.2316 |
| from_ipod /known | color_mean | 4.7983 | 10.5910 | 4.9151 |
| from_ipod /known | color_affine | 4.2610 | 9.3657 | 4.2341 |
| from_ipod /unseen | global_ridge | 7.9382 | 12.8151 | 8.3834 |
| from_ipod /unseen | global_mean | 11.9149 | 20.8247 | 10.7090 |
| from_ipod /unseen | appearance_mean | 6.6968 | 12.0623 | 7.2906 |
| from_ipod /unseen | appearance_affine | 8.2966 | 12.1561 | 8.4029 |
| from_ipod /unseen | color_mean | 8.7939 | 14.1091 | 9.1801 |
| from_ipod /unseen | color_affine | 7.4895 | 12.5179 | 7.9720 |

### Таблица 3

| Source protocol | Best system in this six-method phase | Earlier strong compact neural result |
| --- | --- | --- |
| Mixed known cameras | color_affine3.9343 | mixture3.4406 |
| SLR to unseen iPod | color_affine7.1177 | paired mixture4.8328 |
| iPod to unseen SLR | appearance_mean6.6968 | training-only graph4.9736 |
