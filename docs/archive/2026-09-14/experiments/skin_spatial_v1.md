# skin_spatial_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Свёрточные и recurrent graph ветви с геометрическими контролями. Дополнительные проходы не дали общего выигрыша.

[Полная папка артефактов](../../../../docs/benchmarks/skin_spatial_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_spatial_v1/report.md)

SHA-256 отчёта: `cc804d9332c8e3736d1a229edd36465abb19e05d74c76f38ffcbf36dd82d36ae`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_spatial_next_decision.md](../../../../docs/research/skin_spatial_next_decision.md)
- [skin_spatial_protocol_v1.md](../../../../docs/research/skin_spatial_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_spatial_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_spatial_audit.py](../modules/scripts__skin_spatial_audit.md)
- [scripts/skin_spatial_model.py](../modules/scripts__skin_spatial_model.md)
- [scripts/skin_spatial_offset_probe.py](../modules/scripts__skin_spatial_offset_probe.md)
- [scripts/skin_spatial_profile.py](../modules/scripts__skin_spatial_profile.md)
- [scripts/skin_spatial_recurrence_probe.py](../modules/scripts__skin_spatial_recurrence_probe.md)
- [scripts/skin_spatial_report.py](../modules/scripts__skin_spatial_report.md)
- [scripts/skin_spatial_train.py](../modules/scripts__skin_spatial_train.md)
- [tests/test_skin_spatial.py](../tests/tests__test_skin_spatial.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_spatial_v1/audit.json) | {"status": "PASS"} | `8c73db1b68ddcc891b02102c92b5dcd88faf31d4dc3f6ca4cfa70755cae83a13` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_spatial_v1/audit_mixed.json) | {"status": "PASS"} | `3fb1e6dba9574e1b8f28e6214423ddf36acd20cd3b1455bd435407d699fcc6f0` |
| [offset_probe.json](../../../../docs/benchmarks/skin_spatial_v1/offset_probe.json) | {"scope": "POST-DISCOVERY source opponent control; fixed offset from fitting predictions only; no independent test claim"} | `6bb2fd780a2efe01576e1f799b430443332f32d550e3bc15a30636ed8eb6f0c5` |
| [profile.json](../../../../docs/benchmarks/skin_spatial_v1/profile.json) | {"scope": "CUDA event batch1 prepared-token nativeLab output; excludes JPEG/resize/features/CPU/risk/localization"} | `d183ab12a9113226b7913d9aeb8d90cd48792490c38956e7d5ea46784edcdc8e` |
| [recurrence_probe.json](../../../../docs/benchmarks/skin_spatial_v1/recurrence_probe.json) | {"scope": "POST-SCREEN source diagnostic; all steps0/1/3/8 kept; no new best-epoch/model selection or reserved endpoints"} | `57d26d1c5a49ac9624a7f1a88cd67974b4e3533fe562964c9f6eb6c55ffa4171` |
| [source_lock.json](../../../../docs/benchmarks/skin_spatial_v1/source_lock.json) | {} | `f109f9553fa880c4d33b8acd5896adeb1b379574c2d0355dea57aa31670d8512` |
| [summary.json](../../../../docs/benchmarks/skin_spatial_v1/summary.json) | {"scope": "Source exploration; same seed score averaging, no ensemble or independent test claim"} | `c0c8445e028d0326cc5fb6ce59e587052125f93493fd9313e6385a00c4e448f2` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Method | Mean | Mean at80% | p95 | Seed means |
| --- | --- | --- | --- | --- | --- |
| mixed | plain | 3.4855 | 3.4300 | 7.1819 | 3.4851, 3.5127, 3.4587 |
| mixed | conv1 | 3.4756 | 3.4063 | 7.1453 | 3.4870, 3.5036, 3.4361 |
| mixed | conv3 | 3.4901 | 3.4532 | 7.2079 | 3.4654, 3.5320, 3.4728 |
| mixed | graph1 | 3.6142 | 3.4955 | 7.5631 | 3.6740, 3.5900, 3.5787 |
| mixed | graph3 | 3.6314 | 3.5191 | 7.9576 | 3.6812, 3.6312, 3.5818 |
| mixed | graph3_scrambled | 3.5430 | 3.5256 | 7.3092 | 3.5470, 3.5457, 3.5363 |
| from_SLR | plain | 5.6063 | 5.6837 | 10.9750 | 5.3482, 5.4913, 5.9794 |
| from_SLR | conv1 | 6.3139 | 6.0779 | 12.2734 | 5.6139, 7.0013, 6.3263 |
| from_SLR | conv3 | 6.6501 | 6.1535 | 13.1755 | 5.9482, 6.8641, 7.1381 |
| from_SLR | graph1 | 6.6681 | 6.6331 | 13.1034 | 6.3747, 6.9272, 6.7025 |
| from_SLR | graph3 | 6.3611 | 5.9568 | 13.7110 | 6.2429, 6.1289, 6.7117 |
| from_SLR | graph3_scrambled | 6.4279 | 6.1696 | 12.2903 | 5.4941, 7.2294, 6.5603 |
| from_ipod | plain | 6.1399 | 6.0788 | 10.7483 | 5.8650, 6.3605, 6.1943 |
| from_ipod | conv1 | 5.9123 | 5.8197 | 10.5396 | 5.8414, 6.2994, 5.5962 |
| from_ipod | conv3 | 5.7738 | 5.7106 | 10.5538 | 5.6855, 6.0182, 5.6176 |
| from_ipod | graph1 | 5.8708 | 5.8860 | 10.3710 | 6.0947, 5.8317, 5.6860 |
| from_ipod | graph3 | 5.8068 | 5.7015 | 10.4904 | 6.0462, 5.8499, 5.5243 |
| from_ipod | graph3_scrambled | 5.9265 | 5.9050 | 10.8005 | 6.0038, 6.0226, 5.7530 |

### Таблица 2

| Method | Stored parameters | Active parameters | Max fit+selection VRAM MiB |
| --- | --- | --- | --- |
| plain | 993287 | 924932 | 108.62 |
| conv1 | 993287 | 993028 | 117.40 |
| conv3 | 993287 | 993028 | 125.05 |
| graph1 | 993287 | 990727 | 119.41 |
| graph3 | 993287 | 990727 | 127.56 |
| graph3_scrambled | 993287 | 990727 | 126.66 |
