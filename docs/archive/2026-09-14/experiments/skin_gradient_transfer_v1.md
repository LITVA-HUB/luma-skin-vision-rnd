# skin_gradient_transfer_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Градиенты людей, перемешанные группы и малые шаги: уменьшение TRAIN loss не гарантирует уменьшения ΔE00.

[Полная папка артефактов](../../../../docs/benchmarks/skin_gradient_transfer_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_gradient_transfer_v1/report.md)

SHA-256 отчёта: `3ce8b13fc262a968c1cc83f35511730b3f56a42eee3b5c327ac32cff670477de`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_gradient_transfer_provenance.md](../../../../docs/ip/skin_gradient_transfer_provenance.md)
- [skin_gradient_transfer_next_decision.md](../../../../docs/research/skin_gradient_transfer_next_decision.md)
- [skin_gradient_transfer_protocol_v1.md](../../../../docs/research/skin_gradient_transfer_protocol_v1.md)
- [skin_gradient_transfer_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_gradient_transfer_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_gradient_transfer_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_gradient_transfer.py](../modules/scripts__skin_gradient_transfer.md)
- [scripts/skin_gradient_transfer_report.py](../modules/scripts__skin_gradient_transfer_report.md)
- [scripts/skin_gradient_transfer_run.py](../modules/scripts__skin_gradient_transfer_run.md)
- [scripts/skin_gradient_transfer_verify.py](../modules/scripts__skin_gradient_transfer_verify.md)
- [tests/test_skin_gradient_transfer.py](../tests/tests__test_skin_gradient_transfer.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/audit.json) | {"status": "PASS"} | `fce02d6821f3c7c8960e405ebd1fea903c8866338a5a146465fb283f4ac35fac` |
| [from_ipod__image__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/from_ipod__image__s17.json) | {} | `bb7877cba9aeabe6c79b640529c42447b54f03a594e0457d14b4c92844cf8f2d` |
| [from_ipod__person_color__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/from_ipod__person_color__s17.json) | {} | `7fb1bc95ff86d6038b494cbc608af15f789a488e7d3456ccbb482692d19fdf00` |
| [from_SLR__image__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/from_SLR__image__s17.json) | {} | `3942466650bf3af81b592427421f12170f9d8eacb2b5105b2b9bc89c3550cb9b` |
| [from_SLR__person_color__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/from_SLR__person_color__s17.json) | {} | `9fe0382dfe451b11c1f8299c8816513be599d166309274204935747c3a230f22` |
| [mixed__image__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/mixed__image__s17.json) | {} | `89e241b49d1a186ac4a61d883160000656803d17fd61d9fa0c2fbd8d0dae0c18` |
| [mixed__person_color__s17.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/mixed__person_color__s17.json) | {} | `505d35122bc75f3d903b1a0c2e9508ac80e7a01dc0ba52fc98dbd2fe415250ce` |
| [results.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/results.json) | {} | `398ced5ebc398edd9a17df5ef4a4b09681faa9bcb5e49b476fb7eeb9edb4953e` |
| [source_lock.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/source_lock.json) | {} | `26e6cb3c78dde3680f466855bece48f9156b5fbcd3c1e411dd624a038754dd3d` |
| [summary.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/summary.json) | {} | `ed48ad005b582a70e09843515787af660ae6451dfd31fca6abce7fa313b656d0` |
| [test_receipt.json](../../../../docs/benchmarks/skin_gradient_transfer_v1/test_receipt.json) | {"passed": 364, "seconds": 33.66} | `74913da4d53c3e71243d9532633997b59b5f7f6b5288e69e647f2f97ebe279cb` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Frozen seed17 model | True-person cosine | Shuffled cosine range | True negative pairs | Pooled color/mode cosine | Weighted auxiliary/color norm |
| --- | --- | --- | --- | --- | --- |
| mixed__image__s17 | 0.0164 | 0.1411 to0.2108 | 50.72% | 0.5106 | 0.0423 |
| mixed__person_color__s17 | -0.0041 | 0.0671 to0.1261 | 48.55% | 0.0244 | 0.0696 |
| from_SLR__image__s17 | -0.1155 | -0.1290 to-0.1159 | 57.14% | 0.1074 | 0.3262 |
| from_SLR__person_color__s17 | -0.0582 | -0.0909 to-0.0704 | 64.29% | 0.7539 | 0.2441 |
| from_ipod__image__s17 | -0.0242 | -0.0546 to-0.0167 | 51.67% | 0.3082 | 0.1319 |
| from_ipod__person_color__s17 | -0.0419 | -0.0418 to-0.0066 | 55.00% | 0.4890 | 0.3310 |

### Таблица 2

| Parameter step length | Maximum first-order remainder | Median relative remainder | Mean skin DeltaE00 change range |
| --- | --- | --- | --- |
| 0.0001 | 4.4313187e-06 | 0.005137 | -0.000656 to+0.000634 |
| 0.001 | 0.00044189211 | 0.051206 | -0.005714 to+0.007174 |
