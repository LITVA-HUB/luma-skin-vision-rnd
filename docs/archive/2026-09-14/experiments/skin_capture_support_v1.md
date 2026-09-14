# skin_capture_support_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Перемешивание реально наблюдавшихся патчей одного участка кожи; одинаковый Lab не означает регистрацию пикселей.

[Полная папка артефактов](../../../../docs/benchmarks/skin_capture_support_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_capture_support_v1/report.md)

SHA-256 отчёта: `3f470059dd90506b1d72514918654d6280b797b74c25d210d50ab1588023153c`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_capture_support_next_decision.md](../../../../docs/research/skin_capture_support_next_decision.md)
- [skin_capture_support_protocol_v1.md](../../../../docs/research/skin_capture_support_protocol_v1.md)
- [skin_capture_support_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_capture_support_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_capture_support_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_capture_support.py](../modules/scripts__skin_capture_support.md)
- [scripts/skin_capture_support_audit.py](../modules/scripts__skin_capture_support_audit.md)
- [scripts/skin_capture_support_report.py](../modules/scripts__skin_capture_support_report.md)
- [scripts/skin_capture_support_train.py](../modules/scripts__skin_capture_support_train.md)
- [scripts/skin_capture_support_verify.py](../modules/scripts__skin_capture_support_verify.md)
- [tests/test_skin_capture_support.py](../tests/tests__test_skin_capture_support.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_capture_support_v1/audit.json) | {"status": "PASS"} | `90c84c9161339a8e868bb1d86ca2030533274564067a0bd56a37c8c0bf32940e` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_capture_support_v1/audit_mixed.json) | {"status": "PASS"} | `7acfbd6f43f66f89b38e7a0343202e8eae2ba162287c50d83f5d9db3559f7d12` |
| [routing_probe.json](../../../../docs/benchmarks/skin_capture_support_v1/routing_probe.json) | {"scope": "Teacher-assisted TRAIN virtual-bag diagnostic; no model improvement claim"} | `713df1b1a5fae648624a7d59d61b1a1034f8c98788ba68a8f5c8598500ff38c6` |
| [routing_probe_lock.json](../../../../docs/benchmarks/skin_capture_support_v1/routing_probe_lock.json) | {} | `0ec612e6c06a7f244cd5933a3db0cb763066e5f1ebfb9ac016ea9495b682e086` |
| [source_lock.json](../../../../docs/benchmarks/skin_capture_support_v1/source_lock.json) | {} | `d88b1818a550c1646fd792de7921bb88400393ecd638b3482c94814c52e1be73` |
| [summary.json](../../../../docs/benchmarks/skin_capture_support_v1/summary.json) | {"scope": "REPRODUCED SOURCE; no new independent evaluation"} | `a10967c136f80e1625d4a6ff834c328ca5909e1d1c2f649812c95e14c89a0472` |
| [train_pair_audit.json](../../../../docs/benchmarks/skin_capture_support_v1/train_pair_audit.json) | {"scope": "TRAIN only; paired appearance diagnostics, no new image model results"} | `651956a16be25f1025b86b6071e4da8a4c88bc4f0add56613872fce69eb6e676` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Arm | Mean | Median | p95 | At80% |
| --- | --- | --- | --- | --- | --- |
| mixed | baseline | 3.4406 | 2.9776 | 6.9316 | 3.4407 |
| mixed | self_bootstrap | 3.4509 | 2.9476 | 7.2967 | 3.3923 |
| mixed | soft_mode_control | 3.4654 | 3.0605 | 7.2623 | 3.5426 |
| mixed | paired_union | 3.5614 | 3.0369 | 7.2256 | 3.5978 |
| mixed | paired_stratified | 3.5619 | 3.0658 | 7.2012 | 3.6018 |
| from_SLR | baseline | 5.0193 | 4.8340 | 9.7825 | 5.1087 |
| from_SLR | self_bootstrap | 5.0214 | 4.8120 | 9.8668 | 5.0660 |
| from_SLR | soft_mode_control | 4.9546 | 4.8344 | 9.5594 | 4.9891 |
| from_SLR | paired_union | 4.8620 | 4.6519 | 9.1931 | 4.8488 |
| from_SLR | paired_stratified | 4.8328 | 4.7462 | 9.2002 | 4.8525 |
| from_ipod | baseline | 5.9635 | 5.5527 | 11.0896 | 5.7914 |
| from_ipod | self_bootstrap | 5.8964 | 5.5129 | 10.9336 | 5.7661 |
| from_ipod | soft_mode_control | 6.1421 | 5.8015 | 10.9571 | 6.0659 |
| from_ipod | paired_union | 5.9336 | 5.6443 | 10.6749 | 5.7886 |
| from_ipod | paired_stratified | 5.9520 | 5.6234 | 10.7105 | 5.8597 |

### Таблица 2

| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |
| --- | --- | --- | --- | --- |
| mixed | self_bootstrap vs baseline | 0.0103 | [-0.0440,0.0602] | False |
| mixed | soft_mode_control vs baseline | 0.0248 | [-0.0423,0.0694] | False |
| mixed | paired_union vs baseline | 0.1208 | [0.0021,0.2400] | False |
| mixed | paired_union vs soft_mode_control | 0.0959 | [0.0041,0.1871] | False |
| mixed | paired_stratified vs soft_mode_control | 0.0965 | [-0.0499,0.2384] | False |
| mixed | paired_stratified vs paired_union | 0.0006 | [-0.0767,0.0827] | False |
| from_SLR | self_bootstrap vs baseline | 0.0021 | [-0.1574,0.1169] | False |
| from_SLR | soft_mode_control vs baseline | -0.0647 | [-0.3860,0.1923] | False |
| from_SLR | paired_union vs baseline | -0.1573 | [-0.6845,0.6614] | False |
| from_SLR | paired_union vs soft_mode_control | -0.0927 | [-0.4486,0.4691] | False |
| from_SLR | paired_stratified vs soft_mode_control | -0.1219 | [-0.4622,0.4276] | False |
| from_SLR | paired_stratified vs paired_union | -0.0292 | [-0.0415,-0.0136] | False |
| from_ipod | self_bootstrap vs baseline | -0.0671 | [-0.1820,0.0155] | True |
| from_ipod | soft_mode_control vs baseline | 0.1786 | [-0.1784,0.4951] | False |
| from_ipod | paired_union vs baseline | -0.0299 | [-0.5152,0.2358] | False |
| from_ipod | paired_union vs soft_mode_control | -0.2085 | [-0.3368,0.0166] | False |
| from_ipod | paired_stratified vs soft_mode_control | -0.1901 | [-0.3178,0.0308] | False |
| from_ipod | paired_stratified vs paired_union | 0.0184 | [-0.0124,0.0535] | False |

### Таблица 3

| Source model | Learned global | Known global | Weighted known global | Known local | Shuffled local |
| --- | --- | --- | --- | --- | --- |
| mixed__baseline__s17 | 3.9575 | 4.6686 | 4.6693 | 4.6734 | 4.6723 |
| mixed__baseline__s29 | 4.0116 | 5.1000 | 5.1256 | 5.0989 | 5.1048 |
| mixed__baseline__s43 | 4.3770 | 5.1368 | 5.1519 | 5.1506 | 5.1383 |
| mixed__paired_union__s17 | 3.5421 | 4.5388 | 4.7313 | 4.7114 | 4.5410 |
| mixed__paired_union__s29 | 3.4731 | 4.2153 | 4.3405 | 4.3280 | 4.2178 |
| mixed__paired_union__s43 | 3.5795 | 4.6140 | 4.7978 | 4.7802 | 4.6182 |
