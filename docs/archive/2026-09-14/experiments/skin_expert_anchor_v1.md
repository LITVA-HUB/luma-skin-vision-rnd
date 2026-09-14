# skin_expert_anchor_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Удаление и явная супервизия экспертов; отдельный forward gain, общей победы нет.

[Полная папка артефактов](../../../../docs/benchmarks/skin_expert_anchor_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_expert_anchor_v1/report.md)

SHA-256 отчёта: `4754d87e950844b4666e9112ddbe7d66372b6d80cbaf259d63f0d0debb79f0dd`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_expert_anchor_next_decision.md](../../../../docs/research/skin_expert_anchor_next_decision.md)
- [skin_expert_anchor_protocol_v1.md](../../../../docs/research/skin_expert_anchor_protocol_v1.md)
- [skin_expert_anchor_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_expert_anchor_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_expert_anchor_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_expert_anchor.py](../modules/scripts__skin_expert_anchor.md)
- [scripts/skin_expert_anchor_report.py](../modules/scripts__skin_expert_anchor_report.md)
- [scripts/skin_expert_anchor_train.py](../modules/scripts__skin_expert_anchor_train.md)
- [scripts/skin_expert_anchor_verify.py](../modules/scripts__skin_expert_anchor_verify.md)
- [tests/test_skin_expert_anchor.py](../tests/tests__test_skin_expert_anchor.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_expert_anchor_v1/audit.json) | {"status": "PASS"} | `4ca09365ea06e00ade210121960f26d4fa7727635b2f66888a3c24324d355b86` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_expert_anchor_v1/audit_mixed.json) | {"status": "PASS"} | `eac1cd9e2b611452935ed56460d54b6f80fc31d2a1d4edfab2e07255d2492cf8` |
| [source_lock.json](../../../../docs/benchmarks/skin_expert_anchor_v1/source_lock.json) | {} | `bd172cfd9b1093deb53fd99a1202e82a03c0c290b5f95f7e54de84f93a8bb95e` |
| [summary.json](../../../../docs/benchmarks/skin_expert_anchor_v1/summary.json) | {"scope": "Locally reproduced exploratory source data; no new independent test"} | `50873c8eb955ef5f4804ef176f8c330a6730accde923dea5f2b4cd7ea9df74fa` |
| [test_receipt.json](../../../../docs/benchmarks/skin_expert_anchor_v1/test_receipt.json) | {"passed": 337, "seconds": 40.02} | `3d9bd152e05f159880576c7255d79c5e0125ced1ab9b92c9e49aa0e0665e9a8d` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Arm | Mean | Median | p95 | Mean at 80% |
| --- | --- | --- | --- | --- | --- |
| mixed | baseline_raw | 3.4406 | 2.9776 | 6.9316 | 3.3631 |
| mixed | baseline_paired | 3.5619 | 3.0658 | 7.2012 | 3.4679 |
| mixed | plain_raw | 3.4770 | 2.9910 | 7.0725 | 3.4162 |
| mixed | plain_paired | 3.5436 | 3.0482 | 7.1905 | 3.4722 |
| mixed | uniform_anchor_raw | 3.4755 | 2.9683 | 7.0491 | 3.4225 |
| mixed | uniform_anchor_paired | 3.5383 | 3.1241 | 7.1313 | 3.4445 |
| mixed | conditional_raw | 3.5004 | 2.9766 | 7.0508 | 3.3989 |
| mixed | conditional_paired | 3.6070 | 3.2080 | 7.2163 | 3.5151 |
| from_SLR | baseline_raw | 5.0193 | 4.8340 | 9.7825 | 5.1031 |
| from_SLR | baseline_paired | 4.8328 | 4.7462 | 9.2002 | 4.8112 |
| from_SLR | plain_raw | 5.8301 | 5.4395 | 11.2459 | 5.3698 |
| from_SLR | plain_paired | 5.4150 | 4.9757 | 10.8473 | 4.9554 |
| from_SLR | uniform_anchor_raw | 5.6433 | 5.4743 | 10.4964 | 5.3540 |
| from_SLR | uniform_anchor_paired | 5.2683 | 4.9777 | 10.1849 | 5.0154 |
| from_SLR | conditional_raw | 6.2317 | 6.0018 | 11.6676 | 6.2160 |
| from_SLR | conditional_paired | 5.3966 | 5.1289 | 9.8849 | 5.1184 |
| from_ipod | baseline_raw | 5.9635 | 5.5527 | 11.0896 | 6.1970 |
| from_ipod | baseline_paired | 5.9520 | 5.6234 | 10.7105 | 6.0121 |
| from_ipod | plain_raw | 5.5087 | 5.3784 | 9.4012 | 5.8382 |
| from_ipod | plain_paired | 5.3047 | 5.1265 | 9.5612 | 5.5999 |
| from_ipod | uniform_anchor_raw | 6.7628 | 6.3749 | 11.6373 | 7.2334 |
| from_ipod | uniform_anchor_paired | 5.7660 | 5.6213 | 9.8701 | 6.0936 |
| from_ipod | conditional_raw | 6.1566 | 5.8291 | 10.4416 | 6.5216 |
| from_ipod | conditional_paired | 5.6578 | 5.5357 | 10.0006 | 6.0675 |

### Таблица 2

| Protocol | Candidate vs control | Mean difference | Patient 95% interval | All 3 seeds |
| --- | --- | --- | --- | --- |
| mixed | baseline_paired vs baseline_raw | 0.1213 | [-0.0416, 0.2869] | False |
| mixed | plain_raw vs baseline_raw | 0.0364 | [-0.0886, 0.1456] | False |
| mixed | plain_paired vs baseline_raw | 0.1030 | [-0.0987, 0.3114] | False |
| mixed | uniform_anchor_raw vs baseline_raw | 0.0350 | [-0.0474, 0.1294] | False |
| mixed | uniform_anchor_paired vs baseline_raw | 0.0977 | [-0.0293, 0.2341] | False |
| mixed | conditional_raw vs baseline_raw | 0.0598 | [-0.0368, 0.1326] | False |
| mixed | conditional_paired vs baseline_raw | 0.1664 | [0.0654, 0.2760] | False |
| mixed | conditional_raw vs uniform_anchor_raw | 0.0248 | [-0.0614, 0.1187] | False |
| mixed | conditional_paired vs uniform_anchor_paired | 0.0687 | [0.0307, 0.1031] | False |
| mixed | plain_paired vs plain_raw | 0.0667 | [-0.1283, 0.2617] | False |
| mixed | uniform_anchor_paired vs uniform_anchor_raw | 0.0628 | [0.0047, 0.1183] | False |
| mixed | conditional_paired vs conditional_raw | 0.1067 | [0.0264, 0.1727] | False |
| from_SLR | baseline_paired vs baseline_raw | -0.1865 | [-0.7170, 0.6199] | False |
| from_SLR | plain_raw vs baseline_raw | 0.8108 | [0.0094, 1.9716] | False |
| from_SLR | plain_paired vs baseline_raw | 0.3957 | [-0.5793, 1.8465] | False |
| from_SLR | uniform_anchor_raw vs baseline_raw | 0.6240 | [0.1021, 1.4576] | False |
| from_SLR | uniform_anchor_paired vs baseline_raw | 0.2490 | [-0.5710, 1.5130] | False |
| from_SLR | conditional_raw vs baseline_raw | 1.2124 | [-0.1443, 2.2976] | False |
| from_SLR | conditional_paired vs baseline_raw | 0.3773 | [-0.7681, 2.0530] | False |
| from_SLR | conditional_raw vs uniform_anchor_raw | 0.5885 | [-0.2464, 1.9854] | False |
| from_SLR | conditional_paired vs uniform_anchor_paired | 0.1283 | [-0.1971, 0.5400] | False |
| from_SLR | plain_paired vs plain_raw | -0.4151 | [-0.5887, -0.1252] | True |
| from_SLR | uniform_anchor_paired vs uniform_anchor_raw | -0.3750 | [-0.6731, 0.0553] | False |
| from_SLR | conditional_paired vs conditional_raw | -0.8352 | [-2.4507, 0.5690] | True |
| from_ipod | baseline_paired vs baseline_raw | -0.0115 | [-0.4616, 0.2500] | False |
| from_ipod | plain_raw vs baseline_raw | -0.4547 | [-0.8051, -0.1313] | True |
| from_ipod | plain_paired vs baseline_raw | -0.6588 | [-0.9404, -0.3574] | True |
| from_ipod | uniform_anchor_raw vs baseline_raw | 0.7993 | [-0.0144, 1.6332] | False |
| from_ipod | uniform_anchor_paired vs baseline_raw | -0.1974 | [-0.7666, 0.2414] | False |
| from_ipod | conditional_raw vs baseline_raw | 0.1931 | [-0.4204, 0.7484] | False |
| from_ipod | conditional_paired vs baseline_raw | -0.3057 | [-0.9538, 0.1961] | True |
| from_ipod | conditional_raw vs uniform_anchor_raw | -0.6062 | [-0.8847, -0.4060] | True |
| from_ipod | conditional_paired vs uniform_anchor_paired | -0.1083 | [-0.1873, -0.0454] | False |
| from_ipod | plain_paired vs plain_raw | -0.2040 | [-0.2507, -0.1353] | True |
| from_ipod | uniform_anchor_paired vs uniform_anchor_raw | -0.9968 | [-1.3918, -0.7522] | True |
| from_ipod | conditional_paired vs conditional_raw | -0.4988 | [-0.5524, -0.4107] | False |

### Таблица 3

| Protocol / arm | Deployed prediction mean | Label-assisted head mean | Mode accuracy |
| --- | --- | --- | --- |
| mixed/baseline_raw | 3.4406 | 4.3679 | 0.6048 |
| mixed/baseline_paired | 3.5619 | 4.4928 | 0.5833 |
| mixed/uniform_anchor_raw | 3.4755 | 3.5182 | 0.6477 |
| mixed/uniform_anchor_paired | 3.5383 | 3.5536 | 0.6326 |
| mixed/conditional_raw | 3.5004 | 3.4865 | 0.7184 |
| mixed/conditional_paired | 3.6070 | 3.4280 | 0.6414 |
| from_SLR/baseline_raw | 5.0193 | 7.5998 | 0.4495 |
| from_SLR/baseline_paired | 4.8328 | 6.5957 | 0.4268 |
| from_SLR/uniform_anchor_raw | 5.6433 | 5.6797 | 0.4015 |
| from_SLR/uniform_anchor_paired | 5.2683 | 5.2952 | 0.3838 |
| from_SLR/conditional_raw | 6.2317 | 6.6371 | 0.3813 |
| from_SLR/conditional_paired | 5.3966 | 6.2591 | 0.4091 |
| from_ipod/baseline_raw | 5.9635 | 5.8697 | 0.4268 |
| from_ipod/baseline_paired | 5.9520 | 5.6167 | 0.4091 |
| from_ipod/uniform_anchor_raw | 6.7628 | 6.9307 | 0.4495 |
| from_ipod/uniform_anchor_paired | 5.7660 | 5.5997 | 0.5278 |
| from_ipod/conditional_raw | 6.1566 | 6.7652 | 0.4394 |
| from_ipod/conditional_paired | 5.6578 | 5.7853 | 0.4495 |
