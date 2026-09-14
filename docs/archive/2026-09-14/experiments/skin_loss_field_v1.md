# skin_loss_field_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Предсказывать поле цветовой ошибки вместо единственного ответа; 15 625 кандидатов не дали общей победы.

[Полная папка артефактов](../../../../docs/benchmarks/skin_loss_field_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_loss_field_v1/report.md)

SHA-256 отчёта: `db71c60ae5a6ede8b82ff7b7b2a2b7f61418ae532d9c990a54725a9aa76f3bcc`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_loss_field_protocol_v1.md](../../../../docs/research/skin_loss_field_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_loss_field_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_loss_field.py](../modules/scripts__skin_loss_field.md)
- [scripts/skin_loss_field_audit.py](../modules/scripts__skin_loss_field_audit.md)
- [scripts/skin_loss_field_palette_audit.py](../modules/scripts__skin_loss_field_palette_audit.md)
- [scripts/skin_loss_field_report.py](../modules/scripts__skin_loss_field_report.md)
- [scripts/skin_loss_field_train.py](../modules/scripts__skin_loss_field_train.md)
- [tests/test_skin_loss_field.py](../tests/tests__test_skin_loss_field.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_loss_field_v1/audit.json) | {"status": "PASS"} | `dae88b146707a07750189d07abd172302170e9dcb6b1c050e10afbe65119a403` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_loss_field_v1/audit_mixed.json) | {"status": "PASS"} | `ed33a62df91b4e39839c74df4e93b9593975bb8e86aa8c012abd0ea562193f35` |
| [palette_audit.json](../../../../docs/benchmarks/skin_loss_field_v1/palette_audit.json) | {"status": "PASS"} | `ae065ba95d33ec6b60816292b049fd20c5507f689dd0085ba0c50c8b0ea3b76f` |
| [palette_receipt.json](../../../../docs/benchmarks/skin_loss_field_v1/palette_receipt.json) | {} | `2c580dad6f2b116aa5203ae6618081e858caed19b8cbb7e43d0e0f9a862ceac3` |
| [source_lock.json](../../../../docs/benchmarks/skin_loss_field_v1/source_lock.json) | {} | `53f61fecfc51554ae46693ae572cc63f3a3a6d1b1095ba5696c98910b594dab7` |
| [summary.json](../../../../docs/benchmarks/skin_loss_field_v1/summary.json) | {"scope": "Repeated SOURCE development; no independent test"} | `9e0b2d6e9b07462e5d2e267ead96de846d3fb574067f3d71cf124ee6d62976d0` |
| [verification.json](../../../../docs/benchmarks/skin_loss_field_v1/verification.json) | {"passed": 327, "seconds": 35.6, "scope": "Full suite including2crossing and3field tests; scripts additionally run and independently audited"} | `d027bb1b9977cb0c711391f166c4ea85291cb8d7629f4e06db5c08d9b7800be6` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Arm | Primary mean | Median | p95 | At80% | Secondary mean | Negative risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | direct | 3.4406 | 2.9776 | 6.9316 | 3.4407 | 3.4406 | 0.00% |
| mixed | soft_ce | 3.7810 | 3.1755 | 7.9785 | 3.5943 | 3.6319 | 0.00% |
| mixed | risk_simplex | 4.0426 | 3.6861 | 8.2604 | 4.0284 | 3.6971 | 0.00% |
| mixed | risk_affine | 3.7598 | 3.3510 | 7.6275 | 3.7475 | 3.6247 | 0.00% |
| from_SLR | direct | 5.0193 | 4.8340 | 9.7825 | 5.1087 | 5.0193 | 0.00% |
| from_SLR | soft_ce | 5.5461 | 5.3680 | 10.3963 | 5.4338 | 5.6316 | 0.00% |
| from_SLR | risk_simplex | 5.6813 | 5.1236 | 11.0181 | 5.5767 | 5.4695 | 0.00% |
| from_SLR | risk_affine | 4.9641 | 4.2494 | 10.1217 | 4.9285 | 5.0315 | 0.00% |
| from_ipod | direct | 5.9635 | 5.5527 | 11.0896 | 5.7914 | 5.9635 | 0.00% |
| from_ipod | soft_ce | 7.4636 | 7.4083 | 12.2443 | 7.5342 | 6.8011 | 0.00% |
| from_ipod | risk_simplex | 9.5897 | 8.6170 | 18.7014 | 9.8045 | 8.8714 | 0.00% |
| from_ipod | risk_affine | 8.9582 | 8.8379 | 15.4983 | 9.1635 | 8.2969 | 0.00% |

### Таблица 2

| Protocol | Candidate vs control | Mean difference | Patient95% interval | All3seeds |
| --- | --- | --- | --- | --- |
| mixed | risk_simplex vs soft_ce | 0.2616 | [-0.0324,0.4731] | False |
| mixed | risk_affine vs soft_ce | -0.0212 | [-0.4137,0.3948] | False |
| mixed | risk_affine vs direct | 0.3192 | [-0.0067,0.6266] | False |
| from_SLR | risk_simplex vs soft_ce | 0.1352 | [-0.2456,0.3710] | False |
| from_SLR | risk_affine vs soft_ce | -0.5820 | [-0.9001,-0.1265] | True |
| from_SLR | risk_affine vs direct | -0.0552 | [-1.1498,0.5679] | False |
| from_ipod | risk_simplex vs soft_ce | 2.1261 | [0.7878,3.1179] | False |
| from_ipod | risk_affine vs soft_ce | 1.4945 | [0.4661,2.1933] | False |
| from_ipod | risk_affine vs direct | 2.9947 | [0.2548,4.8980] | False |

### Таблица 3

| Protocol | Direct projected mean | Known-target oracle mean | Oracle p95 |
| --- | --- | --- | --- |
| mixed | 3.5335 | 0.7264 | 1.1592 |
| from_SLR | 5.0652 | 0.7121 | 1.0729 |
| from_ipod | 5.9965 | 0.6926 | 1.1732 |
