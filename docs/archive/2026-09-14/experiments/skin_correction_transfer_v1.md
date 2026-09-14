# skin_correction_transfer_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Финальная проверка: все три correction head ухудшили оба unseen-camera направления. Исследования остановлены.

[Полная папка артефактов](../../../../docs/benchmarks/skin_correction_transfer_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_correction_transfer_v1/report.md)

SHA-256 отчёта: `727252b5f6473092250d0519ecc4ecdf1bd4920afec0bf76fdab30b70a626323`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_correction_transfer_batch_recovery.md](../../../../docs/research/skin_correction_transfer_batch_recovery.md)
- [skin_correction_transfer_next_decision.md](../../../../docs/research/skin_correction_transfer_next_decision.md)
- [skin_correction_transfer_protocol_v1.md](../../../../docs/research/skin_correction_transfer_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_correction_transfer_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_correction_transfer.py](../modules/scripts__skin_correction_transfer.md)
- [scripts/skin_correction_transfer_report.py](../modules/scripts__skin_correction_transfer_report.md)
- [scripts/skin_correction_transfer_verify.py](../modules/scripts__skin_correction_transfer_verify.md)
- [tests/test_skin_correction_transfer.py](../tests/tests__test_skin_correction_transfer.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_correction_transfer_v1/audit.json) | {"status": "PASS"} | `0f1448a287a9b58ae57488abfb6a7022c52431dc2db4138f02f248b098931ca9` |
| [recovery_lock.json](../../../../docs/benchmarks/skin_correction_transfer_v1/recovery_lock.json) | {} | `1af5f9b46bf262d32d5190fbc17e614b69bb55c66b093da60c29faab7d683ba4` |
| [results.json](../../../../docs/benchmarks/skin_correction_transfer_v1/results.json) | {"status": "COMPLETE"} | `d7f28f32d30fcd46ecf8aa01f59f98c6e0b9f575bd64231451bf67ecf6ebf989` |
| [source_lock.json](../../../../docs/benchmarks/skin_correction_transfer_v1/source_lock.json) | {} | `6a9199d72ca7f9948131ef92d851e5ccaef496f59fbd7e6a7fb246d4b1bf784f` |
| [summary.json](../../../../docs/benchmarks/skin_correction_transfer_v1/summary.json) | {} | `d2f69f0d2a0699aeb0633cc0c9b0bbb0d478e97c9cef7d45bff71bc15a4b76cb` |
| [test_receipt.json](../../../../docs/benchmarks/skin_correction_transfer_v1/test_receipt.json) | {"scope": "Full repository suite after the final correction-transfer audit", "passed": 379, "status": "Terminal run observed before research publication; not rerun for this receipt"} | `2070f5316b5c045b3eed844d385cfb3d8bc1b420a14a88d9a260c45cf81e76f6` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Training / evaluation | Arm | Mean DeltaE00 | Median | p95 | Error >10 | Mean at80% |
| --- | --- | --- | --- | --- | --- | --- |
| mixed /known | base | 3.8277 | 3.4822 | 7.9298 | 1.14% | 3.6464 |
| mixed /known | in_full | 3.7626 | 3.3775 | 7.9668 | 1.26% | 3.5790 |
| mixed /known | in_matched | 3.8184 | 3.4257 | 7.9302 | 1.39% | 3.6254 |
| mixed /known | out_person | 3.9889 | 3.6233 | 7.6789 | 1.01% | 3.8151 |
| from_SLR /known | base | 3.4210 | 3.2170 | 6.4298 | 0.00% | 3.2236 |
| from_SLR /known | in_full | 3.5002 | 3.2644 | 6.8836 | 0.51% | 3.2342 |
| from_SLR /known | in_matched | 3.5041 | 3.3133 | 6.3971 | 0.00% | 3.2683 |
| from_SLR /known | out_person | 3.5168 | 2.9690 | 7.0893 | 0.00% | 3.3021 |
| from_SLR /unseen | base | 5.4877 | 5.2429 | 10.3253 | 6.57% | 5.4344 |
| from_SLR /unseen | in_full | 5.9953 | 5.9363 | 11.3036 | 10.86% | 5.8908 |
| from_SLR /unseen | in_matched | 5.7371 | 5.4933 | 10.7327 | 7.58% | 5.6632 |
| from_SLR /unseen | out_person | 6.0910 | 5.8064 | 11.1892 | 11.11% | 6.3439 |
| from_ipod /known | base | 4.0558 | 3.2771 | 9.2260 | 2.27% | 4.0294 |
| from_ipod /known | in_full | 3.9347 | 3.2037 | 8.7666 | 1.77% | 3.9080 |
| from_ipod /known | in_matched | 3.9810 | 3.3729 | 8.7457 | 1.26% | 3.9246 |
| from_ipod /known | out_person | 3.9524 | 3.5089 | 8.4998 | 2.27% | 3.9364 |
| from_ipod /unseen | base | 6.5602 | 6.2911 | 10.7986 | 7.07% | 6.5215 |
| from_ipod /unseen | in_full | 7.2462 | 6.8153 | 12.7450 | 17.42% | 7.8330 |
| from_ipod /unseen | in_matched | 7.2908 | 7.0333 | 12.2689 | 15.91% | 7.7503 |
| from_ipod /unseen | out_person | 7.6725 | 7.2741 | 12.7549 | 21.21% | 8.1845 |
