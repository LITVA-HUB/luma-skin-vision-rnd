# skin_teacher_readout_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Frozen foundation features, ridge и shuffled controls: перенос семантики не стал универсальной колориметрией.

[Полная папка артефактов](../../../../docs/benchmarks/skin_teacher_readout_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_teacher_readout_v1/report.md)

SHA-256 отчёта: `f907376b973410bda07ebe455caea02a019df6342ab0da898385afeef23eae2a`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_teacher_readout_protocol_v1.md](../../../../docs/research/skin_teacher_readout_protocol_v1.md)
- [skin_teacher_readout_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_teacher_readout_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_teacher_readout_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_teacher_readout.py](../modules/scripts__skin_teacher_readout.md)
- [scripts/skin_teacher_readout_audit.py](../modules/scripts__skin_teacher_readout_audit.md)
- [scripts/skin_teacher_readout_report.py](../modules/scripts__skin_teacher_readout_report.md)
- [tests/test_skin_teacher_readout.py](../tests/tests__test_skin_teacher_readout.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_teacher_readout_v1/audit.json) | {"status": "PASS"} | `c979664e8780ef8e6dde3a294c73c81f05d21e80939314f59e3952c57ee84d0b` |
| [feature_audit.json](../../../../docs/benchmarks/skin_teacher_readout_v1/feature_audit.json) | {"status": "PASS", "seconds": 3.670590699999593} | `7745b77349e609271075728ad51c496976d197897f76fa8ece56a79aa19a4a2a` |
| [feature_receipt.json](../../../../docs/benchmarks/skin_teacher_readout_v1/feature_receipt.json) | {"scope": "SOURCE FEATURES ONLY; no fine-tuning or reserved endpoints", "seconds": 3.658660699991742} | `c3a329c926e1c475281f1173b4fa08aa01ef764694c1f7100b7de5ed68958d54` |
| [readout_lock.json](../../../../docs/benchmarks/skin_teacher_readout_v1/readout_lock.json) | {"scope": "SOURCE ONLY; pretraining overlap unknown"} | `bfac429526bf523793bbc1b0f919e9e34a6f6624e073965dcb1a4b68af310985` |
| [summary.json](../../../../docs/benchmarks/skin_teacher_readout_v1/summary.json) | {"scope": "SOURCE READOUT SCREEN; no deployable compact model or independent result"} | `443c4ae889656c0a2c18a5622816895b6aadfca31276d2a607408da4cf412b5d` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Features | Selected alpha | Mean DeltaE00 | Median | p95 | Mean at 80% |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | color | 0.01 | 4.5139 | 4.0781 | 8.7596 | 4.3072 |
| mixed | teacher | 1 | 4.2651 | 3.7437 | 8.4045 | 4.2112 |
| mixed | combined | 1 | 3.9462 | 3.5933 | 7.5969 | 3.8573 |
| mixed | shuffle17 | 10 | 4.7328 | 4.4084 | 8.7900 | 4.6604 |
| mixed | shuffle29 | 10 | 4.7756 | 4.3495 | 8.6209 | 4.6722 |
| mixed | shuffle43 | 1 | 4.6024 | 4.0553 | 8.6708 | 4.6281 |
| from_SLR | color | 0.1 | 7.8220 | 7.0267 | 15.7958 | 8.0710 |
| from_SLR | teacher | 1 | 6.3118 | 4.9849 | 13.8034 | 6.4218 |
| from_SLR | combined | 1 | 5.6611 | 4.5200 | 13.5414 | 6.0811 |
| from_SLR | shuffle17 | 1 | 5.8550 | 5.3907 | 12.2588 | 5.8438 |
| from_SLR | shuffle29 | 1 | 6.0047 | 5.5173 | 12.0678 | 5.8210 |
| from_SLR | shuffle43 | 1 | 6.0191 | 5.5158 | 11.7630 | 5.6697 |
| from_ipod | color | 0.01 | 8.2597 | 8.3407 | 13.2100 | 8.6911 |
| from_ipod | teacher | 1 | 7.5572 | 7.2079 | 12.5224 | 7.5507 |
| from_ipod | combined | 1 | 8.2684 | 8.0297 | 14.6784 | 8.3628 |
| from_ipod | shuffle17 | 10 | 6.1725 | 5.8452 | 11.6615 | 6.6398 |
| from_ipod | shuffle29 | 10 | 5.8540 | 5.6859 | 10.4742 | 6.4064 |
| from_ipod | shuffle43 | 10 | 6.0944 | 5.8630 | 11.6828 | 6.6934 |

### Таблица 2

| Protocol | Strong historical compact model | Mean over three seed scores |
| --- | --- | --- |
| mixed | plain_mse | 3.4771 |
| mixed | mixture_mse | 3.4406 |
| mixed | graph_always | 3.6394 |
| from_SLR | plain_mse | 5.8301 |
| from_SLR | mixture_mse | 5.0193 |
| from_SLR | graph_always | 5.8339 |
| from_ipod | plain_mse | 5.5089 |
| from_ipod | mixture_mse | 5.9635 |
| from_ipod | graph_always | 4.9736 |
