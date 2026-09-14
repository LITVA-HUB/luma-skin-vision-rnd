# chromaseed_neural_readout_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_neural_readout_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_neural_readout_v1/report.md)

SHA-256 отчёта: `639579df4a6f4585f76a31de235346a29595e2966b50f76e8c0e26dd1b1799ad`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_neural_readout_model_card.md](../../../../docs/architecture/chromaseed_neural_readout_model_card.md)
- [chromaseed_neural_readout_next_decision.md](../../../../docs/research/chromaseed_neural_readout_next_decision.md)
- [chromaseed_neural_readout_v1_protocol.md](../../../../docs/research/chromaseed_neural_readout_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_neural_readout_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_neural_readout_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_neural_readout.py](../modules/scripts__chromaseed_neural_readout.md)
- [scripts/chromaseed_neural_readout_audit.py](../modules/scripts__chromaseed_neural_readout_audit.md)
- [scripts/chromaseed_neural_readout_reference.py](../modules/scripts__chromaseed_neural_readout_reference.md)
- [scripts/chromaseed_neural_readout_report.py](../modules/scripts__chromaseed_neural_readout_report.md)
- [scripts/chromaseed_neural_readout_runtime.py](../modules/scripts__chromaseed_neural_readout_runtime.md)
- [scripts/chromaseed_neural_readout_train.py](../modules/scripts__chromaseed_neural_readout_train.md)
- [scripts/chromaseed_neural_readout_verify.py](../modules/scripts__chromaseed_neural_readout_verify.md)
- [tests/test_chromaseed_neural_readout.py](../tests/tests__test_chromaseed_neural_readout.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_neural_readout_v1/audit.json) | {"passed": true} | `58e5c548d86cf33dd644e40f956d06ed1dca8094a96eb26f333877883ed5c9e5` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_neural_readout_v1/runtime.json) | {"scope": "381 actual one-row consumers,20 warmups and3 passes.126 seed17 settings x3 uncached complete fits,one warmup/two measured. Learned representations rebuilt from scratch including all selected-basis epochs; random includes fit normalizers/initialization. Heads include feature moments/weights/metric/solve/export. FG includes fresh width/centers/readout. One CPU thread; imports/I/O/image processing excluded. Representation parameter-state and explicit head design/metric/system byte counts are phase diagnostics, not peak process RAM; random has no iterative representation state and reports null. Initializer temporaries/data/other arrays/objects remain separate."} | `0b50095c0314b88260fd0335a7b90afc4ead6e4ada538db85ba5d07ab004a65a` |
| [summary.json](../../../../docs/benchmarks/chromaseed_neural_readout_v1/summary.json) | {} | `7a1d8ecdc872acc7378bbd52322371586b729682535729c2fd1619f870c76d77` |
| [verification.json](../../../../docs/benchmarks/chromaseed_neural_readout_v1/verification.json) | {"passed": true} | `3e4079df5d688136ef59be6b599633b0bc7c44d9d21f410eaeefc2aef0699c65` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Role | Input | Loss | Chosen basis | Error ΔE00 ↓ | FG ↓ | Full fit ms | Query μs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | raw36 | norm | adam_e16 | 5.502520 | 5.438652 | 613.374 | 5.50 |
| mixed | raw36 | perceptual | adam_e4 | 5.654729 | 5.438652 | 162.562 | 5.50 |
| mixed | mean3 | norm | tagi_full3_e16 | 6.636132 | 6.547224 | 1009.927 | 6.10 |
| mixed | mean3 | perceptual | tagi_full3_e16 | 6.599125 | 6.547224 | 1013.485 | 6.10 |
| slr_to_ipod | raw36 | norm | adam_e16 | 8.747565 | 8.597000 | 268.037 | 5.60 |
| slr_to_ipod | raw36 | perceptual | adam_e16 | 9.091206 | 8.597000 | 269.495 | 5.60 |
| slr_to_ipod | mean3 | norm | tagi_full3_e4 | 7.122817 | 6.608981 | 114.609 | 6.10 |
| slr_to_ipod | mean3 | perceptual | random | 7.653154 | 6.608981 | 3.345 | 6.10 |
| ipod_to_slr | raw36 | norm | adam_e16 | 10.067767 | 8.705018 | 535.603 | 5.60 |
| ipod_to_slr | raw36 | perceptual | adam_e16 | 10.067797 | 8.705018 | 538.022 | 5.60 |
| ipod_to_slr | mean3 | norm | tagi_full3_e4 | 8.391640 | 8.158681 | 226.811 | 6.20 |
| ipod_to_slr | mean3 | perceptual | tagi_full3_e16 | 8.494089 | 8.158681 | 887.228 | 6.20 |

### Таблица 2

| Role | Input | Loss | Error ΔE00 ↓ | Full fit ms | Numeric B |
| --- | --- | --- | --- | --- | --- |
| mixed | raw36 | norm | 6.048158 | 5.929 | 10564 |
| mixed | raw36 | perceptual | 5.975431 | 9.620 | 10564 |
| mixed | mean3 | norm | 6.788076 | 5.079 | 1855 |
| mixed | mean3 | perceptual | 6.722797 | 8.396 | 1855 |
| slr_to_ipod | raw36 | norm | 9.286951 | 1.647 | 10564 |
| slr_to_ipod | raw36 | perceptual | 9.674299 | 3.425 | 10564 |
| slr_to_ipod | mean3 | norm | 7.604862 | 1.561 | 1855 |
| slr_to_ipod | mean3 | perceptual | 7.653154 | 3.345 | 1855 |
| ipod_to_slr | raw36 | norm | 8.790445 | 4.309 | 10564 |
| ipod_to_slr | raw36 | perceptual | 8.822311 | 7.246 | 10564 |
| ipod_to_slr | mean3 | norm | 8.705993 | 4.231 | 1855 |
| ipod_to_slr | mean3 | perceptual | 8.321937 | 7.171 | 1855 |
