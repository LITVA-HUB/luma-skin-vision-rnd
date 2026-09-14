# chromaseed_gaussian_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_gaussian_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_gaussian_v1/report.md)

SHA-256 отчёта: `c779617edf1f14b1d22904ec78e3c914e571ed0f6ede5426a86813e3638ad9ba`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_gaussian_model_card.md](../../../../docs/architecture/chromaseed_gaussian_model_card.md)
- [chromaseed_gaussian_consumer_erratum.md](../../../../docs/research/chromaseed_gaussian_consumer_erratum.md)
- [chromaseed_gaussian_next_decision.md](../../../../docs/research/chromaseed_gaussian_next_decision.md)
- [chromaseed_gaussian_v1_protocol.md](../../../../docs/research/chromaseed_gaussian_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_gaussian_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_gaussian_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_gaussian.py](../modules/scripts__chromaseed_gaussian.md)
- [scripts/chromaseed_gaussian_audit.py](../modules/scripts__chromaseed_gaussian_audit.md)
- [scripts/chromaseed_gaussian_numpy.py](../modules/scripts__chromaseed_gaussian_numpy.md)
- [scripts/chromaseed_gaussian_reference.py](../modules/scripts__chromaseed_gaussian_reference.md)
- [scripts/chromaseed_gaussian_report.py](../modules/scripts__chromaseed_gaussian_report.md)
- [scripts/chromaseed_gaussian_runtime.py](../modules/scripts__chromaseed_gaussian_runtime.md)
- [scripts/chromaseed_gaussian_train.py](../modules/scripts__chromaseed_gaussian_train.md)
- [scripts/chromaseed_gaussian_verify.py](../modules/scripts__chromaseed_gaussian_verify.md)
- [tests/test_chromaseed_gaussian.py](../tests/tests__test_chromaseed_gaussian.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_gaussian_v1/audit.json) | {"passed": true} | `5b109c88bb0ee96450e0181d2c611b3c1d70134fefe8d09bd8caf73b114a92ff` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_gaussian_v1/runtime.json) | {"scope": "75 actual one-row consumers,20 warmups and3 passes;24 seed17 settings x3 complete fits, one warmup and two measured. One CPU thread. New full fit includes normalizers, balanced weights, initialization, orders, all selected-epoch updates and export; FG includes fresh kernel width, centers and readout. Primary sweep batching, process imports, file I/O and image preprocessing excluded. Training-state bytes count persistent parameter/optimizer arrays only; caller data, transient arrays, RNG and Python/process memory are separate."} | `13e9587931b2c6955a511ba273bb86cc06a27158d34d0f5932861da922dac8a6` |
| [summary.json](../../../../docs/benchmarks/chromaseed_gaussian_v1/summary.json) | {"scope": "Seed-average errors, not prediction ensembles. Reused TRAIN people; roles overlap. Clean and stress output counts overlap at identity."} | `b290ab11d777e582e322487b6551d68fb03bcaaaf30e32c079a03b414c594f56` |
| [verification.json](../../../../docs/benchmarks/chromaseed_gaussian_v1/verification.json) | {"passed": true} | `19a14cd570957c02a1dcf2781169d174b1549efeb1fab915276591fdfde3f098` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Role | Input | Method | Parameter | Epoch | Mean ΔE00 ↓ | p90 ↓ | Bytes | Fit ms | Query μs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | raw36 | adam | 0.0003 | 64 | 5.506610 | 10.7355 | 10,564 | 2387.04 | 5.50 |
| mixed | mean3 | adam | 0.001 | 64 | 6.703580 | 14.6066 | 1,855 | 1994.96 | 6.00 |
| mixed | raw36 | tagi_diag | 1.0 | 64 | 5.548741 | 10.7898 | 10,564 | 3883.12 | 5.50 |
| mixed | mean3 | tagi_diag | 1.0 | 64 | 6.982556 | 14.4399 | 1,855 | 3490.45 | 6.00 |
| mixed | raw36 | tagi_full3 | 1.0 | 64 | 5.553386 | 10.8102 | 10,564 | 4482.17 | 5.40 |
| mixed | mean3 | tagi_full3 | 1.0 | 64 | 6.961798 | 14.5161 | 1,855 | 4024.02 | 6.00 |
| mixed | raw36 | fg_norm_static | 0.1 | None | 5.438652 | 10.3506 | 20,284 | 18.27 | 9.80 |
| mixed | mean3 | fg_norm_static | 0.1 | None | 6.547224 | 13.7263 | 3,127 | 12.16 | 10.00 |
| mixed | constant | constant | None | None | 13.656917 | 24.1055 | 12 | — | 1.20 |
| slr_to_ipod | raw36 | adam | 0.0003 | 64 | 8.605746 | 13.7292 | 10,564 | 1081.11 | 5.40 |
| slr_to_ipod | mean3 | adam | 0.003 | 64 | 7.415713 | 12.6593 | 1,855 | 883.52 | 6.00 |
| slr_to_ipod | raw36 | tagi_diag | 1.0 | 64 | 8.579215 | 13.6802 | 10,564 | 1729.83 | 5.40 |
| slr_to_ipod | mean3 | tagi_diag | 0.3 | 64 | 7.357815 | 12.5741 | 1,855 | 1549.45 | 6.00 |
| slr_to_ipod | raw36 | tagi_full3 | 1.0 | 64 | 8.509649 | 13.6345 | 10,564 | 1959.32 | 5.40 |
| slr_to_ipod | mean3 | tagi_full3 | 0.3 | 64 | 7.419135 | 12.5515 | 1,855 | 1769.18 | 6.00 |
| slr_to_ipod | raw36 | fg_norm_static | 0.1 | None | 8.597000 | 13.6453 | 20,284 | 10.08 | 9.80 |
| slr_to_ipod | mean3 | fg_norm_static | 0.1 | None | 6.608981 | 10.6975 | 3,127 | 5.86 | 10.10 |
| slr_to_ipod | constant | constant | None | None | 12.267125 | 19.4611 | 12 | — | 1.20 |
| ipod_to_slr | raw36 | adam | 0.0003 | 16 | 8.921917 | 16.5726 | 10,564 | 529.40 | 5.50 |
| ipod_to_slr | mean3 | adam | 0.0003 | 64 | 8.867700 | 16.4876 | 1,855 | 1743.11 | 6.10 |
| ipod_to_slr | raw36 | tagi_diag | 0.3 | 64 | 9.709960 | 18.7380 | 10,564 | 3434.65 | 5.50 |
| ipod_to_slr | mean3 | tagi_diag | 1.0 | 64 | 8.747354 | 16.5425 | 1,855 | 3128.55 | 6.00 |
| ipod_to_slr | raw36 | tagi_full3 | 0.3 | 64 | 9.995897 | 18.2803 | 10,564 | 3921.07 | 5.50 |
| ipod_to_slr | mean3 | tagi_full3 | 1.0 | 64 | 8.707169 | 16.4549 | 1,855 | 3519.96 | 6.10 |
| ipod_to_slr | raw36 | fg_norm_static | 0.1 | None | 8.705018 | 15.5016 | 20,284 | 15.98 | 9.80 |
| ipod_to_slr | mean3 | fg_norm_static | 0.1 | None | 8.158681 | 14.8289 | 3,127 | 11.03 | 10.00 |
| ipod_to_slr | constant | constant | None | None | 10.599715 | 19.1456 | 12 | — | 1.20 |
