# chromaseed_neural_shrinkage_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/report.md)

SHA-256 отчёта: `696dbbb7b0583daef39995be2c731f65563f53a9c887e333c0844d2238a86192`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_neural_shrinkage_model_card.md](../../../../docs/architecture/chromaseed_neural_shrinkage_model_card.md)
- [chromaseed_neural_shrinkage_next_decision.md](../../../../docs/research/chromaseed_neural_shrinkage_next_decision.md)
- [chromaseed_neural_shrinkage_v1_protocol.md](../../../../docs/research/chromaseed_neural_shrinkage_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_neural_shrinkage_audit.py](../modules/scripts__chromaseed_neural_shrinkage_audit.md)
- [scripts/chromaseed_neural_shrinkage_report.py](../modules/scripts__chromaseed_neural_shrinkage_report.md)
- [scripts/chromaseed_neural_shrinkage_runtime.py](../modules/scripts__chromaseed_neural_shrinkage_runtime.md)
- [scripts/chromaseed_neural_shrinkage_train.py](../modules/scripts__chromaseed_neural_shrinkage_train.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/audit.json) | {"passed": true} | `1f558b60645112b1e5709461a789e21dd2858e04314a1424da657c8b01e7d015` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/runtime.json) | {"scope": "CPU1thread.381 actual consumers;20 warmups/3 query passes.84 seed17 heads and6 FG settings x3 complete fits, first warmup discarded. Every hidden representation reconstructed from scratch. Fit includes all representation epochs, feature moments, weighted metric/readout and export. Prepared color features only; no image/I/O/import/phone/GPU inference claim. Explicit array/state bytes are not process peak RAM."} | `cf2c5fe8cc7bb7359207b9f8a07b228f2c41c68850fd50fc4712fbf816312cec` |
| [summary.json](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/summary.json) | {} | `af73391f231779439a3e13ee99a317deb474decb157b7b0c14f4b5e6b98ef6d1` |
| [verification.json](../../../../docs/benchmarks/chromaseed_neural_shrinkage_v1/verification.json) | {"passed": true} | `e725b4a6e86df7a51dd9825089ca39c218100bc3e31f765a585946353fb9cd6f` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Роль | Вход | Цель | Основа | alpha | NS ΔE00 | NR правило | FG | Полное обучение, мс | Ответ, мкс |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ipod_to_slr | mean3 | norm | tagi_full3_e4 | 10 | 8.391640 | 8.391640 | 8.158681 | 225.512 | 6.20 |
| ipod_to_slr | mean3 | perceptual | tagi_full3_e16 | 10 | 8.494089 | 8.494089 | 8.158681 | 884.774 | 6.10 |
| ipod_to_slr | raw36 | norm | adam_e16 | 100 | 9.539056 | 10.067767 | 8.705018 | 534.851 | 5.60 |
| ipod_to_slr | raw36 | perceptual | adam_e16 | 10 | 10.067797 | 10.067797 | 8.705018 | 550.174 | 5.60 |
| mixed | mean3 | norm | tagi_full3_e16 | 10 | 6.636132 | 6.636132 | 6.547224 | 1011.929 | 6.00 |
| mixed | mean3 | perceptual | tagi_full3_e16 | 10 | 6.599125 | 6.599125 | 6.547224 | 1019.731 | 6.00 |
| mixed | raw36 | norm | adam_e16 | 100 | 5.678073 | 5.502520 | 5.438652 | 625.070 | 5.50 |
| mixed | raw36 | perceptual | adam_e16 | 100 | 5.607624 | 5.654729 | 5.438652 | 618.775 | 5.40 |
| slr_to_ipod | mean3 | norm | tagi_full3_e4 | 10 | 7.122817 | 7.122817 | 6.608981 | 112.619 | 6.10 |
| slr_to_ipod | mean3 | perceptual | random | 10 | 7.653154 | 7.653154 | 6.608981 | 3.335 | 6.10 |
| slr_to_ipod | raw36 | norm | adam_e16 | 10 | 8.747565 | 8.747565 | 8.597000 | 265.995 | 5.60 |
| slr_to_ipod | raw36 | perceptual | adam_e16 | 10 | 9.091206 | 9.091206 | 8.597000 | 270.300 | 5.60 |

### Таблица 2

| Mixed | Ошибка ΔE00 | Числовые байты | Полное обучение, мс | Ответ, мкс |
| --- | --- | --- | --- | --- |
| norm/random/raw36 | 6.048158 | 10564 | 5.266 | 5.60 |
| norm/random/mean3 | 6.788076 | 1855 | 4.965 | 6.00 |
| fg_norm_static/reference/raw36 | 5.438652 | 20284 | 18.124 | 10.00 |
| fg_norm_static/reference/mean3 | 6.547224 | 3127 | 12.139 | 10.30 |
