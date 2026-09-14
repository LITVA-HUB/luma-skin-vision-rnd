# chromaseed_gated_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_gated_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_gated_v1/report.md)

SHA-256 отчёта: `b9521d73e82d018b49da1f716d2f34f5cd591c2d3531d4950180e2688a8a4aaf`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_gated_model_card.md](../../../../docs/architecture/chromaseed_gated_model_card.md)
- [chromaseed_gated_next_decision.md](../../../../docs/research/chromaseed_gated_next_decision.md)
- [chromaseed_gated_v1_protocol.md](../../../../docs/research/chromaseed_gated_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_gated_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_gated_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_gated.py](../modules/scripts__chromaseed_gated.md)
- [scripts/chromaseed_gated_audit.py](../modules/scripts__chromaseed_gated_audit.md)
- [scripts/chromaseed_gated_numpy.py](../modules/scripts__chromaseed_gated_numpy.md)
- [scripts/chromaseed_gated_report.py](../modules/scripts__chromaseed_gated_report.md)
- [scripts/chromaseed_gated_runtime.py](../modules/scripts__chromaseed_gated_runtime.md)
- [scripts/chromaseed_gated_train.py](../modules/scripts__chromaseed_gated_train.md)
- [scripts/chromaseed_gated_verify.py](../modules/scripts__chromaseed_gated_verify.md)
- [tests/test_chromaseed_gated.py](../tests/tests__test_chromaseed_gated.md)
- [tests/test_chromaseed_gated_numpy.py](../tests/tests__test_chromaseed_gated_numpy.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_gated_v1/audit.json) | {"passed": true} | `1b5aecd78d3fbea4e44d74aa9e0267546a37d13dc540bbfa4a99274dba1046f4` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_gated_v1/runtime.json) | {"scope": "Actual batch-one canonical and NumPy-only predictors;20 warmups/3 query passes. All72 outputs verified. Complete fits include weights/preparation/gate/basis/readout; no I/O/import/search. Numeric payload excludes runtime cache/library. Four active cost probes do not generate new quality scores."} | `259a8d021ddfe0d2a655de3acc8d4e32bff5eef789a0808e11533cf3f63c5105` |
| [selections.json](../../../../docs/benchmarks/chromaseed_gated_v1/selections.json) | {} | `da046187e9fa2da00c9605d1e709b28452b0cb0128349ae92b35c83cfbb44418` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_gated_v1/source_lock.json) | {} | `7af66943a24be2f3d41c0f8dd5ad1261ad0126471ea0c9f97d48e9069608dd2f` |
| [summary.json](../../../../docs/benchmarks/chromaseed_gated_v1/summary.json) | {} | `50948c386b6be337dcbb13c606989f88aee572057d527bc1a63114ca8fe9d838` |
| [verification.json](../../../../docs/benchmarks/chromaseed_gated_v1/verification.json) | {"passed": true} | `9b73c11efed3d4fae6e6e41f4987377a68fc841af37a88984194f879a1d00065` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Family | Mixed ΔE00 ↓ | SLR → iPod ↓ | iPod → SLR ↓ | Mixed numeric bytes |
| --- | --- | --- | --- | --- |
| MSE base | 5.438652 | 8.597000 | 8.705018 | 20,284 |
| MSE uniform | 5.384387 | 8.818465 | 8.742294 | 20,284 |
| MSE soft gate | 5.349017 | 8.597000 | 8.705018 | 21,973 |
| MSE hard gate | 5.322671 | 8.597000 | 8.705018 | 21,973 |
| Perceptual base | 5.390066 | 8.654805 | 8.386889 | 20,284 |
| Perceptual uniform | 5.361655 | 8.684220 | 8.585753 | 20,284 |
| Perceptual soft gate | 5.295118 | 8.654805 | 8.386889 | 21,973 |
| Perceptual hard gate | 5.285104 | 8.654805 | 8.386889 | 21,973 |

### Таблица 2

| Role | Family | λ / ρ | Inner person ΔE00 | Held image mean | Held site/person mean | Held image p90 |
| --- | --- | --- | --- | --- | --- | --- |
| Mixed | MSE base | 10 / 0 | 4.455353 | 5.6221 | 5.4572 | 10.3506 |
| Mixed | MSE uniform | 0.1 / 0.5 | 4.442135 | 5.5558 | 5.4003 | 10.2425 |
| Mixed | MSE soft gate | 0.1 / 0.5 | 4.429678 | 5.5213 | 5.3670 | 10.0958 |
| Mixed | MSE hard gate | 0.1 / 0.5 | 4.413827 | 5.4948 | 5.3390 | 9.9906 |
| Mixed | Perceptual base | 10 / 0 | 4.448451 | 5.5736 | 5.4102 | 10.1994 |
| Mixed | Perceptual uniform | 1 / 1 | 4.438933 | 5.5379 | 5.3808 | 10.1299 |
| Mixed | Perceptual soft gate | 0.1 / 0.5 | 4.431187 | 5.4690 | 5.3139 | 9.9522 |
| Mixed | Perceptual hard gate | 0.1 / 0.5 | 4.420066 | 5.4567 | 5.3023 | 10.0012 |
| SLR → iPod | MSE base | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | MSE uniform | 0.1 / 1 | 4.878367 | 9.0389 | 8.8410 | 13.8970 |
| SLR → iPod | MSE soft gate | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | MSE hard gate | 10 / 0 | 5.058676 | 8.8450 | 8.6149 | 13.6453 |
| SLR → iPod | Perceptual base | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| SLR → iPod | Perceptual uniform | 0.1 / 1 | 5.067245 | 8.8698 | 8.7072 | 13.6099 |
| SLR → iPod | Perceptual soft gate | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| SLR → iPod | Perceptual hard gate | 10 / 0 | 5.177163 | 8.9011 | 8.6748 | 13.6978 |
| iPod → SLR | MSE base | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | MSE uniform | 0.1 / 0.5 | 4.364153 | 8.7970 | 8.7647 | 15.2637 |
| iPod → SLR | MSE soft gate | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | MSE hard gate | 10 / 0 | 4.371878 | 8.7759 | 8.7316 | 15.5016 |
| iPod → SLR | Perceptual base | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |
| iPod → SLR | Perceptual uniform | 0.1 / 1 | 4.308546 | 8.6224 | 8.6032 | 15.2084 |
| iPod → SLR | Perceptual soft gate | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |
| iPod → SLR | Perceptual hard gate | 10 / 0 | 4.332083 | 8.4466 | 8.4143 | 15.0304 |

### Таблица 3

| Candidate | Matched reference | Mean difference ↓ | People improved | Descriptive range |
| --- | --- | --- | --- | --- |
| MSE uniform | MSE base | -0.054265 | 4/6 | [-0.134233, +0.020676] |
| MSE soft gate | MSE base | -0.089635 | 4/6 | [-0.239117, +0.016089] |
| MSE soft gate | MSE uniform | -0.035371 | 3/6 | [-0.124019, +0.031529] |
| MSE hard gate | MSE base | -0.115981 | 4/6 | [-0.256804, -0.000754] |
| MSE hard gate | MSE uniform | -0.061716 | 5/6 | [-0.132460, -0.005364] |
| Perceptual uniform | Perceptual base | -0.028411 | 3/6 | [-0.077192, +0.018145] |
| Perceptual soft gate | Perceptual base | -0.094948 | 5/6 | [-0.235523, +0.005784] |
| Perceptual soft gate | Perceptual uniform | -0.066537 | 4/6 | [-0.181396, +0.012361] |
| Perceptual hard gate | Perceptual base | -0.104962 | 4/6 | [-0.229843, +0.002567] |
| Perceptual hard gate | Perceptual uniform | -0.076551 | 5/6 | [-0.164658, -0.006365] |

### Таблица 4

| Role | Family | Numeric B | Canonical µs | NumPy µs / p95 | Cached array B | Full fit ms |
| --- | --- | --- | --- | --- | --- | --- |
| Mixed | MSE base | 20,284 | 16.50 | 9.30 / 9.50 | 41,272 | 16.65 |
| Mixed | MSE uniform | 20,284 | 16.50 | 9.30 / 9.50 | 41,272 | 18.43 |
| Mixed | MSE soft gate | 21,973 | 22.60 | 11.80 / 12.00 | 44,640 | 28.05 |
| Mixed | MSE hard gate | 21,973 | 22.40 | 11.80 / 12.00 | 44,640 | 28.15 |
| Mixed | Perceptual base | 20,284 | 16.60 | 9.40 / 9.50 | 41,272 | 22.98 |
| Mixed | Perceptual uniform | 20,284 | 16.50 | 9.40 / 9.60 | 41,272 | 24.79 |
| Mixed | Perceptual soft gate | 21,973 | 22.80 | 11.80 / 12.00 | 44,640 | 34.51 |
| Mixed | Perceptual hard gate | 21,973 | 22.40 | 11.70 / 11.90 | 44,640 | 35.53 |
| SLR → iPod | MSE base | 20,284 | 16.40 | 9.30 / 9.50 | 41,272 | 8.48 |
| SLR → iPod | MSE uniform | 20,284 | 16.60 | 9.40 / 9.60 | 41,272 | 10.13 |
| SLR → iPod | MSE soft gate | 20,284 | 16.50 | 9.40 / 9.50 | 41,272 | 8.52 |
| SLR → iPod | MSE hard gate | 20,284 | 16.50 | 9.50 / 9.70 | 41,272 | 8.52 |
| SLR → iPod | Perceptual base | 20,284 | 16.80 | 9.60 / 9.70 | 41,272 | 12.31 |
| SLR → iPod | Perceptual uniform | 20,284 | 16.80 | 9.60 / 9.90 | 41,272 | 13.48 |
| SLR → iPod | Perceptual soft gate | 20,284 | 16.90 | 9.60 / 9.90 | 41,272 | 12.19 |
| SLR → iPod | Perceptual hard gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 12.20 |
| iPod → SLR | MSE base | 20,284 | 16.80 | 9.60 / 9.80 | 41,272 | 14.61 |
| iPod → SLR | MSE uniform | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 16.39 |
| iPod → SLR | MSE soft gate | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 14.60 |
| iPod → SLR | MSE hard gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 14.53 |
| iPod → SLR | Perceptual base | 20,284 | 16.90 | 9.60 / 9.70 | 41,272 | 20.27 |
| iPod → SLR | Perceptual uniform | 20,284 | 16.90 | 9.50 / 9.70 | 41,272 | 21.87 |
| iPod → SLR | Perceptual soft gate | 20,284 | 16.90 | 9.60 / 9.80 | 41,272 | 20.35 |
| iPod → SLR | Perceptual hard gate | 20,284 | 16.90 | 9.50 / 9.90 | 41,272 | 20.30 |

### Таблица 5

| Fixed active mixed probe | NumPy µs | Canonical µs | Full fit ms |
| --- | --- | --- | --- |
| MSE soft gate | 12.20 | 23.50 | 28.55 |
| MSE hard gate | 12.00 | 23.20 | 28.06 |
| Perceptual soft gate | 12.10 | 23.40 | 34.39 |
| Perceptual hard gate | 12.10 | 23.10 | 34.27 |
