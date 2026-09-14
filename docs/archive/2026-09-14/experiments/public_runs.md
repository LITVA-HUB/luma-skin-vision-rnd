# public_runs

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Первый реальный SimpleCube++ benchmark: классические и компактные нейросетевые контроли.

[Полная папка артефактов](../../../../docs/benchmarks/public_runs)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/public_benchmark_report.md)

SHA-256 отчёта: `3da9411842b747cb52694577d3f1ec7819e35e2e5f61a1c80afbfc87f8e0c3f2`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.


## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [src/luma_skin_vision/cc/benchmark.py](../modules/src__luma_skin_vision__cc__benchmark.md)
- [src/luma_skin_vision/cc/model.py](../modules/src__luma_skin_vision__cc__model.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Method | Recovery mean | Reproduction mean | Reproduction median | Risk at 80% | AURC |
| --- | --- | --- | --- | --- | --- |
| Gray World + learned selector | 3.476 ± 0.000 | 4.592 ± 0.000 | 2.971 ± 0.000 | 3.368 ± 0.043 | 2.451 ± 0.074 |
| Max RGB + learned selector | 4.895 ± 0.000 | 5.657 ± 0.000 | 3.038 ± 0.000 | 3.352 ± 0.009 | 2.429 ± 0.035 |
| Shades of Gray + learned selector | 2.758 ± 0.000 | 3.573 ± 0.000 | 1.778 ± 0.000 | 2.316 ± 0.021 | 1.900 ± 0.049 |
| Gray Edge + learned selector | 2.967 ± 0.000 | 3.887 ± 0.000 | 2.124 ± 0.000 | 2.638 ± 0.086 | 2.151 ± 0.126 |
| C+ standard context error head | 1.655 ± 0.035 | 2.166 ± 0.061 | 1.065 ± 0.107 | 1.678 ± 0.085 | 1.569 ± 0.178 |
| C + disagreement-only head | 1.655 ± 0.035 | 2.166 ± 0.061 | 1.065 ± 0.107 | 1.657 ± 0.090 | 1.241 ± 0.031 |
| C + combined error head (strong control) | 1.655 ± 0.035 | 2.166 ± 0.061 | 1.065 ± 0.107 | 1.556 ± 0.072 | 1.260 ± 0.119 |
| Mixture + context-only head | 1.573 ± 0.061 | 2.085 ± 0.071 | 0.979 ± 0.077 | 1.809 ± 0.137 | 1.915 ± 0.575 |
| Mixture + disagreement-only head | 1.573 ± 0.061 | 2.085 ± 0.071 | 0.979 ± 0.077 | 1.676 ± 0.068 | 1.435 ± 0.159 |
| Proposed mixture + combined head | 1.573 ± 0.061 | 2.085 ± 0.071 | 0.979 ± 0.077 | 1.586 ± 0.048 | 1.333 ± 0.145 |

### Таблица 2

| Accepted target | C+ | Strong combined control | Proposed |
| --- | --- | --- | --- |
| 100% | 2.166 ± 0.061 | 2.166 ± 0.061 | 2.085 ± 0.071 |
| 95% | 1.970 ± 0.130 | 2.016 ± 0.028 | 1.912 ± 0.024 |
| 90% | 1.867 ± 0.150 | 1.855 ± 0.049 | 1.790 ± 0.065 |
| 80% | 1.678 ± 0.085 | 1.556 ± 0.072 | 1.586 ± 0.048 |
| 70% | 1.535 ± 0.021 | 1.398 ± 0.067 | 1.424 ± 0.088 |
| 60% | 1.429 ± 0.127 | 1.266 ± 0.067 | 1.319 ± 0.154 |

### Таблица 3

| Protocol / method | Reproduction mean | Risk 80 | Frozen source 80 threshold: coverage / risk |
| --- | --- | --- | --- |
| Sony 30 / Gray World + learned selector | 4.073 ± 0.000 | 2.987 ± 0.248 | 54.4% / 2.556 ± 0.265 |
| Sony 30 / Shades of Gray + learned selector | 4.371 ± 0.000 | 3.389 ± 0.285 | 20.0% / 0.746 ± 0.495 |
| Sony 30 / C+ standard context error head | 8.859 ± 0.785 | 8.402 ± 0.336 | 51.1% / 8.455 ± 0.930 |
| Sony 30 / C + combined error head (strong control) | 8.859 ± 0.785 | 6.612 ± 0.566 | 22.2% / 5.151 ± 0.796 |
| Sony 30 / Proposed mixture + combined head | 5.520 ± 0.339 | 4.430 ± 0.320 | 48.9% / 3.181 ± 0.701 |
| 550D→600D / Gray World + learned selector | 2.991 | 2.193 | 79.1% / 2.167 |
| 550D→600D / Shades of Gray + learned selector | 2.869 | 1.966 | 79.9% / 1.966 |
| 550D→600D / C+ standard context error head | 2.983 | 2.637 | 96.2% / 2.884 |
| 550D→600D / C + combined error head (strong control) | 2.983 | 2.080 | 91.7% / 2.470 |
| 550D→600D / Proposed mixture + combined head | 2.848 | 1.891 | 82.2% / 1.896 |

### Таблица 4

| Method (seed 17) | Reproduction p 95 | Worst 25 mean | >10° fraction | >20° fraction |
| --- | --- | --- | --- | --- |
| Shades of Gray + learned selector | 12.512 | 9.586 | 9.5% | 1.3% |
| C+ standard context error head | 8.250 | 6.058 | 3.0% | 0.0% |
| C + combined error head (strong control) | 8.250 | 6.058 | 3.0% | 0.0% |
| Proposed mixture + combined head | 9.614 | 6.211 | 4.5% | 0.0% |

### Таблица 5

| Model seed 17 | Active / stored parameters | Checkpoint MB | Training peak allocated MB | Model-only batch 1 median ms | Inference peak allocated MB |
| --- | --- | --- | --- | --- | --- |
| baseline | 964131 / 966760 | 3.991 | 618.9 | 3.181 | 16.9 |
| proposed | 966760 / 966760 | 3.991 | 619.0 | 3.300 | 16.9 |

### Таблица 6

| Method | Median ms | p 95 ms | CPU expert bank median ms |
| --- | --- | --- | --- |
| C+ standard context error head | 23.21 | 24.23 | 0.01 |
| C + combined error head (strong control) | 65.14 | 66.09 | 41.87 |
| Proposed mixture + combined head | 69.00 | 72.90 | 43.12 |
