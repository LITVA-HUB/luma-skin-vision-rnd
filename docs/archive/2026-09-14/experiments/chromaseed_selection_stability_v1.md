# chromaseed_selection_stability_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_selection_stability_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_selection_stability_v1/report.md)

SHA-256 отчёта: `92f0ed283a79053977808b6d662f19cb4bf880e2e64751b30886dc38e0114a84`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_selection_stability_next_decision.md](../../../../docs/research/chromaseed_selection_stability_next_decision.md)
- [chromaseed_selection_stability_v1_protocol.md](../../../../docs/research/chromaseed_selection_stability_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_selection_stability_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_selection_stability_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_selection_stability.py](../modules/scripts__chromaseed_selection_stability.md)
- [scripts/chromaseed_selection_stability_audit.py](../modules/scripts__chromaseed_selection_stability_audit.md)
- [scripts/chromaseed_selection_stability_report.py](../modules/scripts__chromaseed_selection_stability_report.md)
- [scripts/chromaseed_selection_stability_run.py](../modules/scripts__chromaseed_selection_stability_run.md)
- [tests/test_chromaseed_selection_stability.py](../tests/tests__test_chromaseed_selection_stability.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_selection_stability_v1/audit.json) | {"passed": true} | `4da71c0f2d268673deee907bf3be88314926778ff2459141de9bad41cbc87432` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_selection_stability_v1/source_lock.json) | {} | `88189161a7cee0c87d8ed5da2695083374564984c01ad77e3711f5fd49b3efea` |
| [summary.json](../../../../docs/benchmarks/chromaseed_selection_stability_v1/summary.json) | {} | `4b61aa8d4eca5bbc1ab6047b2ea613332ac7eb458bc6e14612e9e95b929e17eb` |
| [verification.json](../../../../docs/benchmarks/chromaseed_selection_stability_v1/verification.json) | {"passed": true} | `591fd57564487c023b67fa7f8e1b568261228f51d50fde3eb23ab850c192b89e` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Fit-side role | Family | Single-person deletions changing choice | Original choice retained in bootstrap | Weak alpha (<0.1) selected | Original rank, 95th percentile | Full-OOF regret, 95th percentile |
| --- | --- | --- | --- | --- | --- | --- |
| Mixed cameras | Normalized ridge | 0/18 | 46.48% | 97.56% | 4 | 0.1686 |
| Mixed cameras | Shared perceptual | 0/18 | 72.08% | 98.23% | 3 | 0.1680 |
| Mixed cameras | Local perceptual | 0/18 | 57.28% | 98.83% | 3 | 0.1523 |
| Mixed cameras | Fixed-metric correction | 7/18 | 25.17% | 99.19% | 11 | 0.1704 |
| Mixed cameras | Midpoint correction | 5/18 | 23.29% | 99.54% | 11 | 0.1670 |
| SLR fit side | Normalized ridge | 1/8 | 61.72% | 99.97% | 5 | 0.1247 |
| SLR fit side | Shared perceptual | 0/8 | 71.74% | 99.99% | 4 | 0.1701 |
| SLR fit side | Local perceptual | 0/8 | 69.54% | 100.00% | 5 | 0.1670 |
| SLR fit side | Fixed-metric correction | 3/8 | 38.88% | 99.97% | 11 | 0.2785 |
| SLR fit side | Midpoint correction | 2/8 | 38.05% | 99.97% | 11 | 0.2690 |
| iPod fit side | Normalized ridge | 2/16 | 56.02% | 61.81% | 4 | 0.1865 |
| iPod fit side | Shared perceptual | 1/16 | 68.43% | 74.41% | 2 | 0.0563 |
| iPod fit side | Local perceptual | 0/16 | 73.02% | 81.22% | 2 | 0.0465 |
| iPod fit side | Fixed-metric correction | 8/16 | 27.28% | 89.42% | 9 | 0.2337 |
| iPod fit side | Midpoint correction | 9/16 | 17.09% | 91.86% | 9 | 0.2423 |

### Таблица 2

| Fit-side role | Family | W − P inner person mean | Conditional 2.5–97.5% range | Fraction below zero |
| --- | --- | --- | --- | --- |
| Mixed cameras | Normalized ridge | -0.1640 | [-0.4541, 0.0273] | 92.45% |
| Mixed cameras | Shared perceptual | -0.1576 | [-0.4129, 0.0089] | 96.19% |
| Mixed cameras | Local perceptual | -0.1618 | [-0.4045, 0.0008] | 97.39% |
| Mixed cameras | Fixed-metric correction | -0.1674 | [-0.4455, 0.0413] | 92.77% |
| Mixed cameras | Midpoint correction | -0.1739 | [-0.4414, 0.0242] | 94.89% |
| SLR fit side | Normalized ridge | -0.3021 | [-0.6824, -0.0114] | 97.95% |
| SLR fit side | Shared perceptual | -0.3072 | [-0.7053, -0.0133] | 98.14% |
| SLR fit side | Local perceptual | -0.3043 | [-0.6992, -0.0290] | 98.53% |
| SLR fit side | Fixed-metric correction | -0.4559 | [-1.0586, 0.0072] | 97.29% |
| SLR fit side | Midpoint correction | -0.4464 | [-1.0157, -0.0217] | 98.06% |
| iPod fit side | Normalized ridge | -0.0159 | [-0.1101, 0.0710] | 62.31% |
| iPod fit side | Shared perceptual | -0.0327 | [-0.1315, 0.0577] | 74.39% |
| iPod fit side | Local perceptual | -0.0402 | [-0.1328, 0.0449] | 81.31% |
| iPod fit side | Fixed-metric correction | -0.0632 | [-0.1549, 0.0220] | 91.98% |
| iPod fit side | Midpoint correction | -0.0717 | [-0.1597, 0.0081] | 95.96% |
