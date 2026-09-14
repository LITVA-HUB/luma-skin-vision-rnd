# chromaseed_gate_stability_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_gate_stability_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_gate_stability_v1/report.md)

SHA-256 отчёта: `a2173a5c69ce5adeb103c823af7131c8a615d3caf9dfbd3878cc07397dfec856`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_gate_stability_addendum.md](../../../../docs/architecture/chromaseed_gate_stability_addendum.md)
- [chromaseed_gate_stability_next_decision.md](../../../../docs/research/chromaseed_gate_stability_next_decision.md)
- [chromaseed_gate_stability_v1_protocol.md](../../../../docs/research/chromaseed_gate_stability_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_gate_stability_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_gate_stability_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_gate_stability.py](../modules/scripts__chromaseed_gate_stability.md)
- [scripts/chromaseed_gate_stability_audit.py](../modules/scripts__chromaseed_gate_stability_audit.md)
- [scripts/chromaseed_gate_stability_report.py](../modules/scripts__chromaseed_gate_stability_report.md)
- [scripts/chromaseed_gate_stability_run.py](../modules/scripts__chromaseed_gate_stability_run.md)
- [scripts/chromaseed_gate_stability_verify.py](../modules/scripts__chromaseed_gate_stability_verify.md)
- [tests/test_chromaseed_gate_stability.py](../tests/tests__test_chromaseed_gate_stability.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_gate_stability_v1/audit.json) | {"passed": true, "scope": "Independent feature algebra, actual NumPy batch-one consumer, all person/camera metrics and boundary constructions. Shared verified CIEDE2000. Synthetic sensitivity only; no new fits or target-image measurements."} | `1ac12a680ffc0a2232ed97ff6ce25b76d5180bdcdcdd56e2c16c56d20f1af501` |
| [source_lock.json](../../../../docs/benchmarks/chromaseed_gate_stability_v1/source_lock.json) | {} | `5102e3ba875d0a21f71d13dd538edf55b04b1efc0deb4110d19d66522c935b26` |
| [summary.json](../../../../docs/benchmarks/chromaseed_gate_stability_v1/summary.json) | {} | `9ac7d0095a34b9cdd98f5f45e8e3744bbab707dc951bf8dcb4fbdd0f76a7b3fd` |
| [verification.json](../../../../docs/benchmarks/chromaseed_gate_stability_v1/verification.json) | {"passed": true} | `2ad0c77feabbe7d63fe5f6ab4ecf905a4705a95014e4cb4dc7c8b703a8c4c738` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Role | Family | Original | 1/255 | 4/255 | 16/255 | 64/255 |
| --- | --- | --- | --- | --- | --- | --- |
| Mixed | norm_base | 5.43865 | 5.66528 | 6.39770 | 9.75930 | 24.40036 |
| Mixed | norm_uniform | 5.38439 | 5.62068 | 6.38762 | 9.92017 | 25.28959 |
| Mixed | norm_soft | 5.34902 | 5.57924 | 6.32850 | 9.78452 | 24.88204 |
| Mixed | norm_hard | 5.32267 | 5.55550 | 6.30010 | 9.76665 | 25.19345 |
| Mixed | perceptual_base | 5.39007 | 5.62063 | 6.36767 | 9.80445 | 24.65591 |
| Mixed | perceptual_uniform | 5.36166 | 5.60088 | 6.37782 | 9.94506 | 25.26538 |
| Mixed | perceptual_soft | 5.29512 | 5.52850 | 6.29059 | 9.80635 | 25.10125 |
| Mixed | perceptual_hard | 5.28510 | 5.52014 | 6.27800 | 9.80814 | 25.40082 |
| SLR → iPod | norm_base | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | norm_uniform | 8.81846 | 8.97763 | 9.46406 | 11.48746 | 20.06838 |
| SLR → iPod | norm_soft | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | norm_hard | 8.59700 | 8.73759 | 9.16801 | 10.98200 | 18.91509 |
| SLR → iPod | perceptual_base | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| SLR → iPod | perceptual_uniform | 8.68422 | 8.84375 | 9.32990 | 11.33489 | 19.57247 |
| SLR → iPod | perceptual_soft | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| SLR → iPod | perceptual_hard | 8.65480 | 8.79748 | 9.23448 | 11.07693 | 19.04825 |
| iPod → SLR | norm_base | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | norm_uniform | 8.74229 | 8.95003 | 9.61525 | 12.72508 | 22.92760 |
| iPod → SLR | norm_soft | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | norm_hard | 8.70502 | 8.89857 | 9.51912 | 12.42321 | 21.96572 |
| iPod → SLR | perceptual_base | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |
| iPod → SLR | perceptual_uniform | 8.58575 | 8.81311 | 9.53732 | 12.87690 | 23.88457 |
| iPod → SLR | perceptual_soft | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |
| iPod → SLR | perceptual_hard | 8.38689 | 8.58254 | 9.20950 | 12.14965 | 22.07537 |

### Таблица 2

| Role | Family | 1/255 | 4/255 | 16/255 | 64/255 |
| --- | --- | --- | --- | --- | --- |
| Mixed | norm_base | 0.33780 | 1.36375 | 5.60423 | 21.66107 |
| Mixed | norm_uniform | 0.35323 | 1.42568 | 5.85892 | 22.62223 |
| Mixed | norm_soft | 0.34807 | 1.40519 | 5.77925 | 22.40378 |
| Mixed | norm_hard | 0.34985 | 1.40453 | 5.77496 | 22.74650 |
| Mixed | perceptual_base | 0.34267 | 1.38297 | 5.67998 | 21.83040 |
| Mixed | perceptual_uniform | 0.35528 | 1.43357 | 5.88230 | 22.52323 |
| Mixed | perceptual_soft | 0.35091 | 1.41728 | 5.83504 | 22.53903 |
| Mixed | perceptual_hard | 0.35218 | 1.41772 | 5.83676 | 22.88771 |
| SLR → iPod | norm_base | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | norm_uniform | 0.27562 | 1.11069 | 4.55146 | 18.20195 |
| SLR → iPod | norm_soft | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | norm_hard | 0.23768 | 0.95697 | 3.90437 | 15.30820 |
| SLR → iPod | perceptual_base | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| SLR → iPod | perceptual_uniform | 0.28036 | 1.12885 | 4.60698 | 18.01882 |
| SLR → iPod | perceptual_soft | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| SLR → iPod | perceptual_hard | 0.24508 | 0.98568 | 4.00503 | 15.33701 |
| iPod → SLR | norm_base | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | norm_uniform | 0.34119 | 1.38711 | 5.79884 | 18.37424 |
| iPod → SLR | norm_soft | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | norm_hard | 0.31533 | 1.28275 | 5.35820 | 16.97143 |
| iPod → SLR | perceptual_base | 0.32211 | 1.30884 | 5.45167 | 17.13445 |
| iPod → SLR | perceptual_uniform | 0.36875 | 1.49766 | 6.23259 | 19.67748 |
| iPod → SLR | perceptual_soft | 0.32211 | 1.30884 | 5.45167 | 17.13445 |
| iPod → SLR | perceptual_hard | 0.32211 | 1.30884 | 5.45167 | 17.13445 |

### Таблица 3

| Strength | Query rows flipping under any anchor | Equal-person mean | SLR / iPod person means | Boundary roots below strength |
| --- | --- | --- | --- | --- |
| 1/255 | 1/232 | 0.379% | 1.136% / 0.000% | 2 |
| 4/255 | 2/232 | 0.758% | 2.273% / 0.000% | 5 |
| 16/255 | 24/232 | 9.414% | 16.667% / 5.788% | 65 |
| 64/255 | 185/232 | 79.327% | 67.929% / 85.025% | 534 |

### Таблица 4

| Base loss | Output control | Person-mean jump | Image-mean jump | Mean seed p90 | Maximum across seeds |
| --- | --- | --- | --- | --- | --- |
| norm | base | 0.010794 | 0.010913 | 0.018766 | 0.025667 |
| norm | uniform | 0.011311 | 0.011389 | 0.019808 | 0.026554 |
| norm | soft | 0.010841 | 0.010966 | 0.019027 | 0.025754 |
| norm | hard | 2.026345 | 1.712969 | 3.359925 | 8.818099 |
| perceptual | base | 0.010894 | 0.010916 | 0.018967 | 0.025599 |
| perceptual | uniform | 0.011256 | 0.011269 | 0.019643 | 0.026368 |
| perceptual | soft | 0.010986 | 0.010996 | 0.019098 | 0.025728 |
| perceptual | hard | 1.927893 | 1.646498 | 3.230529 | 7.817570 |

### Таблица 5

| Loss | Seed | Mean person RMS distance | Rows passing basic checks | Mean person limit jump | Max limit jump |
| --- | --- | --- | --- | --- | --- |
| norm | 17 | 0.134749 | 185/232 | 1.038901 | 3.103398 |
| norm | 29 | 0.134749 | 185/232 | 1.036260 | 2.996774 |
| norm | 43 | 0.134749 | 185/232 | 1.036175 | 3.079604 |
| perceptual | 17 | 0.134749 | 185/232 | 1.080845 | 3.199779 |
| perceptual | 29 | 0.134749 | 185/232 | 1.099353 | 3.022444 |
| perceptual | 43 | 0.134749 | 185/232 | 1.085113 | 2.993824 |
