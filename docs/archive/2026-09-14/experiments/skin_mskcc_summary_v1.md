# skin_mskcc_summary_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Первый MSKCC контроль по трём опубликованным медианным image-Lab признакам, только source validation.

[Полная папка артефактов](../../../../docs/benchmarks/skin_mskcc_summary_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_mskcc_summary_v1/report.md)

SHA-256 отчёта: `853ac085975de60652f369c217c5ce8998ac240cfce0df3969d38a6be560c4b7`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [report.md](../../../../docs/benchmarks/skin_mskcc_summary_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/skin_mskcc_summary_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_mskcc_summary_pilot.py](../modules/scripts__skin_mskcc_summary_pilot.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [independent_audit.json](../../../../docs/benchmarks/skin_mskcc_summary_v1/independent_audit.json) | {"status": "PASS"} | `f3d8b87d5dc869b89d488ff7fac98e7914dc7de8b4dce90dbf399b48b850e852` |
| [independent_refit.json](../../../../docs/benchmarks/skin_mskcc_summary_v1/independent_refit.json) | {} | `eb66be03e61482a6016453dd785e95e62a2cac98c7d86ad2e70b58b37365671d` |
| [results.json](../../../../docs/benchmarks/skin_mskcc_summary_v1/results.json) | {"scope": "SOURCE VALIDATION, never final test; author image summaries, not local pixel pipeline"} | `d0358a8e99a2f1b177db1111aa23725d24bfe43bdba75e42f0a56653fb194a2f` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Locally fit control | Mean DeltaE00 | Median | p95 | Mean at 80% diagnostic coverage |
| --- | --- | --- | --- | --- |
| constant | 9.9879 | 10.4777 | 17.8445 | 9.7425 |
| affine_a0.01 | 5.0504 | 4.6774 | 9.6788 | 5.0318 |
| affine_a1 | 5.0495 | 4.6809 | 9.6734 | 5.0309 |
| affine_a100 | 5.0217 | 4.6710 | 9.0369 | 5.0079 |
| poly2_a0.01 | 4.4994 | 4.0181 | 9.1291 | 4.4328 |
| poly2_a1 | 4.4994 | 4.0279 | 9.1093 | 4.4331 |
| poly2_a100 | 4.5701 | 4.2037 | 8.8142 | 4.5307 |
| mlp64x32_a0.1 | 5.1635 | 4.4246 | 11.8584 | 5.0076 |
| mlp64x32_a1 | 4.3702 | 3.8038 | 9.7150 | 4.3694 |
| knn5 | 4.3924 | 3.5527 | 10.8903 | 4.3935 |
