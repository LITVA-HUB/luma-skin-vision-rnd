# skin_spectral_probe_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

22 500 спектров из трёх TRAIN лиц. Представимость материала проверена отдельно от восстановления по RGB.

[Полная папка артефактов](../../../../docs/benchmarks/skin_spectral_probe_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_spectral_probe_v1/report.md)

SHA-256 отчёта: `ad02ba9ee159f3579a3976b3eab4433a56f3f344758415ab3abe28605561d110`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_spectral_probe_protocol_v1.md](../../../../docs/research/skin_spectral_probe_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_spectral_probe_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_spectral_probe_audit_v1.py](../modules/scripts__skin_spectral_probe_audit_v1.md)
- [scripts/skin_spectral_probe_v1.py](../modules/scripts__skin_spectral_probe_v1.md)
- [tests/test_skin_spectral_probe.py](../tests/tests__test_skin_spectral_probe.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_spectral_probe_v1/audit.json) | {"status": "PASS"} | `26ac36964336428a87440df855b300fa94888b0e5aea325f2f93ae96a23cd5e4` |
| [source_lock.json](../../../../docs/benchmarks/skin_spectral_probe_v1/source_lock.json) | {} | `f4cf179a32c2dd0ab14607a027f63daa55d12b96fc774e664e076c18feacd409` |
| [spatial_followup.json](../../../../docs/benchmarks/skin_spectral_probe_v1/spatial_followup.json) | {"scope": "POST-SCREEN TRAIN follow-up; same measured regional means; no new labels or pixels"} | `0375c12cefd63af60203fd6b9df8e1067263df9a4376a1388521ec044ca9bb34` |
| [summary.json](../../../../docs/benchmarks/skin_spectral_probe_v1/summary.json) | {"scope": "MEASURED TRAIN SPECTRA; oracle compression and DERIVED RADIANCE stress, not RGB accuracy"} | `9f86a70b1ee812ab917e0d42e85b8457a68b2c7eab0537f1be01d83c53052b7b` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Representation | Source fold0 | Source fold1 | Source fold2 |
| --- | --- | --- | --- |
| Mean training spectrum | 32.56 | 131.42 | 23.08 |
| Nearest training spectrum | 4.68 | 69.07 | 3.87 |
| Linear PCA,3dimensions | 4.06 | 9.00 | 3.64 |
| Log PCA,3dimensions | 3.93 | 9.92 | 4.08 |
| Linear PCA,8dimensions | 2.99 | 3.41 | 2.16 |
| Log PCA,8dimensions | 3.65 | 6.14 | 3.33 |

### Таблица 2

| Source pair | Shared illuminant | Shared shape + local exposure | Independent regions |
| --- | --- | --- | --- |
| 0/1 | 12.95 | 9.17 | 7.34 |
| 0/2 | 10.20 | 5.49 | 4.33 |
| 1/2 | 17.00 | 5.92 | 4.76 |
