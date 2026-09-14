# skin_color_sampling_mass_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Шесть дополнительных контролей массы людей к sampling screen; включены в общий отчёт, не считать дважды.

[Полная папка артефактов](../../../../docs/benchmarks/skin_color_sampling_mass_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_color_sampling_v1/report.md)

SHA-256 отчёта: `356317b865a968d0ee88d1e3ad939bd280d7601054d5f7e2142360b8d90f9702`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_color_sampling_mass_protocol_v1.md](../../../../docs/research/skin_color_sampling_mass_protocol_v1.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_color_sampling_mass.py](../modules/scripts__skin_color_sampling_mass.md)
- [scripts/skin_color_sampling_mass_train.py](../modules/scripts__skin_color_sampling_mass_train.md)
- [scripts/skin_color_sampling_mass_verify.py](../modules/scripts__skin_color_sampling_mass_verify.md)
- [tests/test_skin_color_sampling_mass.py](../tests/tests__test_skin_color_sampling_mass.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_color_sampling_mass_v1/audit.json) | {"status": "PASS"} | `2f27b2b35720c5fecb7d601b20f1620f115144e459d9e96e1a70a1b22d79286c` |
| [source_lock.json](../../../../docs/benchmarks/skin_color_sampling_mass_v1/source_lock.json) | {} | `88788dd86eb23b7bf11173b738095e57a1af1bba449b0af46d182bee0af88976` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Sampler | Mean | Median | p95 | Patient mean | Site mean | At80% | Above10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| image | 6.1082 | 5.1353 | 14.9960 | 5.9237 | 6.0666 | 5.3918 | 15.37% |
| person_site | 6.0084 | 4.9119 | 14.3638 | 5.8129 | 5.9673 | 5.3496 | 14.37% |
| site | 6.2240 | 5.3056 | 15.2028 | 6.0072 | 6.1811 | 5.4664 | 16.38% |
| color | 6.0012 | 5.1767 | 14.2733 | 5.8156 | 5.9630 | 5.3187 | 14.22% |
| color_ipw | 6.1729 | 5.1754 | 14.6528 | 5.9757 | 6.1312 | 5.4306 | 16.09% |
| person_mass | 6.1407 | 5.1396 | 14.7468 | 5.9432 | 6.0991 | 5.4385 | 14.51% |
| within_person_shuffle | 6.1522 | 5.2643 | 14.6378 | 5.9447 | 6.1089 | 5.4265 | 15.37% |

### Таблица 2

| Color vs control | Image difference | Patient difference | Patient interval |
| --- | --- | --- | --- |
| image | -0.1071 | -0.1080 | [-0.2513, 0.0096] |
| person_site | -0.0073 | 0.0028 | [-0.0624, 0.0743] |
| site | -0.2228 | -0.1916 | [-0.4534, 0.0117] |
| color_ipw | -0.1717 | -0.1600 | [-0.3958, 0.0081] |
| person_mass | -0.1395 | -0.1275 | [-0.3003, 0.0178] |
| within_person_shuffle | -0.1510 | -0.1290 | [-0.2927, -0.0058] |
