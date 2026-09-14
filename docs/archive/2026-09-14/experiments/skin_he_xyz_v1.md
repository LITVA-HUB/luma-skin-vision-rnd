# skin_he_xyz_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Реальные парные региональные RGB/XYZ, 20 тестовых людей. XYZ RMSE, без выдуманного ΔE00 при неизвестном белом.

[Полная папка артефактов](../../../../docs/benchmarks/skin_he_xyz_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_he_xyz_v1/report.md)

SHA-256 отчёта: `f6f6b071ffb23e8332852b9b04a502b8127b248918f8cc1fa9263e1ba6c59ce0`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_he_xyz_protocol_v1.md](../../../../docs/research/skin_he_xyz_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_he_xyz_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_he_xyz.py](../modules/scripts__skin_he_xyz.md)
- [tests/test_skin_he_xyz.py](../tests/tests__test_skin_he_xyz.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [independent_audit.json](../../../../docs/benchmarks/skin_he_xyz_v1/independent_audit.json) | {"status": "PASS: all14 model replays, independent scalar XYZ metrics and support scores"} | `7e79419377b73b867c1a39e64b12f3b48211c5401d536665f1f49b7833dee5d0` |
| [model_lock.json](../../../../docs/benchmarks/skin_he_xyz_v1/model_lock.json) | {"status": "ALL14 CONTROLS FROZEN BEFORE SKIN TEST NUMERIC EXTRACTION"} | `38d9a903a2137d51f73a68b89351368c1403033d46ce2182b8b7cfe86a3fc7b7` |
| [source_fit.json](../../../../docs/benchmarks/skin_he_xyz_v1/source_fit.json) | {} | `fe95e25fbb011ce1593fa21941c84cd95139920f8fb71fb905ea401185e650b7` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Input | Control | Selected by source CV | Source OOF XYZ RMSE | Test XYZ RMSE | Test80% XYZ RMSE |
| --- | --- | --- | --- | --- | --- |
| raw | constant | False | 4.346908 | 4.147158 | 3.699085 |
| raw | linear3 | False | 2.340966 | 1.866550 | 1.851333 |
| raw | affine4 | False | 2.290256 | 1.830634 | 1.825671 |
| raw | poly2 | False | 2.082153 | 1.999538 | 1.937395 |
| raw | poly3 | True | 2.072378 | 2.025092 | 1.949136 |
| raw | root2 | False | 2.197008 | 1.998954 | 1.950271 |
| raw | mlp_5_25_5 | False | 2.133432 | 1.947308 | 1.961226 |
| jpg | constant | False | 4.346908 | 4.147158 | 3.635247 |
| jpg | linear3 | False | 2.460762 | 2.302701 | 2.133216 |
| jpg | affine4 | False | 2.117998 | 1.860912 | 1.765333 |
| jpg | poly2 | True | 2.047101 | 1.952336 | 1.822263 |
| jpg | poly3 | False | 2.123525 | 1.980025 | 1.896993 |
| jpg | root2 | False | 2.399310 | 2.323195 | 2.126804 |
| jpg | mlp_5_25_5 | False | 2.064161 | 1.920980 | 1.895292 |
