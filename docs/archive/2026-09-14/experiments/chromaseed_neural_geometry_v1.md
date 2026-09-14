# chromaseed_neural_geometry_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_neural_geometry_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_neural_geometry_v1/report.md)

SHA-256 отчёта: `f7c084c9b3a30a73823bfad70f3e91faf1225035fd7d83d25079460338355001`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_neural_geometry_v1_protocol.md](../../../../docs/research/chromaseed_neural_geometry_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_neural_geometry_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_neural_geometry.py](../modules/scripts__chromaseed_neural_geometry.md)
- [scripts/chromaseed_neural_geometry_reference.py](../modules/scripts__chromaseed_neural_geometry_reference.md)
- [scripts/chromaseed_neural_geometry_report.py](../modules/scripts__chromaseed_neural_geometry_report.md)
- [scripts/chromaseed_neural_geometry_run.py](../modules/scripts__chromaseed_neural_geometry_run.md)
- [tests/test_chromaseed_neural_geometry.py](../tests/tests__test_chromaseed_neural_geometry.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [verification.json](../../../../docs/benchmarks/chromaseed_neural_geometry_v1/verification.json) | {"passed": true} | `57b1d6b00b51e9f761ad34b2062f28a1186cd403f0ba40a52e24b0d105014e25` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Роль | Вход | Цель | df при .1 | при1 | при10 | при100 | при1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mixed | raw36 | norm | 63.872 | 57.360 | 39.786 | 21.219 | 9.080 |
| mixed | raw36 | perceptual | 62.494 | 53.090 | 35.412 | 18.694 | 7.868 |
| mixed | mean3 | norm | 50.086 | 33.999 | 19.736 | 10.998 | 5.986 |
| mixed | mean3 | perceptual | 45.626 | 30.262 | 17.857 | 10.062 | 5.369 |
| slr_to_ipod | raw36 | norm | 58.820 | 44.842 | 27.099 | 13.741 | 5.631 |
| slr_to_ipod | raw36 | perceptual | 54.972 | 39.986 | 23.693 | 11.689 | 4.839 |
| slr_to_ipod | mean3 | norm | 33.773 | 21.467 | 13.206 | 7.785 | 4.117 |
| slr_to_ipod | mean3 | perceptual | 30.577 | 19.454 | 11.887 | 6.873 | 3.586 |
| ipod_to_slr | raw36 | norm | 64.174 | 58.771 | 41.642 | 22.332 | 9.263 |
| ipod_to_slr | raw36 | perceptual | 62.923 | 54.079 | 36.320 | 19.060 | 7.850 |
| ipod_to_slr | mean3 | norm | 54.658 | 37.963 | 21.608 | 11.728 | 6.413 |
| ipod_to_slr | mean3 | perceptual | 50.235 | 33.777 | 19.332 | 10.565 | 5.489 |
