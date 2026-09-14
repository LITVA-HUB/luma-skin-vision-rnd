# chromaseed_neural_prefix_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_neural_prefix_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/report.md)

SHA-256 отчёта: `b15e21db87671adb38f83215d77f871cb03efc141626465288e1c2ecb661656d`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_neural_prefix_model_card.md](../../../../docs/architecture/chromaseed_neural_prefix_model_card.md)
- [chromaseed_neural_prefix_next_decision.md](../../../../docs/research/chromaseed_neural_prefix_next_decision.md)
- [chromaseed_neural_prefix_v1_protocol.md](../../../../docs/research/chromaseed_neural_prefix_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_neural_prefix_audit.py](../modules/scripts__chromaseed_neural_prefix_audit.md)
- [scripts/chromaseed_neural_prefix_numpy.py](../modules/scripts__chromaseed_neural_prefix_numpy.md)
- [scripts/chromaseed_neural_prefix_report.py](../modules/scripts__chromaseed_neural_prefix_report.md)
- [scripts/chromaseed_neural_prefix_run.py](../modules/scripts__chromaseed_neural_prefix_run.md)
- [scripts/chromaseed_neural_prefix_runtime.py](../modules/scripts__chromaseed_neural_prefix_runtime.md)
- [tests/test_chromaseed_neural_prefix.py](../tests/tests__test_chromaseed_neural_prefix.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/audit.json) | {"passed": true, "seconds": 37.13716939999722} | `6024d35a02f43c80c184633226492550e87e7b7dcff0c5dac3b9ba955b87cf42` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/runtime.json) | {"seconds": 33.339150900021195} | `b5a7ab55ed7e20124ba2910d0ee257ea21ed5dbfc9659fe25c7145a53a7948b2` |
| [summary.json](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/summary.json) | {} | `1b2f9dbd06b86c210642d5c7dce7e958ed4e7ac211f679d7bbf2ede92baab492` |
| [verification.json](../../../../docs/benchmarks/chromaseed_neural_prefix_v1/verification.json) | {"passed": true} | `f9bd98ddad578678278581ce2fde786ced806e13e1db3a15cea1aa0fd33fe024` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Компактная политика | J: mixed / вперёд / обратно | Числовые байты | ΔE00 mixed | SLR → iPod | iPod → SLR |
| --- | --- | --- | --- | --- | --- |
| plain | 1 / 1 / 1 | 11366 / 11366 / 11366 | 5.63659 | 9.08204 | 9.25791 |
| local2 | 1 / 1 / 1 | 5446 / 5446 / 5446 | 5.63062 | 8.33649 | 9.45806 |
| local4 | 1 / 3 / 2 | 2886 / 8414 / 5650 | 5.86041 | 8.30599 | 8.82204 |
| blind4 | 2 / 3 / 2 | 2886 / 2886 / 2886 | 5.77054 | 8.29435 | 8.58953 |
| e2e4 | 1 / 1 / 1 | 2886 / 2886 / 2886 | 5.86584 | 8.88825 | 7.76477 |
| fg_norm_static | — | 20284 | 5.43865 | 8.59700 | 8.70502 |
| random_head | — | 10564 | 6.04816 | 9.28695 | 8.79045 |

### Таблица 2

| Политика минимальной внутренней ошибки | J mixed / вперёд / обратно | ΔE00 mixed | SLR → iPod | iPod → SLR |
| --- | --- | --- | --- | --- |
| plain | 1 / 1 / 1 | 5.63659 | 9.08204 | 9.25791 |
| local2 | 2 / 1 / 2 | 5.72584 | 8.33649 | 9.39577 |
| local4 | 4 / 3 / 4 | 5.70054 | 8.30599 | 8.32429 |
| blind4 | 2 / 3 / 2 | 5.77054 | 8.29435 | 8.58953 |
| e2e4 | 3 / 3 / 3 | 5.73097 | 8.43298 | 8.62946 |

### Таблица 3

| Вариант compact | Ответ CPU, мкс: mixed / вперёд / обратно | Полное обучение + экспорт GPU, мс |
| --- | --- | --- |
| plain | 6.5 / 6.5 / 6.5 | 118.44 / 278.53 / 282.82 |
| local2 | 6.4 / 6.4 / 6.4 | 123.40 / 314.24 / 316.68 |
| local4 | 6.3 / 14.6 / 10.1 | 125.13 / 317.81 / 323.43 |
| blind4 | 6.3 / 6.3 / 6.3 | 308.84 / 304.21 / 306.83 |
| e2e4 | 6.3 / 6.3 / 6.3 | 258.59 / 803.43 / 256.76 |
