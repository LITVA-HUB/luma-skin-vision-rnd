# chromaseed_local_denoise_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_local_denoise_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_local_denoise_v1/report.md)

SHA-256 отчёта: `9efb2ea5bdccb51b1e617a046af461028b7de2d7381b0c4cedcea6c0931c0f5d`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_local_denoise_model_card.md](../../../../docs/architecture/chromaseed_local_denoise_model_card.md)
- [chromaseed_local_denoise_next_decision.md](../../../../docs/research/chromaseed_local_denoise_next_decision.md)
- [chromaseed_local_denoise_v1_protocol.md](../../../../docs/research/chromaseed_local_denoise_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_local_denoise_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_local_denoise.py](../modules/scripts__chromaseed_local_denoise.md)
- [scripts/chromaseed_local_denoise_audit.py](../modules/scripts__chromaseed_local_denoise_audit.md)
- [scripts/chromaseed_local_denoise_fit.py](../modules/scripts__chromaseed_local_denoise_fit.md)
- [scripts/chromaseed_local_denoise_numpy.py](../modules/scripts__chromaseed_local_denoise_numpy.md)
- [scripts/chromaseed_local_denoise_report.py](../modules/scripts__chromaseed_local_denoise_report.md)
- [scripts/chromaseed_local_denoise_runtime.py](../modules/scripts__chromaseed_local_denoise_runtime.md)
- [scripts/chromaseed_local_denoise_train.py](../modules/scripts__chromaseed_local_denoise_train.md)
- [tests/test_chromaseed_local_denoise.py](../tests/tests__test_chromaseed_local_denoise.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_local_denoise_v1/audit.json) | {"passed": true, "seconds": 27.771717299998272} | `ae29d3272acfd28205b4028019fc8bc5f8f372499ad2008acd11dbd860d7927c` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_local_denoise_v1/runtime.json) | {"seconds": 14.299679999996442} | `30afe0965f30fbb9375ef78e714905c55a45fba49994b2e9af7f8a28560d8ca5` |
| [summary.json](../../../../docs/benchmarks/chromaseed_local_denoise_v1/summary.json) | {} | `8385d3949496ad505bd380a8c7f9ee299e10f0ecdfb86a93c7bcaa12c67b7be0` |
| [verification.json](../../../../docs/benchmarks/chromaseed_local_denoise_v1/verification.json) | {"passed": true} | `36442da6dabe927b4056ef5532fc7a6dd3956c14eb8ed35736a3f161ac3e2376` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Модель | Mixed ΔE00 | SLR → iPod | iPod → SLR | Числовые байты |
| --- | --- | --- | --- | --- |
| plain | 5.636588 | 9.082035 | 9.257910 | 11,364 |
| local2 | 5.725844 | 8.705199 | 9.395766 | 11,344 |
| local4 | 5.700539 | 8.495492 | 8.324291 | 11,368 |
| blind4 | 5.782196 | 8.685042 | 8.607785 | 11,368 |
| e2e4 | 5.696842 | 8.621149 | 8.026476 | 11,368 |
| fg_norm_static | 5.438652 | 8.597000 | 8.705018 | 20,284 |
| random_head | 6.048158 | 9.286951 | 8.790445 | 10,564 |

### Таблица 2

| Модель | CPU ответ, мкс | Полное обучение GPU, мс |
| --- | --- | --- |
| plain | 9.00 | 99.498 |
| local2 | 15.00 | 98.268 |
| local4 | 25.50 | 102.997 |
| blind4 | 28.40 | 275.210 |
| e2e4 | 25.50 | 228.336 |
| fg_norm_static | 10.00 | не перемерялось |
| random_head | 5.50 | не перемерялось |
