# chromaseed_widen_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_widen_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_widen_v1/report.md)

SHA-256 отчёта: `4cf8e563fb89ce720aeb39dbdc3f5cda2aa3aec5d134b7f7a025e81e51602aba`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_widen_early_model_card.md](../../../../docs/architecture/chromaseed_widen_early_model_card.md)
- [chromaseed_widen_model_card.md](../../../../docs/architecture/chromaseed_widen_model_card.md)
- [chromaseed_widen_early_next_decision.md](../../../../docs/research/chromaseed_widen_early_next_decision.md)
- [chromaseed_widen_early_v1_protocol.md](../../../../docs/research/chromaseed_widen_early_v1_protocol.md)
- [chromaseed_widen_next_decision.md](../../../../docs/research/chromaseed_widen_next_decision.md)
- [chromaseed_widen_v1_protocol.md](../../../../docs/research/chromaseed_widen_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_widen_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_widen.py](../modules/scripts__chromaseed_widen.md)
- [scripts/chromaseed_widen_audit.py](../modules/scripts__chromaseed_widen_audit.md)
- [scripts/chromaseed_widen_early_audit.py](../modules/scripts__chromaseed_widen_early_audit.md)
- [scripts/chromaseed_widen_early_fit.py](../modules/scripts__chromaseed_widen_early_fit.md)
- [scripts/chromaseed_widen_early_report.py](../modules/scripts__chromaseed_widen_early_report.md)
- [scripts/chromaseed_widen_early_run.py](../modules/scripts__chromaseed_widen_early_run.md)
- [scripts/chromaseed_widen_early_runtime.py](../modules/scripts__chromaseed_widen_early_runtime.md)
- [scripts/chromaseed_widen_report.py](../modules/scripts__chromaseed_widen_report.md)
- [scripts/chromaseed_widen_run.py](../modules/scripts__chromaseed_widen_run.md)
- [scripts/chromaseed_widen_runtime.py](../modules/scripts__chromaseed_widen_runtime.md)
- [tests/test_chromaseed_widen.py](../tests/tests__test_chromaseed_widen.md)
- [tests/test_chromaseed_widen_audit.py](../tests/tests__test_chromaseed_widen_audit.md)
- [tests/test_chromaseed_widen_early.py](../tests/tests__test_chromaseed_widen_early.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_widen_v1/audit.json) | {"passed": true, "seconds": 96.20530750002945} | `1f8898e310f4c5563170c150af3e938cccf9a180f78aad7cffc586f25b186e49` |
| [checks.json](../../../../docs/benchmarks/chromaseed_widen_v1/checks.json) | {"passed": true} | `e5f3d881d89cd883d1abb35e386036a2332b75ab06734921f4ee7b07ea1a14cc` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_widen_v1/runtime.json) | {"passed": true, "seconds": 171.0781110000098} | `e6259ecb48070c35fb74a321c18afaceda68739ed23ca3d254d5d9b127ef7ce4` |
| [summary.json](../../../../docs/benchmarks/chromaseed_widen_v1/summary.json) | {} | `6ceea1c8722bd71b574e16c21489cd64084d1bbe849352de4025db662f36bd96` |
| [verification.json](../../../../docs/benchmarks/chromaseed_widen_v1/verification.json) | {"passed": true} | `d86795ebc9659fdefe869298fa8a44106555e5ac90a5fa153105d408be3b4876` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Модель | Параметры | Данные модели, КБ | Смешанная группа | SLR → iPod | iPod → SLR | CPU, мкс |
| --- | --- | --- | --- | --- | --- | --- |
| np | 643 | 2.89 | 5.7705 | 8.2943 | 8.5895 | 6.4 |
| tiny | 1,179 | 5.17 | 5.5107 | 8.4355 | 8.8056 | 25.0 |
| m31 | 30,915 | 124.12 | 5.5340 | 8.3989 | 8.7331 | 33.2 |
| m61 | 60,611 | 242.90 | 5.5066 | 8.4086 | 8.6773 | 36.0 |
| m111 | 110,979 | 444.37 | 5.4739 | 8.2647 | 8.6865 | 47.7 |
| m832 | 832,259 | 3329.49 | 5.3223 | 8.6801 | 8.3243 | 110.3 |

### Таблица 2

| Роль | Модель | Шаги | Скорость обучения | Ошибка внутри | Ошибка снаружи |
| --- | --- | --- | --- | --- | --- |
| mixed | m111 | 512 | 0.0001 | 4.1221 | 5.4739 |
| slr_to_ipod | m61 | 512 | 0.0001 | 4.7489 | 8.4086 |
| ipod_to_slr | m111 | 512 | 0.0001 | 4.3898 | 8.6865 |

### Таблица 3

| Модель | Шаги: смешанная / SLR→iPod / iPod→SLR | Полная сборка трёх seed, секунды |
| --- | --- | --- |
| np | 0 / 0 / 0 | 0.896 / 0.903 / 0.900 |
| tiny | 8192 / 2048 / 2048 | 5.881 / 2.211 / 2.201 |
| m31 | 512 / 512 / 2048 | 1.594 / 1.625 / 3.379 |
| m61 | 512 / 512 / 512 | 1.647 / 1.677 / 1.672 |
| m111 | 512 / 512 / 512 | 2.315 / 2.377 / 2.321 |
| m832 | 512 / 512 / 512 | 7.709 / 7.876 / 7.708 |
