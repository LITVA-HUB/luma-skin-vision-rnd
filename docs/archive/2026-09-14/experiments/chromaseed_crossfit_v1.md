# chromaseed_crossfit_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_crossfit_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_crossfit_v1/report.md)

SHA-256 отчёта: `8aa750792b4ac44898cd0c6eecd605baa5d7066f4878407451c9c2ae2a3bb23a`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_crossfit_model_card.md](../../../../docs/architecture/chromaseed_crossfit_model_card.md)
- [chromaseed_crossfit_next_decision.md](../../../../docs/research/chromaseed_crossfit_next_decision.md)
- [chromaseed_crossfit_v1_protocol.md](../../../../docs/research/chromaseed_crossfit_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_crossfit_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_crossfit_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_crossfit.py](../modules/scripts__chromaseed_crossfit.md)
- [scripts/chromaseed_crossfit_audit.py](../modules/scripts__chromaseed_crossfit_audit.md)
- [scripts/chromaseed_crossfit_reference.py](../modules/scripts__chromaseed_crossfit_reference.md)
- [scripts/chromaseed_crossfit_report.py](../modules/scripts__chromaseed_crossfit_report.md)
- [scripts/chromaseed_crossfit_runtime.py](../modules/scripts__chromaseed_crossfit_runtime.md)
- [scripts/chromaseed_crossfit_train.py](../modules/scripts__chromaseed_crossfit_train.md)
- [scripts/chromaseed_crossfit_verify.py](../modules/scripts__chromaseed_crossfit_verify.md)
- [tests/test_chromaseed_crossfit.py](../tests/tests__test_chromaseed_crossfit.md)
- [tests/test_chromaseed_crossfit_reference.py](../tests/tests__test_chromaseed_crossfit_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_crossfit_v1/audit.json) | {"passed": true} | `3ead4c746a2fe14b6a7c33012badcf8d7531ac28110a6a9ac437225951e43184` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_crossfit_v1/runtime.json) | {"scope": "183 actual one-record consumers;20 warmups/3 passes.216 complete fits,96 C include n-person excluded-teacher pool plus native tables/routing and full student geometry/readout/export;120 same-run H controls. No imports/I/O/grid selection/image extraction/phone or GPU timing. Cached arrays exclude process, temporaries and retained caller objects. One CPU thread, no concurrent heavy work."} | `25ac183a9e1c2e7987caa65a493097f94076e5b4f7b8d7761139add17b163469` |
| [summary.json](../../../../docs/benchmarks/chromaseed_crossfit_v1/summary.json) | {} | `cf6b5a50260d11ce0a3f7830f93d88ac4560e48d536f3560a63d5dff90ebd936` |
| [verification.json](../../../../docs/benchmarks/chromaseed_crossfit_v1/verification.json) | {"passed": true} | `80c4129565931e204d663d4e833d8513e7d70111b47c965b6ffe63f2eb2c18dd` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Perceptual-вариант | Mixed ΔE00 | SLR→iPod | iPod→SLR | Fit734, мс | Ответ mixed, мкс |
| --- | --- | --- | --- | --- | --- |
| Исходная без поправки | 5.272642 | 8.654805 | 8.386889 | 35.22 | 12.3 |
| Поправка H | 5.185283 | 8.922499 | 8.627988 | 46.44 | 22.9 |
| Поправка: знакомый человек | 5.174110 | 8.964355 | 8.656877 | 640.56 | 22.9 |
| Поправка: исключённый человек | 5.508646 | 8.918466 | 8.721196 | 649.75 | 22.8 |
| Взвешенная H | 5.239053 | 8.987209 | 8.433364 | 47.12 | 24.5 |
| Взвешенная: знакомый | 5.231517 | 9.037662 | 8.440880 | 637.61 | 24.5 |
| Взвешенная: исключённый | 5.212255 | 8.989075 | 8.435833 | 639.88 | 24.4 |
