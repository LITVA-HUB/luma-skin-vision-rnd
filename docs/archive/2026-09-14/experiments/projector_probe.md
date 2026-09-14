# projector_probe

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Проверка геометрии допустимых поправок; привилегированный диагностический контроль, не skin accuracy.

[Полная папка артефактов](../../../../docs/benchmarks/projector_probe)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/projector_probe_report.md)

SHA-256 отчёта: `7ecfaa96ebc807b0207c50a382b36173f9232901e7c79463d15365e7c822499c`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [projector_probe_report.md](../../../../docs/benchmarks/projector_probe_report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/cc_projector_probe.py](../modules/scripts__cc_projector_probe.md)
- [tests/test_projector_probe.py](../tests/tests__test_projector_probe.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [config.json](../../../../docs/benchmarks/projector_probe/config.json) | {"scope": "TRAIN ONLY; GT-assisted oracle, no new learned model or held-out accuracy"} | `908d7e858d438c9c24877707d84405cd0d6213fc1a788970ca15c5b6ff6f0f97` |
| [manifest.json](../../../../docs/benchmarks/projector_probe/manifest.json) | {} | `a20ed567a2d289e9f6a71ae6f78dc6373828afb6320f88ee28ccde7a3c493dc2` |
| [results.json](../../../../docs/benchmarks/projector_probe/results.json) | {"scope": "TRAIN-ONLY REPRESENTATION DIAGNOSTIC; oracles use GT; no deployable accuracy result", "seconds": 3.8496161999937613} | `bd1685f87aecd5724900d8fa24586af9df93f6bd75792a3e9ff939513505c7be` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Patch grid | Minimum recovery error, GT oracle ° | Feasible oracle reproduction ° | GT outside positive cone |
| --- | --- | --- | --- |
| 4 × 4 | 1.569883 | 2.062213 | 76.73% |
| 8 × 8 | 1.027486 | 1.338250 | 53.82% |
| 16 × 16 | 0.675077 | 0.880496 | 34.81% |
