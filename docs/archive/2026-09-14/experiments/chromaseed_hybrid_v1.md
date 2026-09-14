# chromaseed_hybrid_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/chromaseed_hybrid_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/chromaseed_hybrid_v1/report.md)

SHA-256 отчёта: `6ecdfa0c7e6acdf6eb5108fc98c1742fca2e77b3fd8df952f79251ee93d5626e`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [chromaseed_hybrid_model_card.md](../../../../docs/architecture/chromaseed_hybrid_model_card.md)
- [chromaseed_hybrid_next_decision.md](../../../../docs/research/chromaseed_hybrid_next_decision.md)
- [chromaseed_hybrid_v1_protocol.md](../../../../docs/research/chromaseed_hybrid_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/chromaseed_hybrid_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/chromaseed_hybrid_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/chromaseed_hybrid.py](../modules/scripts__chromaseed_hybrid.md)
- [scripts/chromaseed_hybrid_audit.py](../modules/scripts__chromaseed_hybrid_audit.md)
- [scripts/chromaseed_hybrid_numpy.py](../modules/scripts__chromaseed_hybrid_numpy.md)
- [scripts/chromaseed_hybrid_reference.py](../modules/scripts__chromaseed_hybrid_reference.md)
- [scripts/chromaseed_hybrid_report.py](../modules/scripts__chromaseed_hybrid_report.md)
- [scripts/chromaseed_hybrid_runtime.py](../modules/scripts__chromaseed_hybrid_runtime.md)
- [scripts/chromaseed_hybrid_train.py](../modules/scripts__chromaseed_hybrid_train.md)
- [scripts/chromaseed_hybrid_verify.py](../modules/scripts__chromaseed_hybrid_verify.md)
- [tests/test_chromaseed_hybrid.py](../tests/tests__test_chromaseed_hybrid.md)
- [tests/test_chromaseed_hybrid_reference.py](../tests/tests__test_chromaseed_hybrid_reference.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/chromaseed_hybrid_v1/audit.json) | {"passed": true} | `c947ca9173426ba3b262596ec06ffcde52e6a064b3fcd32dd2c16106b31db89a` |
| [runtime.json](../../../../docs/benchmarks/chromaseed_hybrid_v1/runtime.json) | {"scope": "Actual111 one-record NumPy consumers,20 warmups/3 passes,144 complete fits including warmups. Includes original weights/moments/exact widths/raw landmarks/gate/readout, plus projection/shared projected basis/branch when needed. Single projected fit also computes raw backbone readout. Excludes imports, I/O, grid search and image/skin extraction. Cached arrays exclude process, temporaries and retained caller objects. One CPU thread, no GPU claim."} | `690dcdeb3b5cceecf14c7df82eb38a2a0ac111c3bcabb124a78166344a56f905` |
| [summary.json](../../../../docs/benchmarks/chromaseed_hybrid_v1/summary.json) | {} | `bc4d08361572d1f24bc6d7490b3dbd2430b0dc59a28dd0d20eb5eda740ea54bd` |
| [verification.json](../../../../docs/benchmarks/chromaseed_hybrid_v1/verification.json) | {"passed": true} | `53883c8debdd447f0237640384439df2a886743b21152e7473cb7d959e242daa` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Вариант (perceptual) | Mixed ΔE00 | SLR→iPod | iPod→SLR | Веса mixed, Б | Ответ mixed, мкс | Fit734, мс |
| --- | --- | --- | --- | --- | --- | --- |
| Исходная | 5.272642 | 8.654805 | 8.386889 | 21,973 | 12.2 | 35.87 |
| Сжатая, общие опоры | 5.202796 | 9.322320 | 9.213744 | 14,181 | 13.6 | 46.19 |
| Смешивание | 5.206195 | 8.800746 | 8.466563 | 27,503 | 23.3 | 44.87 |
| Поправка | 5.185283 | 8.922499 | 8.627988 | 27,503 | 22.7 | 45.41 |
| Взвешенная поправка | 5.239053 | 8.987209 | 8.433364 | 27,503 | 24.4 | 46.69 |
| Сжатая X | 5.206762 | 9.147710 | 9.479015 | 14,181 | 13.8 | — |
