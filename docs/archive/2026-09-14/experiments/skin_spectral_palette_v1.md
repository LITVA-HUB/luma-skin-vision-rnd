# skin_spectral_palette_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

См. полный отчёт, протокол и журнал решения; количественные результаты не экстраполируются на другие задачи.

[Полная папка артефактов](../../../../docs/benchmarks/skin_spectral_palette_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_spectral_palette_v1/report.md)

SHA-256 отчёта: `6de4d50ade3c9182114bcdd942c236090b7e65570ce008d61f4bad3db4e180a5`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_spectral_palette_next_decision.md](../../../../docs/research/skin_spectral_palette_next_decision.md)
- [skin_spectral_palette_v1_protocol.md](../../../../docs/research/skin_spectral_palette_v1_protocol.md)
- [report.md](../../../../docs/benchmarks/skin_spectral_palette_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_spectral_palette.py](../modules/scripts__skin_spectral_palette.md)
- [scripts/skin_spectral_palette_audit.py](../modules/scripts__skin_spectral_palette_audit.md)
- [tests/test_skin_spectral_palette.py](../tests/tests__test_skin_spectral_palette.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Артефакт | SHA256 |
| --- | --- |
| protocol.json | 11a08261e738f99f2dc718e38ee16a9965a5d3d923252a8ce801147443bf763e |
| palette.npz | 9e5bb25250258fb79a547b3ac6dc93792d1808967fe5842e21b42f57a6cc2112 |
| profile.json | 008034fd2c82648fa3aa316d37c35ce3a70320681bd40810c5b62472147f97f6 |
| audit.json | 7df3009f561f1ba6992656777e7266ecaebfb46f63b88415dc0d7e7ddcaf8ec5 |
