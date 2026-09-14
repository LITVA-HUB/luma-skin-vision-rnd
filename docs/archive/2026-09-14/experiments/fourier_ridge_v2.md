# fourier_ridge_v2

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Отмена глобального prior и проверка границы регуляризации; отдельный source screen.

[Полная папка артефактов](../../../../docs/benchmarks/fourier_ridge_v2)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/fourier_representation_report.md)

SHA-256 отчёта: `c518e86e74b30be08e962a34db3753a6d833de6921ad6bc0289e604deb2f52df`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [fourier_ridge_protocol.md](../../../../docs/research/fourier_ridge_protocol.md)
- [fourier_ridge_v2_protocol.md](../../../../docs/research/fourier_ridge_v2_protocol.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [docs/benchmarks/fourier_ridge_v1/scripts/cc_fourier_ridge.py](../modules/docs__benchmarks__fourier_ridge_v1__scripts__cc_fourier_ridge.md)
- [docs/benchmarks/fourier_ridge_v2/scripts/cc_fourier_ridge_v2.py](../modules/docs__benchmarks__fourier_ridge_v2__scripts__cc_fourier_ridge_v2.md)
- [scripts/cc_fourier_ridge.py](../modules/scripts__cc_fourier_ridge.md)
- [scripts/cc_fourier_ridge_v2.py](../modules/scripts__cc_fourier_ridge_v2.md)
- [tests/test_fourier_ridge.py](../tests/tests__test_fourier_ridge.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [verification.json](../../../../docs/benchmarks/fourier_ridge_v2/verification.json) | {"status": "ALL HASHES AND REPLAYS PASSED"} | `29b5458788d683ca84c502da7a7b0d0e2196c937a0edca087f0b94635f92c477` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Phase | Candidate | Active real coefficients | Mean reproduction° | Raw risk80° | Fit+validation seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | sigma1.0_ridge0.001_gray_light | 12288 | 6.8843 | 3.6743 | 0.515 |
| 1 | sigma1.0_ridge0.001_gray_world | 12288 | 5.9448 | 3.3655 | 0.515 |
| 1 | sigma1.0_ridge0.01_gray_light | 12288 | 6.9308 | 3.3882 | 0.512 |
| 1 | sigma1.0_ridge0.01_gray_world | 12288 | 6.0306 | 3.0812 | 0.512 |
| 1 | sigma1.0_ridge0.1_gray_light | 12288 | 7.2242 | 3.2708 | 0.499 |
| 1 | sigma1.0_ridge0.1_gray_world | 12288 | 6.4660 | 3.2708 | 0.499 |
| 1 | sigma1.0_ridge1.0_gray_light | 12288 | 6.8464 | 2.9578 | 0.473 |
| 1 | sigma1.0_ridge1.0_gray_world | 12288 | 6.0778 | 2.9578 | 0.473 |
| 1 | sigma2.0_ridge0.001_gray_light | 12288 | 4.0918 | 3.2096 | 0.528 |
| 1 | sigma2.0_ridge0.001_gray_world | 12288 | 2.7468 | 2.0714 | 0.528 |
| 1 | sigma2.0_ridge0.01_gray_light | 12288 | 4.1040 | 3.2336 | 0.461 |
| 1 | sigma2.0_ridge0.01_gray_world | 12288 | 2.7655 | 2.1005 | 0.461 |
| 1 | sigma2.0_ridge0.1_gray_light | 12288 | 4.2702 | 2.7312 | 0.439 |
| 1 | sigma2.0_ridge0.1_gray_world | 12288 | 2.9693 | 2.1399 | 0.439 |
| 1 | sigma2.0_ridge1.0_gray_light | 12288 | 4.5370 | 2.3331 | 0.443 |
| 1 | sigma2.0_ridge1.0_gray_world | 12288 | 3.2636 | 2.3331 | 0.443 |
| 2 | sigma2.0_ridge1e-05_gray_world_biasTrue | 12288 | 2.7446 | 2.0674 | 0.718 |
| 2 | sigma2.0_ridge0.0001_gray_world_biasTrue | 12288 | 2.7448 | 2.0678 | 0.747 |
| 2 | sigma2.0_ridge0.001_gray_world_biasTrue | 12288 | 2.7468 | 2.0714 | 0.585 |
| 2 | sigma2.0_ridge1e-05_gray_world_biasFalse | 8192 | 3.3792 | 3.0981 | 0.489 |
| 2 | sigma2.0_ridge0.0001_gray_world_biasFalse | 8192 | 3.3792 | 3.0983 | 0.438 |
| 2 | sigma2.0_ridge0.001_gray_world_biasFalse | 8192 | 3.3794 | 3.0999 | 0.440 |
| 2 | sigma4.0_ridge1e-05_gray_world_biasTrue | 12288 | 2.8727 | 2.7325 | 0.447 |
| 2 | sigma4.0_ridge0.0001_gray_world_biasTrue | 12288 | 2.8726 | 2.7326 | 0.420 |
| 2 | sigma4.0_ridge0.001_gray_world_biasTrue | 12288 | 2.8720 | 2.7337 | 0.431 |
| 2 | sigma4.0_ridge1e-05_gray_world_biasFalse | 8192 | 3.3954 | 3.1634 | 0.434 |
| 2 | sigma4.0_ridge0.0001_gray_world_biasFalse | 8192 | 3.3956 | 3.1638 | 0.434 |
| 2 | sigma4.0_ridge0.001_gray_world_biasFalse | 8192 | 3.3976 | 3.1674 | 0.433 |
| 2 | sigma8.0_ridge1e-05_gray_world_biasTrue | 12288 | 3.1365 | 2.9817 | 0.809 |
| 2 | sigma8.0_ridge0.0001_gray_world_biasTrue | 12288 | 3.1364 | 2.9818 | 0.768 |
| 2 | sigma8.0_ridge0.001_gray_world_biasTrue | 12288 | 3.1353 | 2.9821 | 0.426 |
| 2 | sigma8.0_ridge1e-05_gray_world_biasFalse | 8192 | 3.4523 | 3.2233 | 0.768 |
| 2 | sigma8.0_ridge0.0001_gray_world_biasFalse | 8192 | 3.4525 | 3.2235 | 0.435 |
| 2 | sigma8.0_ridge0.001_gray_world_biasFalse | 8192 | 3.4541 | 3.2254 | 0.434 |
