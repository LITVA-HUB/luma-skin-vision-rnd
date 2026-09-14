# skin_risk_cross_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Перекрёстные сочетания цвета и риска: 90 endpoints из существующих моделей, не 90 новых обучений.

[Полная папка артефактов](../../../../docs/benchmarks/skin_risk_cross_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_risk_cross_v1/report.md)

SHA-256 отчёта: `2503f884c5e5ead3b1582f4cca251b0bea3c08b61ad17418816246806b6d7eb3`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_risk_cross_protocol_v1.md](../../../../docs/research/skin_risk_cross_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_risk_cross_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_risk_cross.py](../modules/scripts__skin_risk_cross.md)
- [scripts/skin_risk_cross_report.py](../modules/scripts__skin_risk_cross_report.md)
- [tests/test_skin_risk_cross.py](../tests/tests__test_skin_risk_cross.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [lock.json](../../../../docs/benchmarks/skin_risk_cross_v1/lock.json) | {} | `13891b511103c68cd6117a4c1c968ad6f887794dc7d58031decd0d006937e412` |
| [results.json](../../../../docs/benchmarks/skin_risk_cross_v1/results.json) | {"scope": "POSTHOC SOURCE; no fitting, no calibrated C+"} | `061ad39e8c48d838813c125f4b6eb6ce9e50b5faaa736f23c96c1fb7566a8666` |
| [summary.json](../../../../docs/benchmarks/skin_risk_cross_v1/summary.json) | {} | `cf0c63ad35ae32e02f054172657bdd0602fc88dfe3e4408bebfb3f645b08ab60` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | System | Mean | p95 | At80% | p95at80% | Models |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | ordinary_dispersion | 3.4406 | 6.9316 | 3.4407 | 6.8591 | 1 |
| mixed | ordinary_gaussian_expected | 3.4406 | 6.9316 | 3.2621 | 6.5107 | 2 |
| mixed | ordinary_gaussian_covariance | 3.4406 | 6.9316 | 3.2659 | 6.7278 | 2 |
| mixed | ordinary_cross_disagreement | 3.4406 | 6.9316 | 3.2875 | 6.5671 | 2 |
| mixed | ordinary_pair_disagreement | 3.3680 | 6.9395 | 3.4018 | 6.8602 | 2 |
| mixed | ordinary_pair_dispersion | 3.3680 | 6.9395 | 3.3625 | 6.7866 | 2 |
| mixed | mixed_pair_expected | 3.3788 | 6.9971 | 3.1977 | 6.7769 | 2 |
| mixed | mixed_pair_covariance | 3.3788 | 6.9971 | 3.1954 | 6.8067 | 2 |
| mixed | mixed_pair_disagreement | 3.3788 | 6.9971 | 3.2429 | 6.6265 | 2 |
| mixed | gaussian_expected | 3.4605 | 7.5348 | 3.2593 | 7.0951 | 1 |
| from_SLR | ordinary_dispersion | 5.0193 | 9.7825 | 5.1087 | 9.6635 | 1 |
| from_SLR | ordinary_gaussian_expected | 5.0193 | 9.7825 | 4.9293 | 9.7357 | 2 |
| from_SLR | ordinary_gaussian_covariance | 5.0193 | 9.7825 | 4.9639 | 9.6683 | 2 |
| from_SLR | ordinary_cross_disagreement | 5.0193 | 9.7825 | 5.0409 | 9.9071 | 2 |
| from_SLR | ordinary_pair_disagreement | 4.8747 | 9.4964 | 4.8547 | 9.5500 | 2 |
| from_SLR | ordinary_pair_dispersion | 4.8747 | 9.4964 | 5.0339 | 9.6907 | 2 |
| from_SLR | mixed_pair_expected | 5.1263 | 10.1101 | 5.0262 | 10.0203 | 2 |
| from_SLR | mixed_pair_covariance | 5.1263 | 10.1101 | 5.0250 | 10.0203 | 2 |
| from_SLR | mixed_pair_disagreement | 5.1263 | 10.1101 | 5.2910 | 10.3901 | 2 |
| from_SLR | gaussian_expected | 5.6571 | 11.2796 | 5.4955 | 10.9031 | 1 |
| from_ipod | ordinary_dispersion | 5.9635 | 11.0896 | 5.7914 | 10.1810 | 1 |
| from_ipod | ordinary_gaussian_expected | 5.9635 | 11.0896 | 6.1737 | 11.1575 | 2 |
| from_ipod | ordinary_gaussian_covariance | 5.9635 | 11.0896 | 6.1282 | 11.2014 | 2 |
| from_ipod | ordinary_cross_disagreement | 5.9635 | 11.0896 | 6.0235 | 10.5178 | 2 |
| from_ipod | ordinary_pair_disagreement | 5.9025 | 10.7281 | 5.8333 | 10.3266 | 2 |
| from_ipod | ordinary_pair_dispersion | 5.9025 | 10.7281 | 5.6588 | 9.9522 | 2 |
| from_ipod | mixed_pair_expected | 5.7134 | 10.5926 | 5.9451 | 10.9189 | 2 |
| from_ipod | mixed_pair_covariance | 5.7134 | 10.5926 | 5.9368 | 10.9189 | 2 |
| from_ipod | mixed_pair_disagreement | 5.7134 | 10.5926 | 5.8009 | 10.4063 | 2 |
| from_ipod | gaussian_expected | 5.7474 | 10.2485 | 5.9347 | 10.3551 | 1 |

### Таблица 2

| Protocol | Candidate vs control | Difference at80% | Patient95% interval |
| --- | --- | --- | --- |
| mixed | mixed_pair_expected vs ordinary_pair_dispersion | -0.1648 | [-0.4428,0.1349] |
| mixed | mixed_pair_covariance vs ordinary_pair_dispersion | -0.1671 | [-0.4692,0.1323] |
| mixed | ordinary_gaussian_expected vs ordinary_dispersion | -0.1785 | [-0.4763,0.0949] |
| from_SLR | mixed_pair_expected vs ordinary_pair_dispersion | -0.0077 | [-0.2263,0.4876] |
| from_SLR | mixed_pair_covariance vs ordinary_pair_dispersion | -0.0089 | [-0.2523,0.4739] |
| from_SLR | ordinary_gaussian_expected vs ordinary_dispersion | -0.1794 | [-0.2693,0.1297] |
| from_ipod | mixed_pair_expected vs ordinary_pair_dispersion | 0.2863 | [-0.0252,0.5385] |
| from_ipod | mixed_pair_covariance vs ordinary_pair_dispersion | 0.2780 | [-0.0314,0.4869] |
| from_ipod | ordinary_gaussian_expected vs ordinary_dispersion | 0.3822 | [0.1345,0.6874] |
