# skin_mskcc_pixels_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Переход от готовых региональных статистик к JPEG-to-Lab; CNN, patch votes, fusion и robust recurrence.

[Полная папка артефактов](../../../../docs/benchmarks/skin_mskcc_pixels_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_mskcc_pixels_v1/report.md)

SHA-256 отчёта: `b392963d3606449f5c7cc503417fec8e5316fddb4b0a53ac8925667fc12add8c`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [report.md](../../../../docs/benchmarks/skin_mskcc_pixels_v1/report.md)
- [reproduce.md](../../../../docs/benchmarks/skin_mskcc_pixels_v1/reproduce.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_mskcc_pixels.py](../modules/scripts__skin_mskcc_pixels.md)
- [tests/test_skin_mskcc_pixels.py](../tests/tests__test_skin_mskcc_pixels.md)
- [scripts/skin_mskcc_vote.py](../modules/scripts__skin_mskcc_vote.md)
- [scripts/skin_mskcc_vote_v2.py](../modules/scripts__skin_mskcc_vote_v2.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [cache.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/cache.json) | {} | `a4e296c48def73c1d65f5544a5bb68270e9455ec03c77c15ce68a459208046f1` |
| [cache_issue.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/cache_issue.json) | {} | `b221c12091d7beb932e38b1f5178901bf578639e770e823da0bc0fba70766614` |
| [cache_rejected_fp32.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/cache_rejected_fp32.json) | {} | `22c8979a45cae2a4f176fbaed406e469ec1456ef15ced7957b2f416e16348063` |
| [controls.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/controls.json) | {} | `58d6ad7532bd8c3011e1cd423266a3de141dd0a8faa50e278f333e1625fc1417` |
| [fusion_probe.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/fusion_probe.json) | {"scope": "Six validation people reused for architecture and epoch selection; intervals do not account for selection and are not confirmatory."} | `8b7743b1ba104f9da676dde82f5b37a1488cf13412d3eb768ad007331aca6c5e` |
| [profile.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/profile.json) | {} | `b5b2bb80d26fe49ea0861599c82eda175a19db63844d3e92c67407ef00c2dde4` |
| [summary.json](../../../../docs/benchmarks/skin_mskcc_pixels_v1/summary.json) | {} | `c559c67bea70577857eb9ad218599664a17caa9c6d34d28a35a923ce582df961` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Model | Seed | Mean DeltaE00 | Median | p95 | Parameters |
| --- | --- | --- | --- | --- | --- |
| cnn | 17 | 3.8139 | 3.1696 | 8.0428 | 1520931 |
| cnn | 29 | 4.0029 | 3.5096 | 7.9454 | 1520931 |
| cnn | 43 | 4.1147 | 3.7380 | 8.6781 | 1520931 |
| votes_mean | 17 | 3.4888 | 2.9278 | 7.1438 | 924932 |
| votes_mean | 29 | 3.5074 | 3.1112 | 7.2062 | 924932 |
| votes_mean | 43 | 3.4964 | 3.0540 | 7.2758 | 924932 |
| votes_huber3 | 17 | 3.4892 | 2.9394 | 7.1414 | 924932 |
| votes_huber3 | 29 | 3.5156 | 3.0999 | 7.1849 | 924932 |
| votes_huber3 | 43 | 3.4956 | 3.0529 | 7.2567 | 924932 |

### Таблица 2

| Architecture | Mean DeltaE00 | Median | p95 | Mean at80%, density | Mean at80%, seed disagreement |
| --- | --- | --- | --- | --- | --- |
| cnn | 3.6152 | 3.2097 | 7.1517 | 3.3885 | 3.4225 |
| votes_mean | 3.4588 | 3.0118 | 7.1105 | 3.2426 | 3.4491 |
| votes_huber3 | 3.4611 | 3.0028 | 7.1375 | 3.2453 | 3.4417 |
