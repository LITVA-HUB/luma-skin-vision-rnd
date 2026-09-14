# skin_shared_bias_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Общий bias нескольких снимков против их согласованности; согласие не гарантирует правильный цвет.

[Полная папка артефактов](../../../../docs/benchmarks/skin_shared_bias_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_shared_bias_v1/report.md)

SHA-256 отчёта: `be8999acf56d02ee3fc6c6a7287c2d324081e8707c30abe73f3affb4404aba68`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_shared_bias_protocol_v1.md](../../../../docs/research/skin_shared_bias_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_shared_bias_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_shared_bias.py](../modules/scripts__skin_shared_bias.md)
- [scripts/skin_shared_bias_audit.py](../modules/scripts__skin_shared_bias_audit.md)
- [scripts/skin_shared_bias_report.py](../modules/scripts__skin_shared_bias_report.md)
- [scripts/skin_shared_bias_train.py](../modules/scripts__skin_shared_bias_train.md)
- [tests/test_skin_shared_bias.py](../tests/tests__test_skin_shared_bias.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_shared_bias_v1/audit.json) | {"status": "PASS"} | `71ef85675cf288725cdc4b2bb37637138734d911e6fa1988417842e6b4358331` |
| [audit_mixed.json](../../../../docs/benchmarks/skin_shared_bias_v1/audit_mixed.json) | {"status": "PASS"} | `02468b2a864c7c4ddb475742eed8744eb9caf8a83ccee5c9381238da5a5acacf` |
| [source_lock.json](../../../../docs/benchmarks/skin_shared_bias_v1/source_lock.json) | {} | `c2ecb1c09f694c57341a6db91bb3ec1557f606be667d327cc8a7919a43e73896` |
| [summary.json](../../../../docs/benchmarks/skin_shared_bias_v1/summary.json) | {"scope": "SOURCE ONLY; three seed scores averaged, not an ensemble"} | `6e0f9c0e537f88b426a01ce30515e2e245152a2e92b5cf85a15d99847c6c031f` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Objective | Mean DeltaE00 | p95 | Three seed means |
| --- | --- | --- | --- | --- |
| mixed | individual | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| mixed | consistency | 3.4178 | 6.9574 | 3.3449, 3.4495, 3.4590 |
| mixed | shared_half | 3.5246 | 7.2980 | 3.4837, 3.5525, 3.5376 |
| mixed | shared_only | 3.7398 | 7.8739 | 3.7515, 3.7150, 3.7529 |
| mixed | historical_plain_mse | 3.4771 | 7.0733 | 3.4957, 3.5284, 3.4072 |
| mixed | historical_mixture_mse | 3.4406 | 6.9316 | 3.3783, 3.4698, 3.4736 |
| from_SLR | individual | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_SLR | consistency | 5.3455 | 10.4776 | 4.9746, 6.0530, 5.0089 |
| from_SLR | shared_half | 5.2770 | 10.3365 | 5.2087, 5.3088, 5.3136 |
| from_SLR | shared_only | 6.1655 | 11.3368 | 6.2568, 6.2203, 6.0193 |
| from_SLR | historical_plain_mse | 5.8301 | 11.2459 | 5.6802, 6.0602, 5.7499 |
| from_SLR | historical_mixture_mse | 5.0193 | 9.7825 | 4.9975, 5.2417, 4.8187 |
| from_ipod | individual | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |
| from_ipod | consistency | 5.9443 | 10.0827 | 6.3482, 5.9495, 5.5351 |
| from_ipod | shared_half | 5.8533 | 11.2498 | 5.5949, 6.3378, 5.6272 |
| from_ipod | shared_only | 6.3318 | 11.7212 | 5.9044, 7.2157, 5.8753 |
| from_ipod | historical_plain_mse | 5.5089 | 9.4018 | 5.3087, 5.6826, 5.5353 |
| from_ipod | historical_mixture_mse | 5.9635 | 11.0896 | 5.7634, 6.4146, 5.7124 |

### Таблица 2

| Protocol | Objective | Mean at 80% | Stored parameters | Peak fit+selection MiB |
| --- | --- | --- | --- | --- |
| mixed | individual | 3.4407 | 929297 | 108.36 |
| mixed | consistency | 3.4848 | 929297 | 108.36 |
| mixed | shared_half | 3.4790 | 929297 | 108.36 |
| mixed | shared_only | 3.6172 | 929297 | 108.36 |
| from_SLR | individual | 5.1087 | 929297 | 104.91 |
| from_SLR | consistency | 5.0748 | 929297 | 104.91 |
| from_SLR | shared_half | 5.3922 | 929297 | 104.91 |
| from_SLR | shared_only | 5.9525 | 929297 | 104.91 |
| from_ipod | individual | 5.7914 | 929297 | 106.33 |
| from_ipod | consistency | 5.9729 | 929297 | 106.33 |
| from_ipod | shared_half | 5.5306 | 929297 | 106.33 |
| from_ipod | shared_only | 5.8205 | 929297 | 106.33 |

### Таблица 3

| Protocol | Objective | Reference | Seed wins | Person mean difference | 95% descriptive interval |
| --- | --- | --- | --- | --- | --- |
| mixed | individual | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| mixed | consistency | mixture_mse | 3/3 | -0.0228 | [-0.0650, 0.0254] |
| mixed | shared_half | mixture_mse | 0/3 | 0.0840 | [0.0340, 0.1272] |
| mixed | shared_only | mixture_mse | 0/3 | 0.2992 | [0.2081, 0.3771] |
| from_SLR | individual | mixture_mse | 0/3 | 0.0000 | [0.0000, 0.0000] |
| from_SLR | consistency | mixture_mse | 1/3 | 0.3262 | [-0.5392, 1.2023] |
| from_SLR | shared_half | mixture_mse | 0/3 | 0.2577 | [-0.3046, 0.5762] |
| from_SLR | shared_only | mixture_mse | 0/3 | 1.1462 | [-0.6378, 3.1769] |
| from_ipod | individual | plain_mse | 0/3 | 0.4546 | [0.1312, 0.8048] |
| from_ipod | consistency | plain_mse | 1/3 | 0.4354 | [0.0151, 0.6613] |
| from_ipod | shared_half | plain_mse | 0/3 | 0.3444 | [0.1055, 0.6744] |
| from_ipod | shared_only | plain_mse | 0/3 | 0.8229 | [0.5124, 1.1705] |
