# skin_patch_likelihood_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Распределение патчей против среднего изображения; сложность не принесла универсальной точности.

[Полная папка артефактов](../../../../docs/benchmarks/skin_patch_likelihood_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_patch_likelihood_v1/report.md)

SHA-256 отчёта: `97de11580bf4d421b03314946bef26b4fc50be64e9334dc7341f39249a9d473f`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_patch_likelihood_protocol_v1.md](../../../../docs/research/skin_patch_likelihood_protocol_v1.md)
- [report.md](../../../../docs/benchmarks/skin_patch_likelihood_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_patch_likelihood.py](../modules/scripts__skin_patch_likelihood.md)
- [scripts/skin_patch_likelihood_train.py](../modules/scripts__skin_patch_likelihood_train.md)
- [scripts/skin_patch_likelihood_verify.py](../modules/scripts__skin_patch_likelihood_verify.md)
- [tests/test_skin_patch_likelihood.py](../tests/tests__test_skin_patch_likelihood.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_patch_likelihood_v1/audit.json) | {"status": "PASS"} | `960874bbe116b5aa902478ab206361d095c4b039190091834bef63b46a5cb4b9` |
| [source_lock.json](../../../../docs/benchmarks/skin_patch_likelihood_v1/source_lock.json) | {} | `b3999970210cf98237defab6b6bc6f727527333ce9ffaee0192a1e65d7c39804` |
| [summary.json](../../../../docs/benchmarks/skin_patch_likelihood_v1/summary.json) | {"scope": "SOURCE exploratory only, no new independent test"} | `10c26e33a4fa72301cf1def9871348db1f3ea7ed17dcf20efd2d72ed3745526c` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol / family | Initializations | Mean | Median | p95 | Common 80% |
| --- | --- | --- | --- | --- | --- |
| mixed/bag_d1_k1 | 1 | 4.9960 | 4.4199 | 9.9673 | 4.9910 |
| mixed/bag_d1_k3 | 3 | 4.2103 | 3.8964 | 8.6981 | 4.1913 |
| mixed/bag_d2_k1 | 1 | 4.7854 | 4.2638 | 9.9554 | 4.7591 |
| mixed/bag_d2_k3 | 3 | 4.0554 | 3.4875 | 8.6304 | 3.9947 |
| mixed/mean_d1_k1 | 1 | 5.1126 | 4.4212 | 10.7429 | 5.1404 |
| mixed/mean_d1_k3 | 3 | 4.1832 | 3.6952 | 8.4315 | 4.1538 |
| mixed/mean_d2_k1 | 1 | 4.8533 | 4.1516 | 10.4523 | 4.8495 |
| mixed/mean_d2_k3 | 3 | 4.0120 | 3.4598 | 8.6074 | 3.9506 |
| from_SLR/bag_d1_k1 | 1 | 5.9690 | 5.4105 | 10.7989 | 5.6272 |
| from_SLR/bag_d1_k3 | 3 | 6.4100 | 6.0247 | 12.1073 | 6.1340 |
| from_SLR/bag_d2_k1 | 1 | 5.7480 | 5.4769 | 10.4481 | 5.4046 |
| from_SLR/bag_d2_k3 | 3 | 6.6379 | 6.0045 | 14.6197 | 5.5029 |
| from_SLR/mean_d1_k1 | 1 | 5.8589 | 5.2602 | 10.6963 | 5.5374 |
| from_SLR/mean_d1_k3 | 3 | 6.6616 | 6.0116 | 12.5591 | 6.1750 |
| from_SLR/mean_d2_k1 | 1 | 5.6521 | 5.4606 | 10.2894 | 5.3286 |
| from_SLR/mean_d2_k3 | 3 | 7.3072 | 5.7168 | 15.6043 | 5.8339 |
| from_ipod/bag_d1_k1 | 1 | 8.0489 | 6.9837 | 14.9378 | 7.2322 |
| from_ipod/bag_d1_k3 | 3 | 7.8077 | 6.8819 | 14.6904 | 6.7560 |
| from_ipod/bag_d2_k1 | 1 | 8.4358 | 7.2554 | 15.9115 | 7.7306 |
| from_ipod/bag_d2_k3 | 3 | 6.6877 | 5.7653 | 13.4961 | 6.2920 |
| from_ipod/mean_d1_k1 | 1 | 9.7372 | 8.7850 | 17.3444 | 9.0319 |
| from_ipod/mean_d1_k3 | 3 | 8.5645 | 8.0872 | 16.0283 | 7.5717 |
| from_ipod/mean_d2_k1 | 1 | 9.7187 | 8.6462 | 17.9220 | 9.2626 |
| from_ipod/mean_d2_k3 | 3 | 8.3810 | 7.7791 | 15.8246 | 7.4360 |

### Таблица 2

| Protocol | Candidate vs control | Mean difference | Patient 95% interval |
| --- | --- | --- | --- |
| mixed | bag_d1_k1 vs mean_d1_k1 | -0.1165 | [-0.2902, 0.0570] |
| mixed | bag_d1_k3 vs mean_d1_k3 | 0.0271 | [-0.0779, 0.1561] |
| mixed | bag_d2_k1 vs mean_d2_k1 | -0.0678 | [-0.2082, 0.0725] |
| mixed | bag_d2_k3 vs mean_d2_k3 | 0.0434 | [-0.0630, 0.1528] |
| mixed | bag_d1_k3 vs bag_d1_k1 | -0.7858 | [-1.2638, -0.3339] |
| mixed | bag_d2_k3 vs bag_d2_k1 | -0.7300 | [-1.2955, -0.2848] |
| mixed | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [0.0000, 0.0000] |
| mixed | bag_d1_k3 vs bag_d1_k3 collapsed inference | 0.0098 | [-0.0382, 0.0599] |
| mixed | bag_d2_k1 vs bag_d2_k1 collapsed inference | 0.0000 | [0.0000, 0.0000] |
| mixed | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0030 | [-0.0391, 0.0331] |
| from_SLR | bag_d1_k1 vs mean_d1_k1 | 0.1101 | [0.0210, 0.1797] |
| from_SLR | bag_d1_k3 vs mean_d1_k3 | -0.2516 | [-0.5016, -0.0455] |
| from_SLR | bag_d2_k1 vs mean_d2_k1 | 0.0959 | [0.0095, 0.1628] |
| from_SLR | bag_d2_k3 vs mean_d2_k3 | -0.6692 | [-1.1065, -0.4003] |
| from_SLR | bag_d1_k3 vs bag_d1_k1 | 0.4410 | [-0.3774, 1.1282] |
| from_SLR | bag_d2_k3 vs bag_d2_k1 | 0.8900 | [-0.0026, 1.8394] |
| from_SLR | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [-0.0000, 0.0000] |
| from_SLR | bag_d1_k3 vs bag_d1_k3 collapsed inference | -0.1875 | [-0.4183, -0.0575] |
| from_SLR | bag_d2_k1 vs bag_d2_k1 collapsed inference | -0.0000 | [-0.0000, 0.0000] |
| from_SLR | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0716 | [-0.2060, 0.1193] |
| from_ipod | bag_d1_k1 vs mean_d1_k1 | -1.6883 | [-2.1146, -1.1049] |
| from_ipod | bag_d1_k3 vs mean_d1_k3 | -0.7568 | [-0.8682, -0.5481] |
| from_ipod | bag_d2_k1 vs mean_d2_k1 | -1.2830 | [-1.5004, -1.1029] |
| from_ipod | bag_d2_k3 vs mean_d2_k3 | -1.6933 | [-2.3370, -0.7110] |
| from_ipod | bag_d1_k3 vs bag_d1_k1 | -0.2412 | [-0.5414, 0.1959] |
| from_ipod | bag_d2_k3 vs bag_d2_k1 | -1.7481 | [-2.3769, -0.9551] |
| from_ipod | bag_d1_k1 vs bag_d1_k1 collapsed inference | 0.0000 | [-0.0000, 0.0000] |
| from_ipod | bag_d1_k3 vs bag_d1_k3 collapsed inference | 0.0775 | [0.0495, 0.1152] |
| from_ipod | bag_d2_k1 vs bag_d2_k1 collapsed inference | -0.0000 | [-0.0000, 0.0000] |
| from_ipod | bag_d2_k3 vs bag_d2_k3 collapsed inference | -0.0131 | [-0.0336, 0.0228] |

### Таблица 3

| Protocol / family | Strengths | Full bag/model mean | Collapsed mean | Posterior 80% |
| --- | --- | --- | --- | --- |
| mixed/bag_d1_k1 | [1] | 4.9960 | 4.9960 | 5.2028 |
| mixed/bag_d1_k3 | [1, 1, 1] | 4.2103 | 4.2005 | 4.1098 |
| mixed/bag_d2_k1 | [1] | 4.7854 | 4.7854 | 4.8944 |
| mixed/bag_d2_k3 | [1, 1, 1] | 4.0554 | 4.0584 | 3.9921 |
| mixed/mean_d1_k1 | [1] | 5.1126 | 5.1126 | 5.3385 |
| mixed/mean_d1_k3 | [1, 1, 1] | 4.1832 | 4.1832 | 4.2454 |
| mixed/mean_d2_k1 | [1] | 4.8533 | 4.8533 | 4.9936 |
| mixed/mean_d2_k3 | [1, 1, 1] | 4.0120 | 4.0120 | 3.9645 |
| from_SLR/bag_d1_k1 | [1] | 5.9690 | 5.9690 | 5.9566 |
| from_SLR/bag_d1_k3 | [1, 1, 1] | 6.4100 | 6.5974 | 5.9600 |
| from_SLR/bag_d2_k1 | [1] | 5.7480 | 5.7480 | 5.8336 |
| from_SLR/bag_d2_k3 | [1, 1, 1] | 6.6379 | 6.7095 | 6.7746 |
| from_SLR/mean_d1_k1 | [1] | 5.8589 | 5.8589 | 5.8862 |
| from_SLR/mean_d1_k3 | [1, 1, 1] | 6.6616 | 6.6616 | 6.5225 |
| from_SLR/mean_d2_k1 | [1] | 5.6521 | 5.6521 | 5.7663 |
| from_SLR/mean_d2_k3 | [1, 1, 1] | 7.3072 | 7.3072 | 7.5242 |
| from_ipod/bag_d1_k1 | [1] | 8.0489 | 8.0489 | 8.7506 |
| from_ipod/bag_d1_k3 | [1, 1, 1] | 7.8077 | 7.7301 | 8.3281 |
| from_ipod/bag_d2_k1 | [1] | 8.4358 | 8.4358 | 9.0992 |
| from_ipod/bag_d2_k3 | [1, 1, 1] | 6.6877 | 6.7007 | 6.6324 |
| from_ipod/mean_d1_k1 | [1] | 9.7372 | 9.7372 | 10.5093 |
| from_ipod/mean_d1_k3 | [1, 1, 1] | 8.5645 | 8.5645 | 9.3065 |
| from_ipod/mean_d2_k1 | [1] | 9.7187 | 9.7187 | 10.4317 |
| from_ipod/mean_d2_k3 | [1, 1, 1] | 8.3810 | 8.3810 | 8.8616 |
