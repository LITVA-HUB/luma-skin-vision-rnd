# skin_capture_v1

[Все серии](../EXPERIMENTS.md) · [Архитектуры](../ARCHITECTURES.md) · [Текущий статус](../STOP_STATUS.md)

**Статус документации:** Есть исходный отчёт.

Скрытые условия съёмки и смесь четырёх гипотез. Сильный компактный source baseline: 929 297 параметров.

[Полная папка артефактов](../../../../docs/benchmarks/skin_capture_v1)

[Полный исходный отчёт: методика, все результаты, ограничения](../../../../docs/benchmarks/skin_capture_v1/report.md)

SHA-256 отчёта: `28474238a39fb08a5efaae8f613cb7a1af90316a29a22c52db2045e58b02b199`.

## Архитектура, протокол и решения

Все связанные документы сохранены целиком. Model card задаёт контракт; протокол определяет сравниваемые варианты, сплиты, критерий выбора и бюджет; decision описывает наблюдения и ограничения.

- [skin_capture_next_decision.md](../../../../docs/research/skin_capture_next_decision.md)
- [skin_capture_protocol_v1.md](../../../../docs/research/skin_capture_protocol_v1.md)
- [skin_capture_support_next_decision.md](../../../../docs/research/skin_capture_support_next_decision.md)
- [skin_capture_support_protocol_v1.md](../../../../docs/research/skin_capture_support_protocol_v1.md)
- [skin_capture_and_spectra_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_capture_and_spectra_evidence_2026_09_11.md)
- [skin_capture_support_evidence_2026_09_11.md](../../../../docs/skolkovo/skin_capture_support_evidence_2026_09_11.md)
- [report.md](../../../../docs/benchmarks/skin_capture_v1/report.md)

## Реализация и все связанные тесты

Ссылки ведут к подробному разбору модулей с размерностями, конфигурациями, конструкторами, вычислениями и полными тестовыми условиями. Общие зависимости перечислены в каждом модуле; [глобальный индекс](../SOURCE_INDEX.md) охватывает также реализации с историческими именами.

- [scripts/skin_capture_audit.py](../modules/scripts__skin_capture_audit.md)
- [scripts/skin_capture_model.py](../modules/scripts__skin_capture_model.md)
- [scripts/skin_capture_report.py](../modules/scripts__skin_capture_report.md)
- [scripts/skin_capture_support.py](../modules/scripts__skin_capture_support.md)
- [scripts/skin_capture_support_audit.py](../modules/scripts__skin_capture_support_audit.md)
- [scripts/skin_capture_support_report.py](../modules/scripts__skin_capture_support_report.md)
- [scripts/skin_capture_support_train.py](../modules/scripts__skin_capture_support_train.md)
- [scripts/skin_capture_support_verify.py](../modules/scripts__skin_capture_support_verify.md)
- [scripts/skin_capture_train.py](../modules/scripts__skin_capture_train.md)
- [tests/test_skin_capture_loss.py](../tests/tests__test_skin_capture_loss.md)
- [tests/test_skin_capture_support.py](../tests/tests__test_skin_capture_support.md)

## Сохранённые проверки

Флаги ниже дословно взяты из JSON. `passed` у аудита не заменяет результат проверки гипотезы; например, корректно зафиксированная неэквивалентность может пройти проверку архива.

| Артефакт | Зафиксированные поля | SHA-256 |
|---|---|---|
| [audit.json](../../../../docs/benchmarks/skin_capture_v1/audit.json) | {"status": "PASS"} | `f06c3722d8d918ee2bde893a474f00aa9ec1a2903de063f33909e079f00e86ff` |
| [numerical_loss_audit.json](../../../../docs/benchmarks/skin_capture_v1/numerical_loss_audit.json) | {"scope": "Numerical formula check, not synthetic scientific skin accuracy"} | `66c401cdeded632feb1d07644211749c187ed3d87fcd8da2af1cbb5ffc525faa` |
| [source_lock.json](../../../../docs/benchmarks/skin_capture_v1/source_lock.json) | {} | `46dac9e639450ec46663b5206f35ba8aa49d35d0811403342f74c314038765b0` |
| [summary.json](../../../../docs/benchmarks/skin_capture_v1/summary.json) | {"scope": "Source-only exploratory factorial, not a new independent test"} | `e7c5f3ee098cad6bbfe1cc474e444a4e620dcf458655f242be2fb4dc7737f4d5` |

## Все таблицы исходного отчёта

Значения перенесены без округления или пересчёта. Повторённая в двух отчётах строка не является двумя независимыми опытами. Единицы, выборка и смысл столбцов определены в полном отчёте выше.

### Таблица 1

| Protocol | Architecture | Objective | Mean DeltaE00: seed17 /29 /43 | Mean over seeds | Mean p95 | Mode accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| mixed | plain | mse | 3.4957 / 3.5284 / 3.4072 | 3.4771 | 7.0733 | 24.6% |
| mixed | uniform | mse | 3.5242 / 3.5189 / 3.4576 | 3.5002 | 7.2727 | 65.0% |
| mixed | mixture | mse | 3.3783 / 3.4698 / 3.4736 | 3.4406 | 6.9316 | 60.5% |
| mixed | plain | de2 | 3.4835 / 3.5571 / 3.4391 | 3.4932 | 7.4447 | 25.4% |
| mixed | uniform | de2 | 3.5204 / 3.5608 / 3.4493 | 3.5102 | 7.1286 | 61.4% |
| mixed | mixture | de2 | 3.4442 / 3.5692 / 3.4394 | 3.4843 | 7.2796 | 43.3% |
| from_SLR | plain | mse | 5.6802 / 6.0602 / 5.7499 | 5.8301 | 11.2459 | 27.0% |
| from_SLR | uniform | mse | 5.8079 / 6.0563 / 5.3514 | 5.7385 | 10.7133 | 40.2% |
| from_SLR | mixture | mse | 4.9975 / 5.2417 / 4.8187 | 5.0193 | 9.7825 | 44.9% |
| from_SLR | plain | de2 | 5.2945 / 5.9691 / 5.8270 | 5.6969 | 10.5457 | 23.2% |
| from_SLR | uniform | de2 | 5.2978 / 5.7979 / 5.7131 | 5.6029 | 10.4527 | 37.6% |
| from_SLR | mixture | de2 | 5.2773 / 5.7605 / 5.3011 | 5.4463 | 9.9017 | 45.2% |
| from_ipod | plain | mse | 5.3087 / 5.6826 / 5.5353 | 5.5089 | 9.4018 | 27.8% |
| from_ipod | uniform | mse | 5.8259 / 7.2178 / 7.2009 | 6.7482 | 11.4012 | 47.0% |
| from_ipod | mixture | mse | 5.7634 / 6.4146 / 5.7124 | 5.9635 | 11.0896 | 42.7% |
| from_ipod | plain | de2 | 7.1997 / 6.4725 / 6.7385 | 6.8036 | 11.5086 | 29.8% |
| from_ipod | uniform | de2 | 7.4360 / 6.5850 / 6.8899 | 6.9703 | 11.8014 | 41.7% |
| from_ipod | mixture | de2 | 7.8231 / 7.8796 / 6.8225 | 7.5084 | 12.3615 | 42.9% |

### Таблица 2

| Protocol | Objective | Mixture minus uniform | Mixture minus plain | Seedwise mixture minus uniform |
| --- | --- | --- | --- | --- |
| mixed | mse | -0.0596 | -0.0365 | -0.1458 / -0.0491 / +0.0161 |
| mixed | de2 | -0.0259 | -0.0090 | -0.0762 / +0.0084 / -0.0099 |
| from_SLR | mse | -0.7192 | -0.8108 | -0.8103 / -0.8146 / -0.5326 |
| from_SLR | de2 | -0.1566 | -0.2506 | -0.0205 / -0.0374 / -0.4120 |
| from_ipod | mse | -0.7847 | +0.4546 | -0.0625 / -0.8032 / -1.4885 |
| from_ipod | de2 | +0.5381 | +0.7049 | +0.3871 / +1.2947 / -0.0674 |
