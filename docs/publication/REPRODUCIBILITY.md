# Воспроизводимость и структура архива

[Главная](../../README.md) · [Отчёт](RESEARCH_REPORT_RU.md) · [Атлас](GENERATIONS.md)

Публикация не запускала новые исследования и не переобучала модели. Она построена по сохранённым результатам. «Reproduced locally» в исторических отчётах означает локальную реализацию/проверку, а не независимую репликацию внешней лабораторией.

## Три уровня воспроизведения

| Уровень | Доступен из GitHub | Что требуется |
|---|---|---|
| Читать все таблицы, методы, выводы, историю и графики | Да | Любой браузер; [catalog.json](catalog.json), [CSV](all_report_tables.csv) |
| Пересобрать публикационные графики и проверить их исходники | Да | Python, matplotlib, NumPy, Git; не нужны фотографии или GPU |
| Повторить обучение, inference и exact replay всех поколений | Код и протоколы доступны; необходимые данные/нейронные веса не размещены | Получить исходные данные по их условиям; создать кэши, обучить модели; учитывать исторические версии кода, locks и hardware/software |

Некоторые NPZ в бенчмарках — небольшие архивные числовые predictions/coefficients, не изображения. Наличие этих результатов не означает наличие всех исходных model checkpoints. JSON-only построение графика не является повторным измерением качества модели.

## Быстрая проверка публичной работы

Из корня чистого checkout:

```bash
uv sync --extra report --extra dev
uv run python scripts/build_research_archive.py
uv run python scripts/verify_research_archive.py
```

`build_research_archive.py` читает отчёты и aggregate JSON, извлекает все Markdown tables, строит figures, каталог и индекс Git-истории. Он не читает `data/private`, `data/public` или `experiments/runs`, не вызывает training/inference и не пересчитывает раскрытый TEST. Основные графики используют неокруглённые JSON; семейные обзорные графики отображают округлённые значения из первой подходящей таблицы соответствующего отчёта. Полные таблицы доступны отдельно. Межсемейные бюджеты и покрытия могут различаться.

`verify_research_archive.py` проверяет полноту 53 каталогов, соответствие таблиц отчётам, source SHA256 и числовые ключевые утверждения; валидирует локальные ссылки новых публикационных документов. Это проверка оформления и прослеживаемости, не новая accuracy validation. Matplotlib может менять рендеринг между версиями; научные значения должны оставаться теми же.

## Окружение исследований

Windows, Python 3.12, PyTorch 2.8.0 / CUDA 12.8, torchvision 0.23.0, RTX 4060 8 GB. Точные зависимости: [pyproject](../../pyproject.toml), [uv.lock](../../uv.lock), receipts соответствующей фазы. Seed обычно 17/29/43; конкретные варианты и число итераций задаются протоколом, не общим правилом.

Для полного программного окружения:

```bash
uv sync --all-extras
uv run python -m pytest -q
```

Исторический последний полный запуск завершился **379 passed, 14 warnings, 35.85 s, exit 0**. [Receipt](../benchmarks/skin_correction_transfer_v1/test_receipt.json). Некоторые воспроизводящие проверки за пределами pytest требуют локальных cached data и checkpoints; они не обязаны проходить в checkout без данных. Полный последний аудит не является обязательной частью пересборки документации и может включать реальные refits — публикационный workflow его повторно не запускает.

## Где находятся методы

| Область | Код / инструкция |
|---|---|
| Базовая библиотека и синтетический инженерный baseline | [src](../../src), [scripts](../../scripts), [tests](../../tests) |
| Real CC V1 | [Протокол](../research/public_protocol_v1.md), `scripts/prepare_public_cc.py`, `scripts/classical_public_cc.py` |
| Real CC V2 | [Подробный REPRODUCE](../benchmarks/cc_v2/REPRODUCE.md), [audit instructions](../benchmarks/cc_v2/audit_readme.md) |
| Independent MSKCC | [Полный отчёт](../benchmarks/skin_mskcc_selective_v1/report.md), `skin_mskcc_selective_*`, `skin_mskcc_oof_risk.py` |
| Последний stable correction | [Frozen protocol](../research/skin_correction_transfer_protocol_v1.md), `skin_correction_transfer.py`, `skin_correction_transfer_verify.py` |
| Все промежуточные поколения | [Атлас с report/protocol links](GENERATIONS.md), исходники с соответствующим префиксом в `scripts/` |

Используйте отдельный checkout/output для повторного эксперимента: исторические generators могут записывать результаты по фиксированным путям. Source locks фиксируют точные байты и иногда исходные локальные пути; смена ОС/checkout не должна тихо переписывать научную историю. Не запускайте recovery автоматически как гиперпараметрический поиск.

## История и публичная фильтрация

Исходный локальный Git остаётся неизменённым по истории. Публичная история сохраняет последовательность коммитов и теги, но исключает архивную полную стороннюю HTML-страницу MSKCC и два бинарных фрагмента ZIP directory. Это не результат эксперимента и не код метода; доступ/права и checksums описаны в оригинальных provenance receipts. Они остаются локально. Все остальные исторические blob bytes сохранены. Фильтрация изменяет Git commit IDs: [original history](original_history.json) и [commit map](commit_map.json) связывают их с публичной цепочкой. Удалённые из публичной версии provenance bytes нельзя независимо перепроверить из GitHub; это явное ограничение, а не успешный hash check.

Исходные хэши в текстах pretest locks не переписаны. Например, `2685bf0` — исходная synthetic freeze, а соответствующий публичный коммит определяется картой. Исходный final correction recovery отдельно сохраняет прежние batch32 границы; [описание исправления](../research/skin_correction_transfer_batch_recovery.md). Исходные references на приватные кэши/веса не являются обещанием, что файлы раздаются в репозитории.

## Что не следует делать при проверке

Не использовать раскрытый independent TEST для нового выбора архитектуры. Не смешивать обе камеры в train/test и называть это unseen-camera validation. Не объединять таблицы с разными когортами в общую кривую «роста точности». Не считать known-camera clinical benchmark проверкой обычных iPhone/Android селфи. Не копировать авторские опубликованные результаты в колонку локального воспроизведения.
