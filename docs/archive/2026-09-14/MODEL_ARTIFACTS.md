# Модели, веса и происхождение артефактов

[Архив](README.md) · [Архитектуры](ARCHITECTURES.md) · [Все модели HR](HR_ALL_MODELS.md) · [Все серии](EXPERIMENTS.md)

## Какой объект называется моделью

**Архитектура** задаёт вычисление ответа. **Вариант** меняет выбранные слои/голову/обучающий механизм. **Траектория** — отдельный ход обучения с seed, LR, ролью и бюджетом. **Checkpoint** — состояние траектории на шаге. **Selected model** — checkpoint, выбранный по зарегистрированному INNER-критерию. **Ансамбль** объединяет несколько ответов и должен быть измерен как отдельная конструкция. **Повторный audit/runtime** не добавляет новую архитектуру.

Пример HR: 7 структур ×3 head ×3 роли ×3 seeds =189 timing endpoints. Новых selected final models 126; унаследованных AS-контролей 63. При этом первичная HR-процедура потребовала 1008 успешно завершённых траекторий, а results.json содержит 207 записей с дополнительными контролями. Счётчики описывают разные уровни.

## Что опубликовано

- Исходники, конфигурации, протоколы и model cards всех сохранённых серий.
- Полные таблицы отчётов, aggregate JSON, audits, runtime receipts и failure receipts в пределах указанного охвата.
- Для HR: [source lock](evidence/chromaseed_head_range_v1/source_lock.json), [все решения INNER](evidence/chromaseed_head_range_v1/selections.json), [207 результатов](evidence/chromaseed_head_range_v1/results.json), [69 сводных role/variant групп](hr_quality_groups.json).
- Для AS: [summary](../../benchmarks/chromaseed_architecture_scale_v1/summary.json), [runtime](../../benchmarks/chromaseed_architecture_scale_v1/runtime.json), [seal](../../benchmarks/chromaseed_architecture_scale_v1/verification.json).
- Для Seg1: [checkpoint selection](evidence/data_growth_2026_09_14/facial_skin_v1/selection.json), [история](evidence/data_growth_2026_09_14/facial_skin_v1/history.json), [TEST](evidence/data_growth_2026_09_14/facial_skin_v1/test_results.json), [seal](evidence/data_growth_2026_09_14/facial_skin_v1/final_verification.json).
- Прежние небольшие NPZ-артефакты сохраняются в исторической публичной части; исходные модели новых тяжёлых серий не подменяют их.

## Что остаётся локальным

Checkpoint-массивы новых серий, bank checkpoints всех шагов, raw/prepared datasets и participant photographs не загружены в Git. Например, `.npz` в `experiments/runs/...` внутри первичного отчёта обозначает путь исходного экспериментального хранилища, а не обещание скачать этот файл из репозитория. Хэш и размер в сохранённой записи позволяют сопоставить локальный экземпляр с опытом. Публикация исходника и хэша не равна публикации весов.

Это **исчерпывающая документация сохранённого исследования**, а не готовый общий model-zoo пакет с каждой большой матрицей весов. Результаты незапущенных P3/Seg2 и их несуществующие веса не создаются. Для повторения полных моделей потребуются исходные данные и локальные артефакты зарегистрированных родителей, либо новое разрешённое воспроизведение после снятия паузы.

## Как сопоставлять доказательства

Цепочка: source lock → выбор по INNER → selected artifact hash → result record → audit → runtime → итоговая verification. Отсутствующее звено явно отмечается. Для HR цепочка заканчивается принятым аудитом качества и неудачным runtime. Сохранённый `passed: true` в одном JSON не считается заменой всей цепочки.

`numeric_bytes` — числовой payload, `parameters` — счёт параметров с оговорённым включением якоря, размер NPZ — сжатый файл, runtime memory — рабочая память процесса. Численные нормализаторы, служебные индексы, кэш FP64, Adam moments, пакеты Python и память GPU считаются отдельно. Поэтому «5 млн параметров» не означает «5 МБ приложения».

Точные копии нового импорта перечислены в [import_manifest.json](import_manifest.json). Хэши frozen-source артефактов не переписаны под публикационный commit. Предыдущая фильтрация публичной истории и сопоставление исходных коммитов сохраняются в [истории](../../publication/HISTORY.md).

## Дополнительный машиночитаемый реестр

[model_records.csv](model_records.csv) извлекает каждую подходящую запись variant/seed с параметрами, метрикой или хэшем из сохранённых JSON. Каждая строка содержит исходный файл и JSON pointer. Повторы между result/audit/parent receipts сохранены; число строк не является числом уникальных моделей. Записи иных схем остаются в полных исходных JSON. [Все 39 run-каталогов](RUNS.md) включают также preflight, precision, compact и superseded receipts.
