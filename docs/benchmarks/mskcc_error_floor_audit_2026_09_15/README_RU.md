# Luma: MSKCC error-floor audit

Откройте REPORT_RU.html (самодостаточный отчёт) или REPORT_RU.md.

Это аналитическая поставка, не новая модель или inference-приложение.
Все доступные reference/capture вычисления завершены на966TRAIN фото/24людях.
Разбор C residuals заблокирован отсутствием сохранённого oof_anonymized.npz:
нужно повторно прикрепить LUMA_TRAIN_OOF_C.zip. C не переобучалась.

Состав:
- REPORT_RU.html / .md: результаты, границы интерпретации, конкретный блокер;
- data/: агрегаты, координатные гистограммы, обезличенные признаки и маски;
- figures/: PNG/PDF с численными диагностическими распределениями;
- code/: новые audit-скрипты и synthetic tests; исходная модель не изменяется;
- source_provenance/: неизменённые прежние receipts и словарь данных;
- COMMANDS.md, STATE.md, requirements-audit.txt, logs/;
- audit.patch: только новые audit-код/протокол/тесты для исходного repository;
- MANIFEST_SHA256.json: SHA256 всех вложенных файлов, кроме самогоmanifest.

Сначала установите audit requirements в отдельном окружении. Для полного
повторения клонируйте исходныйrepository на проверенномcommit, добавьте только
новые scripts/error_floor и tests (не перезаписывайте пользовательские правки),
затем выполните COMMANDS.md. Для продолжения C достаточно вернуть его старый
OOF и использовать обезличенный cache; фотографии повторно скачивать не нужно.

Все oracle и исключения строк — LEAKY_DIAGNOSTIC_ONLY. Числа capture ΔE не
являются ошибкой модели, а нулевые same-site oracle повторяют общий target.
Точный noise floor и причинные доли instrument/capture/model не установлены.

Исходные фото участников, прямыеsource/patient/site IDs и веса не вложены.
Агрегаты и численныепроизводные: оригинальный MSKCC/ISIC release
DOI10.34970-962049,CC-BY; научнаяпубликация DOI10.1038/s41746-025-02245-2.
