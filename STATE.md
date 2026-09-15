# Luma — состояние на 15 сентября 2026

Ветка: `research/luma-audit-recovery-2026-09-15`.
Исходный commit проекта: `9cad271aad159d73083259967b4107b2ce828eb1`.
Дообучение и публикация этой аналитической работы в отдельной ветке разрешены владельцем. Merge и платные ресурсы не использовались.

## Текущий результат

**C_REBASE_V1 выполнен. Аудит завершён. Диагностический вывод MIXED.**

- TRAIN: 966 снимков /24 человека /248 sites, шесть person-disjoint folds по4 человека.
- До OOF в GitHub commit `b744a5130d144b350885db4f92ce6801c614fcac` сохранены config, mapping, preprocessing, versions, hashes и diagnostic quantiles.
- Обучено шесть MLP36→64→3 с нуля по одному фиксированному рецепту,100эпох. Нового model search и error predictor нет.
- OOF ΔE00 mean4.618814 /median3.896521 /p9510.190053; >5=34.6791%, >10=5.5901%.
- Fold0 перемещён в quarantine и воспроизведён с нуля в новом процессе. Predictions/веса/нормализация/RNG совпадают побитово, maximum Lab difference0. Итог: `BITWISE_REPRODUCED_FOLD_0`.
- Готовы top50, patient/site/device/mode/anatomical strata, correlations с patient-bootstrap, заранее зафиксированные counterfactuals и leaky oracles.
- Аудит повторён только по опубликованным обезличенным rows: все численные результаты совпали.

Полные результаты и ограничения: [REPORT_RU.md](experiments/c_rebase_v1/REPORT_RU.md).
Репродукция: [COMMANDS.md](experiments/c_rebase_v1/COMMANDS.md).
OOF SHA256: `f0b492bc17087c86359f9c05527fbf01863651c9d6bd3b444cc8bc52cdf0affa`.

## Что установлено

Q4 нестабильности capture содержит37/49(75.5%) ошибок выше общегоp95. Один patient P12 или palms/soles вместе —126/966снимков,39/49ошибок хвоста. При этом даже стабильный срез reference/capture/clipping оставляет median3.699,p958.291 при51.14%coverage.

Within-site prediction variation составляет23.03% суммы coordinate MSE; site-mean bias76.97%. Это алгебраическое разложение фиксированного estimator, не причинные доли. ТолькоIID-assumption reference-noise proxy равен5.20% coordinateMSE, допущения не подтверждены. Истинный физический floor не идентифицируется без независимых повторных reference acquisitions exactsite, которых нет в release.

## Историческое и неизменное

Исторический C: **HISTORICAL_NOT_REPRODUCIBLE** из-за потерянных folds и полного training config. Пользователь разрешил новый фиксированный протокол. Старые агрегаты не использовались для настройки или controlled comparison.

Production C / face inference не менялись. Текущая поставка аналитическая; trainedfoldweights не назначены production моделью. Историческая source-validation, calibration и test не использовались и не оценивались. Независимая selfie/instrument приёмка не выполнена, целевая точность2/5не достигнута. Фоновых тренировок нет.

## Сохранённые материалы

- [Новые checkpoints, логи, OOF](experiments/c_rebase_v1/run/)
- [Повтор fold0](experiments/c_rebase_v1/reproduction_fresh/)
- [Статус повторения](experiments/c_rebase_v1/reproduction_result.json)
- [Зафиксированный протокол](experiments/c_rebase_v1/PROTOCOL_RU.md)
- [Первый reference/capture audit](docs/benchmarks/mskcc_error_floor_audit_2026_09_15/REPORT_RU.md)
- [История восстановления потерянного C](docs/benchmarks/c_oof_recovery_2026_09_15/REPORT_RU.md)

Старые отчёты и их STATE сохранены как исторические снимки. Первоначальный `run/oof_metrics.json` содержит PENDING повторения на момент окончания обучения; более поздний `reproduction_result.json` подтверждает успешный repeat.

GitHub содержит код, config/mapping/manifest, веса, логи, обезличенные OOF/diagnostics и отчёт. Фотографии, прямые IDs и приватный полный mapping не публикуются. Приватный список24людей сохранён отдельно владельцу; его SHA256 включён в публичный mapping. Oracle/привилегированные срезы всегда LEAKY_DIAGNOSTIC_ONLY.
