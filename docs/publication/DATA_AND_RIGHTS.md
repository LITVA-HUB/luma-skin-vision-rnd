# Данные, права и состав публичного архива

Это сводка зафиксированного исследовательского provenance на 11 сентября 2026. Она не изменяет исходные лицензии и не даёт новой blanket legal clearance. Оригинальные условия датасета имеют приоритет над метками сторонних mirrors/Hugging Face.

| Источник | Зафиксированные исходные условия | Как использован |
|---|---|---|
| SimpleCube++ | CC BY 4.0 | Основное реальное CC обучение/валидация; сохранять атрибуцию авторов |
| INTEL-TAU | CC BY-SA 4.0 | Evaluation only; модель не обучалась на его целевых камерах; mirror provenance ограничен |
| Beyond RGB | CC BY 4.0 | Селективно полученное Samsung/Oppo подмножество, component evaluation |
| MSKCC Skin Tone Labeling | CC-BY в original release/per-image metadata; версия не приписана | Реальные приборно размеченные изображения для direct-skin TRAIN/VAL/CAL/TEST; атрибуция Memorial Sloan Kettering Cancer Center, DOI 10.34970/962049 |
| He 2021 RGB/XYZ workbook | CC BY 4.0 | Региональный XYZ pilot, не полные изображения и не fabricated ΔE00 |
| UMINHO-HSFD | CC BY 4.0 | TRAIN reflectance cubes, spectral mechanism checks |
| ISSA v4 | CC BY 4.0 | Spectral controls с сохранёнными role/source restrictions |
| Zhang et al., Dryad iPhone auxiliary | CC0 | 30 объектных PNG, не достаточный absolute skin-color benchmark; не primary model training |
| NUS-8 / Gehler-Shi и другие кандидаты | Статус по каждому original source в inventory; не автоматически cleared | Не объявлены production-training основой без нужных прав |
| Rendered WB / FaceOLAT и иные restricted candidates | Research/non-commercial ограничения явно учитываются | Prior art/кандидаты не означают принятия данных или weights |

Полные оригинальные ссылки, размеры, camera/RAW/linear/sRGB/GT/CCM и категории:

- [Color constancy inventory](../data/public_dataset_inventory.md)
- [Прямой skin inventory](../data/public_skin_color_inventory.md)
- [Телефонные источники](../data/smartphone_benchmark_plan.md)
- [UMINHO rights](../data/uminho_hsfd_verified_inventory.md)
- [MSKCC provenance](../ip/skin_mskcc_provenance.md)
- [Installed libraries и лицензии](../ip/licensing_inventory.md)
- [DINOv2 teacher](../ip/dinov2_teacher_adoption.md)

## Разделение артефактов

В публичной работе размещены исследовательские исходники, существующие permissively licensed code snapshots с их notices, агрегированные результаты, небольшие числовые prediction/fit архивы, протоколы, hashes и графики. Не размещены исходные фотографии участников, приватные датасеты/кэши, нейросетевые checkpoints, токены доступа или credentials. Папки `data/private`, `data/public`, `data/processed`, `experiments/runs`, `.venv` и `.env` не входят в публикацию.

Полная сторонняя HTML-страница и два бинарных ZIP directory fragments исключены из публичной Git-истории; оригинальные sources/checksums сохраняются в provenance. Исследовательские результаты не удалялись. Копии пользовательской постановки/аудита в `docs/context` — provided context, а не независимо проверенные результаты этой реализации и не новые инструкции читателю.

Repository-wide license для собственного кода и документации этой публикацией **не установлена**. Это публично доступный исследовательский архив; не следует объявлять его OSI-open-source или считать публичность разрешением на любое коммерческое повторное использование. Существующие third-party licenses сохраняют действие в своей области. Для общей лицензии собственного проекта требуется отдельное решение правообладателя; оно не подменено выбором агента.

Данные, код, teacher weights, distilled weights и derived numerical buffers имеют отдельные права. Например, Apache-2.0 DINOv2 не выдаёт права на исходный training corpus; CC BY-SA 4.0 CIE данные в спектральных derivations требуют своей атрибуции/условий. Никакие checkpoint с такими buffers не выданы как неограниченный proprietary production artifact. Конкретные цепочки — в [IP provenance](../ip).

Исследовательское использование лицензированного набора не подтверждает clinical suitability, device independence или готовность косметических рекомендаций. Публикация не раскрывает новые изображения и не меняет условия доступа к исходным датасетам.
