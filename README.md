<div align="center">

# Luma Skin Vision · Research Archive

**Компактные модели цвета кожи, цветовая нормализация и оценка надёжности**

Реальные публичные данные · воспроизводимые контроли · отрицательные результаты

[Исследовательский отчёт](docs/publication/RESEARCH_REPORT_RU.md) · [Все поколения](docs/publication/GENERATIONS.md) · [Галерея](docs/publication/GALLERY.md) · [Воспроизведение](docs/publication/REPRODUCIBILITY.md)

![Путь исследования](docs/publication/figures/research_journey.png)

</div>

> **Исследования остановлены 11 сентября 2026 по решению автора проекта.** Это завершённый архив. Универсальная точная модель цвета кожи для обычных смартфонов пока не получена. Неудачные гипотезы и сильные обычные контроли сохранены.

## Что удалось измерить

На **реальных фотографиях кожи с приборным эталоном** основной компактный ансамбль получил **4,457 ΔE00** на независимом тесте из **400 изображений / 10 новых людей**. При принятии 80% изображений ошибка составила **4,159 ΔE00**. Это измерение цвета кожи, а не угловая ошибка освещения.

| Независимый MSKCC тест, известные камеры | Полное покрытие · mean ΔE00 ↓ | 80% покрытия · mean ΔE00 ↓ |
|---|---:|---:|
| C+ — тот же цветовой ансамбль, стандартный error head | 4,457 | 4,333 |
| Proposed — дополнительные признаки несогласия | 4,457 | 4,159 |
| Обычная fusion из шести моделей — более крупный контроль | **4,300** | **4,145** |

Разница Proposed−C+ при 80%: −0,174 ΔE00; 95% интервал по людям **[−0,568; +0,165]** пересекает ноль. **Убедительная победа специального механизма не доказана; сильнейший обычный полный контроль не побеждён.** Эталон — среднее опубликованных SkinColorCatch Lab измерений участка кожи в исходной D65/10° конвенции. Это клинические/дермоскопические снимки Canon SLR и iPod Touch, не обычные селфи.

![Независимый тест цвета кожи](docs/publication/figures/independent_skin.png)

[Полный независимый отчёт](docs/benchmarks/skin_mskcc_selective_v1/report.md) · [Исходный JSON](docs/benchmarks/skin_mskcc_selective_v1/test_results.json) · [Профиль вычислений](docs/benchmarks/skin_mskcc_selective_v1/profile.json)

## Все поколения — вместе с неудачами

Архив содержит **53 каталога бенчмарков**, исходники, протоколы, source locks, проверки, историю обучения, JSON/CSV, графики и журнал решений. Каталоги включают переоценки и служебные материалы: это не 53 независимых архитектуры. В сводном CSV сохранены **1 532 строки опубликованных таблиц**; разные протоколы не объединены в искусственный рейтинг.

| Линия исследования | Что проверяли | Итог |
|---|---|---|
| [Синтетический прототип](docs/benchmarks/benchmark_results.md) | Pipeline, A0/A1/A2/C/C+/Proposed, ONNX | Proposed проиграл A2; исходное состояние сохранено |
| [Color constancy V1–V2](docs/benchmarks/cc_v2_report.md) | Классические якоря, residual, selective risk | V2: 3,805° против 5,558° у matched C+ при 80% на внешней выборке; не skin accuracy |
| [V3–V7](docs/publication/GENERATIONS.md) | Цветовые базисы, графы, критик, повторные проходы, teacher, виртуальные сенсоры | Сложная архитектура и скрещивание компонентов не обеспечили общего прироста |
| [Samsung/Oppo](docs/benchmarks/phone_v1_alias_report.md) | Source-only модели на Beyond RGB | 79/88 пригодных эталонов; старая V2 сильнее новых критиков; не JPEG/HEIC skin benchmark |
| [Прямой цвет кожи](docs/benchmarks/skin_mskcc_selective_v1/report.md) | JPEG→Lab, patch votes, CNN, fusion, OOF error head | Независимые приборные измерения получены; продуктовая цель не достигнута |
| [Механизмы и альтернативы](docs/publication/GENERATIONS.md) | Спектры, вероятностные поля, обратная задача, relational и local-reference модели | Отдельные source gains; нет доказанной универсальной модели |
| [Последняя проверка](docs/benchmarks/skin_correction_transfer_v1/report.md) | Корректор +188 035 параметров и перенос между камерами | Внутренний выигрыш 7,40% не перенёсся: все три head ухудшили оба направления |

![Внутренний результат и отрицательный перенос](docs/publication/figures/correction_falsifier.png)

## Навигация

1. **[Основной отчёт](docs/publication/RESEARCH_REPORT_RU.md)** — постановка, методы, данные, результаты, ограничения и выводы.
2. **[Атлас поколений](docs/publication/GENERATIONS.md)** — каждый каталог, гипотеза, наблюдение и доказательства.
3. **[Галерея](docs/publication/GALLERY.md)** — сводные PNG/SVG и исходные risk–coverage, абляции и диагностики.
4. **[История](docs/publication/HISTORY.md)** — фиксация протоколов, исправлений и результатов.
5. **[Воспроизводимость](docs/publication/REPRODUCIBILITY.md)** — команды, окружение и доступные артефакты.
6. **[Данные и права](docs/publication/DATA_AND_RIGHTS.md)** — первичные лицензии и ограничения.
7. **[Досье Luma / Сколково](docs/publication/SKOLKOVO_EVIDENCE.md)** — реализовано / измерено / ещё не подтверждено.

## Размер и вычисления

| Измеренный объект | Параметры | Ресурсы и предел утверждения |
|---|---:|---|
| Основной независимый patch ensemble | 2 774 796 | 11,11 MB весов; **1,470 ms** batch 1 на RTX 4060, только модель по готовым признакам; inference allocation 20,04 MiB |
| Обычный шестимодельный fusion | 7 337 589 | Более крупный контроль; его задержка не равна задержке patch ensemble |
| Финальный core + correction head | 1 117 332 | 4,48 MB checkpoint; inner-core training allocation 104,42–109,50 MiB, head 73,81–76,78 MiB; новая latency не измерялась |

JPEG decoding, подготовка признаков, локализация и error head не включены в 1,470 ms. Это не полная задержка приложения. Независимая модель кожи не экспортировалась в ONNX; исторические ONNX результаты относятся к другим моделям.

## Что это означает для Luma

Получены исследовательский код и измерительная база для будущих photometric/reliability и skin-color подсистем. Подтверждена возможность компактной регрессии цвета кожи по реальным приборным меткам. Найдены сильные обычные контроли, проблемы переноса и ограничения отказа по ожидаемой ошибке.

**Не подтверждены:** точность лица на обычных iPhone/Android, независимость от произвольной камеры, надёжный подбор косметического оттенка, клиническая пригодность или научная/патентная новизна. Цель проекта median ≤2 и p95 ≤5 ΔE00 при ≥80% покрытия на новых обычных телефонах остаётся недостигнутой; это исследовательская цель, не отраслевой стандарт.

## Пересобрать оформление без данных и обучения

```bash
uv sync --extra report --extra dev
uv run python scripts/build_research_archive.py
uv run python scripts/verify_research_archive.py
```

Фотографии участников, приватные кэши и нейросетевые checkpoints не размещены. Для повторного обучения нужны отдельное получение данных по исходным условиям и восстановление окружения. [Подробности](docs/publication/REPRODUCIBILITY.md).

**Public research / source-available archive.** Публичность репозитория сама по себе не выдаёт общую лицензию на повторное использование. Лицензии данных, стороннего кода и весов учитываются отдельно: [права](docs/publication/DATA_AND_RIGHTS.md).

<details>
<summary>English abstract</summary>

This archive documents compact color normalization, selective reliability estimation and instrument-referenced skin-color regression. The independent MSKCC known-camera test contains 400 photographs from 10 new people. The primary 2.775M-parameter patch ensemble achieves mean CIEDE2000 4.457, and 4.159 at 80% coverage. Its selective improvement over matched C+ is not statistically convincing; an ordinary larger fusion achieves 4.300 and 4.145. Exploratory experiments cover canonical color frames, recurrent critics, teacher transfer, graphs, conditional densities, spectral decoders, local references and person-excluded correction. No universal camera-independent skin model is established. Negative results, source reuse, license restrictions and historical protocol repairs are preserved. Research stopped at the owner's request; this is an evidence archive, not a claim of production readiness or peer-reviewed novelty.

</details>
