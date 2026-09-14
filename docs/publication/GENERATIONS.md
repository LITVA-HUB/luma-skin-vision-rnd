> Историческая публикация этапа 10–11 сентября. [Полный обновлённый архив 14 сентября](../archive/2026-09-14/README.md).

# Атлас всех поколений и экспериментов

В архиве **53 каталога бенчмарков**. Это не число независимых архитектур или обучений: здесь также общие контроли, переоценки и служебные папки.

Порядок — первое появление каталога в исходной Git-истории. Исторические слова «next», «unopened» и «active» описывают момент написания; исследования остановлены 11 сентября 2026 по запросу автора проекта.

[Главная](../../README.md) · [Исследовательская работа](RESEARCH_REPORT_RU.md) · [Галерея](GALLERY.md) · [Все таблицы JSON](catalog.json) · [История](HISTORY.md)

## Синтетический нулевой этап

[Отчёт](../benchmarks/benchmark_results.md), [baseline](../benchmarks/baseline_report.md), [абляции](../benchmarks/ablation_report.md). Proposed проиграл классическому A2. 48 тестов / 34 smoke-команды относятся к тому этапу; синтетическая точность не подтверждает реальную.

| № | Семейство / исходный отчёт | Наблюдение |
|---:|---|---|
| 01 | [public_runs](../../docs/benchmarks/public_benchmark_report.md) | Первый реальный SimpleCube++ benchmark: классические и компактные нейросетевые контроли. |
| 02 | [source_snapshots](../../docs/benchmarks/source_snapshots) | Архив исходников при исправлении формата. Это служебная доказательная папка, не поколение модели. |
| 03 | [cc_v2](../../docs/benchmarks/cc_v2_report.md) | Якорный остаточный метод: выигрыш у matched C+ на внешних камерах, регрессия на части доменов. |
| 04 | [cc_v3](../../docs/benchmarks/cc_v3_report.md) | Полный цветовой базис и граф: проигрыш обычному RGB при одинаковой ёмкости. |
| 05 | [cc_v4](../../docs/benchmarks/cc_v4_report.md) | Оценка исправленного цвета и повторное уточнение: механизм transport не выиграл source screen. |
| 06 | [cc_v5](../../docs/benchmarks/cc_v5_report.md) | Парное обучение критика и равное число запросов: устойчивого преимущества повторных проходов нет. |
| 07 | [phone_v1](../../docs/benchmarks/phone_v1_report.md) | Первый фиксированный Samsung/Oppo тест; строгий HDF5 loader ограничивал достижимое покрытие. |
| 08 | [phone_v1_alias](../../docs/benchmarks/phone_v1_alias_report.md) | Документированное исправление HDF5 имён: 79/88 пригодных эталонов. Это повторный, уже раскрытый тест. |
| 09 | [cc_v6](../../docs/benchmarks/cc_v6_report.md) | Скрещивание канонического базиса и физического критика: улучшения не сложились. |
| 10 | [fourier_ridge_v1](../../docs/benchmarks/fourier_representation_report.md) | Выпуклая регрессия в пространстве цветовых гистограмм проверяет необходимость пространственной CNN. |
| 11 | [fourier_ridge_v2](../../docs/benchmarks/fourier_representation_report.md) | Отмена глобального prior и проверка границы регуляризации; отдельный source screen. |
| 12 | [cc_v7](../../docs/benchmarks/cc_v7/seed17_report.md) | Семантический teacher и виртуальный сенсор: стандартные контроли остаются конкурентными. |
| 13 | [cc_v7_external](../../docs/benchmarks/cc_v7_external/report.md) | 29 заранее зафиксированных методов; новые изображения внешних камер, с оговоркой о сценах и истории камер. |
| 14 | [skin_he_xyz_v1](../../docs/benchmarks/skin_he_xyz_v1/report.md) | Реальные парные региональные RGB/XYZ, 20 тестовых людей. XYZ RMSE, без выдуманного ΔE00 при неизвестном белом. |
| 15 | [projector_probe](../../docs/benchmarks/projector_probe_report.md) | Проверка геометрии допустимых поправок; привилегированный диагностический контроль, не skin accuracy. |
| 16 | [skin_mskcc_summary_v1](../../docs/benchmarks/skin_mskcc_summary_v1/report.md) | Первый MSKCC контроль по трём опубликованным медианным image-Lab признакам, только source validation. |
| 17 | [skin_mskcc_pixel_ablation_v2](../../docs/benchmarks/skin_mskcc_pixel_ablation_v2/report.md) | Локальные/глобальные голоса и усиленная робастная итерация. Изъятие контекста не помогло. |
| 18 | [skin_mskcc_pixels_v1](../../docs/benchmarks/skin_mskcc_pixels_v1/report.md) | Переход от готовых региональных статистик к JPEG-to-Lab; CNN, patch votes, fusion и robust recurrence. |
| 19 | [skin_mskcc_selective_v1](../../docs/benchmarks/skin_mskcc_selective_v1/report.md) | Основной независимый тест: 400 изображений / 10 новых людей. 4,457 ΔE00; обычная fusion сильнее. |
| 20 | [skin_pair_v1](../../docs/benchmarks/skin_pair_v1/report.md) | Paired invariance, quotient, output consistency и VICReg; пара снимков не устранила зависимость от условий съёмки. |
| 21 | [skin_capture_v1](../../docs/benchmarks/skin_capture_v1/report.md) | Скрытые условия съёмки и смесь четырёх гипотез. Сильный компактный source baseline: 929 297 параметров. |
| 22 | [skin_spectral_probe_v1](../../docs/benchmarks/skin_spectral_probe_v1/report.md) | 22 500 спектров из трёх TRAIN лиц. Представимость материала проверена отдельно от восстановления по RGB. |
| 23 | [skin_branch_combination_v1](../../docs/benchmarks/skin_branch_combination_v1/report.md) | Равновесные пары уже обученных моделей: source gains при удвоенной стоимости, не новый независимый результат. |
| 24 | [skin_spatial_v1](../../docs/benchmarks/skin_spatial_v1/report.md) | Свёрточные и recurrent graph ветви с геометрическими контролями. Дополнительные проходы не дали общего выигрыша. |
| 25 | [skin_train_branch_v1](../../docs/benchmarks/skin_train_branch_v1/report.md) | Граф/свёртка только при обучении; направленные gains без универсальной победы и независимого подтверждения. |
| 26 | [skin_copula_v1](../../docs/benchmarks/skin_copula_v1/report.md) | Ранги, copula и абсолютные гистограммы. Удаление абсолютного цвета ухудшает точность. |
| 27 | [skin_nuisance_v1](../../docs/benchmarks/skin_nuisance_v1/report.md) | Learned graph, grid, global graph и constant bias; инвариантность не должна уничтожать полезный абсолютный сигнал. |
| 28 | [skin_correspondence_v1](../../docs/benchmarks/skin_correspondence_v1/report.md) | Повторяемость реальных приборных измерений и многовидовые диагностики; это не известный предел ошибки модели. |
| 29 | [skin_shared_bias_v1](../../docs/benchmarks/skin_shared_bias_v1/report.md) | Общий bias нескольких снимков против их согласованности; согласие не гарантирует правильный цвет. |
| 30 | [skin_teacher_readout_v1](../../docs/benchmarks/skin_teacher_readout_v1/report.md) | Frozen foundation features, ridge и shuffled controls: перенос семантики не стал универсальной колориметрией. |
| 31 | [skin_local_teacher_v1](../../docs/benchmarks/skin_local_teacher_v1/report.md) | Локальное соответствие teacher/RGB: aligned, global и shuffled контроли, отрицательный перенос. |
| 32 | [skin_issa_v1](../../docs/benchmarks/skin_issa_v1/report.md) | Измеренные спектры кожи: oracle compression с полным спектром на входе, не точность фотографии. |
| 33 | [skin_material_image_v1](../../docs/benchmarks/skin_material_image_v1/report.md) | Физический спектральный декодер на реальных изображениях: диапазон представимых цветов не объясняет основной провал. |
| 34 | [skin_distribution_v1](../../docs/benchmarks/skin_distribution_v1/report.md) | Условные распределения цвета и решения по ожидаемой ошибке; универсального выигрыша над прямой регрессией нет. |
| 35 | [skin_risk_cross_v1](../../docs/benchmarks/skin_risk_cross_v1/report.md) | Перекрёстные сочетания цвета и риска: 90 endpoints из существующих моделей, не 90 новых обучений. |
| 36 | [skin_loss_field_v1](../../docs/benchmarks/skin_loss_field_v1/report.md) | Предсказывать поле цветовой ошибки вместо единственного ответа; 15 625 кандидатов не дали общей победы. |
| 37 | [skin_capture_support_v1](../../docs/benchmarks/skin_capture_support_v1/report.md) | Перемешивание реально наблюдавшихся патчей одного участка кожи; одинаковый Lab не означает регистрацию пикселей. |
| 38 | [skin_expert_anchor_v1](../../docs/benchmarks/skin_expert_anchor_v1/report.md) | Удаление и явная супервизия экспертов; отдельный forward gain, общей победы нет. |
| 39 | [skin_appearance_inverse_v1](../../docs/benchmarks/skin_appearance_inverse_v1/report.md) | Обратная задача: предсказывать распределение наблюдения по цвету и инвертировать его; универсального выигрыша нет. |
| 40 | [skin_graph_support_v1](../../docs/benchmarks/skin_graph_support_v1/report.md) | Комбинация графа и реальных paired patches помогает слабому transfer baseline, но не сильнейшим рецептам. |
| 41 | [skin_patch_likelihood_v1](../../docs/benchmarks/skin_patch_likelihood_v1/report.md) | Распределение патчей против среднего изображения; сложность не принесла универсальной точности. |
| 42 | [skin_support_curve_v1](../../docs/benchmarks/skin_support_curve_v1/report.md) | 6/12/18 обучающих людей при одинаковом числе обновлений: больше людей помогло, pixel adapter не убедил. |
| 43 | [skin_color_sampling_v1](../../docs/benchmarks/skin_color_sampling_v1/report.md) | Распределение бюджета по измеренному цвету и людям; небольшие внутренние улучшения, слабое отличие от обычной балансировки. |
| 44 | [skin_color_sampling_mass_v1](../../docs/benchmarks/skin_color_sampling_v1/report.md) | Шесть дополнительных контролей массы людей к sampling screen; включены в общий отчёт, не считать дважды. |
| 45 | [skin_sampling_transfer_v1](../../docs/benchmarks/skin_sampling_transfer_v1/report.md) | Скрещивание цветовой и person/site балансировки: небольшие known gains не переносятся в обе стороны. |
| 46 | [skin_offset_diagnostic_v1](../../docs/benchmarks/skin_offset_diagnostic_v1/report.md) | Привилегированные offsets по эталонам других evaluation людей; не результат без калибровки. |
| 47 | [skin_gradient_transfer_v1](../../docs/benchmarks/skin_gradient_transfer_v1/report.md) | Градиенты людей, перемешанные группы и малые шаги: уменьшение TRAIN loss не гарантирует уменьшения ΔE00. |
| 48 | [skin_relational_probe_v1](../../docs/benchmarks/skin_relational_probe_v1/report.md) | Сравнение участков и ложных пар; 1 421 пара не создаёт новых независимых людей или cross-camera пар. |
| 49 | [skin_local_reference_v1](../../docs/benchmarks/skin_local_reference_v1/report.md) | Локальная аффинная регрессия на TRAIN leave-person-out улучшает ridge. Известный статистический механизм. |
| 50 | [skin_local_reference_transfer_v1](../../docs/benchmarks/skin_local_reference_transfer_v1/report.md) | Перенос локального эталонного банка на камеры не побеждает сильные нейронные контроли. |
| 51 | [skin_neural_reference_v1](../../docs/benchmarks/skin_neural_reference_v1/report.md) | Более ёмкий neural reference head в лимите +200k; снижение TRAIN ошибки не перенеслось универсально. |
| 52 | [skin_crossfit_correction_v1](../../docs/benchmarks/skin_crossfit_correction_v1/report.md) | Внутренний выигрыш 7,40% при +188 035 параметрах. Гипотеза превосходства OOF не подтвердилась. |
| 53 | [skin_correction_transfer_v1](../../docs/benchmarks/skin_correction_transfer_v1/report.md) | Финальная проверка: все три correction head ухудшили оба unseen-camera направления. Исследования остановлены. |
