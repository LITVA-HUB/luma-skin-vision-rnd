# Галерея исследовательских поколений

Все изображения — графики измерений или схемы. Фотографий участников здесь нет. Графики не образуют единую шкалу прогресса: протоколы и единицы различаются.

[Атлас](GENERATIONS.md) · [Основной отчёт](RESEARCH_REPORT_RU.md)

![Путь исследования](figures/research_journey.png)

![Независимый тест](figures/independent_skin.png)

![Последняя проверка](figures/correction_falsifier.png)

## 01 · public_runs

Первый реальный SimpleCube++ benchmark: классические и компактные нейросетевые контроли.

[Методика, все результаты и ограничения](../../docs/benchmarks/public_benchmark_report.md)

![public_runs — 1](figures/family_public_runs.png)


## 02 · source_snapshots

Архив исходников при исправлении формата. Это служебная доказательная папка, не поколение модели.

[Методика, все результаты и ограничения](../../docs/benchmarks/source_snapshots)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 03 · cc_v2

Якорный остаточный метод: выигрыш у matched C+ на внешних камерах, регрессия на части доменов.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v2_report.md)

![cc_v2 — 1](figures/family_cc_v2.png)

![cc_v2 — 2](../../docs/benchmarks/cc_v2/figures/camera_comparison.png)

![cc_v2 — 3](../../docs/benchmarks/cc_v2/figures/frozen_thresholds.png)

![cc_v2 — 4](../../docs/benchmarks/cc_v2/figures/risk_coverage.png)


## 04 · cc_v3

Полный цветовой базис и граф: проигрыш обычному RGB при одинаковой ёмкости.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v3_report.md)

![cc_v3 — 1](figures/family_cc_v3.png)

![cc_v3 — 2](../../docs/benchmarks/cc_v3/source_screen/source_screen.png)


## 05 · cc_v4

Оценка исправленного цвета и повторное уточнение: механизм transport не выиграл source screen.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v4_report.md)

![cc_v4 — 1](figures/family_cc_v4.png)

![cc_v4 — 2](../../docs/benchmarks/cc_v4/source_screen/source_screen.png)


## 06 · cc_v5

Парное обучение критика и равное число запросов: устойчивого преимущества повторных проходов нет.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v5_report.md)

![cc_v5 — 1](figures/family_cc_v5.png)

![cc_v5 — 2](../../docs/benchmarks/cc_v5/final_summary/risk_coverage.png)


## 07 · phone_v1

Первый фиксированный Samsung/Oppo тест; строгий HDF5 loader ограничивал достижимое покрытие.

[Методика, все результаты и ограничения](../../docs/benchmarks/phone_v1_report.md)

![phone_v1 — 1](figures/family_phone_v1.png)

![phone_v1 — 2](../../docs/benchmarks/phone_v1/risk_coverage.png)


## 08 · phone_v1_alias

Документированное исправление HDF5 имён: 79/88 пригодных эталонов. Это повторный, уже раскрытый тест.

[Методика, все результаты и ограничения](../../docs/benchmarks/phone_v1_alias_report.md)

![phone_v1_alias — 1](figures/family_phone_v1_alias.png)

![phone_v1_alias — 2](../../docs/benchmarks/phone_v1_alias/risk_coverage.png)


## 09 · cc_v6

Скрещивание канонического базиса и физического критика: улучшения не сложились.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v6_report.md)

![cc_v6 — 1](figures/family_cc_v6.png)


## 10 · fourier_ridge_v1

Выпуклая регрессия в пространстве цветовых гистограмм проверяет необходимость пространственной CNN.

[Методика, все результаты и ограничения](../../docs/benchmarks/fourier_representation_report.md)

![fourier_ridge_v1 — 1](figures/family_fourier_ridge_v1.png)


## 11 · fourier_ridge_v2

Отмена глобального prior и проверка границы регуляризации; отдельный source screen.

[Методика, все результаты и ограничения](../../docs/benchmarks/fourier_representation_report.md)

![fourier_ridge_v2 — 1](figures/family_fourier_ridge_v2.png)


## 12 · cc_v7

Семантический teacher и виртуальный сенсор: стандартные контроли остаются конкурентными.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v7/seed17_report.md)

![cc_v7 — 1](figures/family_cc_v7.png)


## 13 · cc_v7_external

29 заранее зафиксированных методов; новые изображения внешних камер, с оговоркой о сценах и истории камер.

[Методика, все результаты и ограничения](../../docs/benchmarks/cc_v7_external/report.md)

![cc_v7_external — 1](figures/family_cc_v7_external.png)

![cc_v7_external — 2](../../docs/benchmarks/cc_v7_external/report/risk_coverage.png)


## 14 · skin_he_xyz_v1

Реальные парные региональные RGB/XYZ, 20 тестовых людей. XYZ RMSE, без выдуманного ΔE00 при неизвестном белом.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_he_xyz_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 15 · projector_probe

Проверка геометрии допустимых поправок; привилегированный диагностический контроль, не skin accuracy.

[Методика, все результаты и ограничения](../../docs/benchmarks/projector_probe_report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 16 · skin_mskcc_summary_v1

Первый MSKCC контроль по трём опубликованным медианным image-Lab признакам, только source validation.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_mskcc_summary_v1/report.md)

![skin_mskcc_summary_v1 — 1](figures/family_skin_mskcc_summary_v1.png)


## 17 · skin_mskcc_pixel_ablation_v2

Локальные/глобальные голоса и усиленная робастная итерация. Изъятие контекста не помогло.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_mskcc_pixel_ablation_v2/report.md)

![skin_mskcc_pixel_ablation_v2 — 1](figures/family_skin_mskcc_pixel_ablation_v2.png)


## 18 · skin_mskcc_pixels_v1

Переход от готовых региональных статистик к JPEG-to-Lab; CNN, patch votes, fusion и robust recurrence.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_mskcc_pixels_v1/report.md)

![skin_mskcc_pixels_v1 — 1](figures/family_skin_mskcc_pixels_v1.png)

![skin_mskcc_pixels_v1 — 2](../../docs/benchmarks/skin_mskcc_pixels_v1/risk_coverage.png)


## 19 · skin_mskcc_selective_v1

Основной независимый тест: 400 изображений / 10 новых людей. 4,457 ΔE00; обычная fusion сильнее.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_mskcc_selective_v1/report.md)

![skin_mskcc_selective_v1 — 1](../../docs/benchmarks/skin_mskcc_selective_v1/risk_coverage.png)


## 20 · skin_pair_v1

Paired invariance, quotient, output consistency и VICReg; пара снимков не устранила зависимость от условий съёмки.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_pair_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 21 · skin_capture_v1

Скрытые условия съёмки и смесь четырёх гипотез. Сильный компактный source baseline: 929 297 параметров.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_capture_v1/report.md)

![skin_capture_v1 — 1](figures/family_skin_capture_v1.png)


## 22 · skin_spectral_probe_v1

22 500 спектров из трёх TRAIN лиц. Представимость материала проверена отдельно от восстановления по RGB.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_spectral_probe_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 23 · skin_branch_combination_v1

Равновесные пары уже обученных моделей: source gains при удвоенной стоимости, не новый независимый результат.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_branch_combination_v1/report.md)

![skin_branch_combination_v1 — 1](figures/family_skin_branch_combination_v1.png)


## 24 · skin_spatial_v1

Свёрточные и recurrent graph ветви с геометрическими контролями. Дополнительные проходы не дали общего выигрыша.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_spatial_v1/report.md)

![skin_spatial_v1 — 1](figures/family_skin_spatial_v1.png)


## 25 · skin_train_branch_v1

Граф/свёртка только при обучении; направленные gains без универсальной победы и независимого подтверждения.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_train_branch_v1/report.md)

![skin_train_branch_v1 — 1](figures/family_skin_train_branch_v1.png)


## 26 · skin_copula_v1

Ранги, copula и абсолютные гистограммы. Удаление абсолютного цвета ухудшает точность.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_copula_v1/report.md)

![skin_copula_v1 — 1](figures/family_skin_copula_v1.png)


## 27 · skin_nuisance_v1

Learned graph, grid, global graph и constant bias; инвариантность не должна уничтожать полезный абсолютный сигнал.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_nuisance_v1/report.md)

![skin_nuisance_v1 — 1](figures/family_skin_nuisance_v1.png)


## 28 · skin_correspondence_v1

Повторяемость реальных приборных измерений и многовидовые диагностики; это не известный предел ошибки модели.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_correspondence_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 29 · skin_shared_bias_v1

Общий bias нескольких снимков против их согласованности; согласие не гарантирует правильный цвет.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_shared_bias_v1/report.md)

![skin_shared_bias_v1 — 1](figures/family_skin_shared_bias_v1.png)


## 30 · skin_teacher_readout_v1

Frozen foundation features, ridge и shuffled controls: перенос семантики не стал универсальной колориметрией.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_teacher_readout_v1/report.md)

![skin_teacher_readout_v1 — 1](figures/family_skin_teacher_readout_v1.png)


## 31 · skin_local_teacher_v1

Локальное соответствие teacher/RGB: aligned, global и shuffled контроли, отрицательный перенос.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_local_teacher_v1/report.md)

![skin_local_teacher_v1 — 1](figures/family_skin_local_teacher_v1.png)


## 32 · skin_issa_v1

Измеренные спектры кожи: oracle compression с полным спектром на входе, не точность фотографии.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_issa_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 33 · skin_material_image_v1

Физический спектральный декодер на реальных изображениях: диапазон представимых цветов не объясняет основной провал.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_material_image_v1/report.md)

![skin_material_image_v1 — 1](figures/family_skin_material_image_v1.png)

![skin_material_image_v1 — 2](../../docs/benchmarks/skin_material_image_v1/risk_coverage.png)


## 34 · skin_distribution_v1

Условные распределения цвета и решения по ожидаемой ошибке; универсального выигрыша над прямой регрессией нет.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_distribution_v1/report.md)

![skin_distribution_v1 — 1](figures/family_skin_distribution_v1.png)

![skin_distribution_v1 — 2](../../docs/benchmarks/skin_distribution_v1/risk_coverage.png)


## 35 · skin_risk_cross_v1

Перекрёстные сочетания цвета и риска: 90 endpoints из существующих моделей, не 90 новых обучений.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_risk_cross_v1/report.md)

![skin_risk_cross_v1 — 1](figures/family_skin_risk_cross_v1.png)

![skin_risk_cross_v1 — 2](../../docs/benchmarks/skin_risk_cross_v1/risk_coverage.png)


## 36 · skin_loss_field_v1

Предсказывать поле цветовой ошибки вместо единственного ответа; 15 625 кандидатов не дали общей победы.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_loss_field_v1/report.md)

![skin_loss_field_v1 — 1](figures/family_skin_loss_field_v1.png)

![skin_loss_field_v1 — 2](../../docs/benchmarks/skin_loss_field_v1/risk_coverage.png)


## 37 · skin_capture_support_v1

Перемешивание реально наблюдавшихся патчей одного участка кожи; одинаковый Lab не означает регистрацию пикселей.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_capture_support_v1/report.md)

![skin_capture_support_v1 — 1](figures/family_skin_capture_support_v1.png)

![skin_capture_support_v1 — 2](../../docs/benchmarks/skin_capture_support_v1/risk_coverage.png)


## 38 · skin_expert_anchor_v1

Удаление и явная супервизия экспертов; отдельный forward gain, общей победы нет.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_expert_anchor_v1/report.md)

![skin_expert_anchor_v1 — 1](figures/family_skin_expert_anchor_v1.png)

![skin_expert_anchor_v1 — 2](../../docs/benchmarks/skin_expert_anchor_v1/risk_coverage.png)


## 39 · skin_appearance_inverse_v1

Обратная задача: предсказывать распределение наблюдения по цвету и инвертировать его; универсального выигрыша нет.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_appearance_inverse_v1/report.md)

![skin_appearance_inverse_v1 — 1](figures/family_skin_appearance_inverse_v1.png)

![skin_appearance_inverse_v1 — 2](../../docs/benchmarks/skin_appearance_inverse_v1/risk_coverage.png)


## 40 · skin_graph_support_v1

Комбинация графа и реальных paired patches помогает слабому transfer baseline, но не сильнейшим рецептам.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_graph_support_v1/report.md)

![skin_graph_support_v1 — 1](figures/family_skin_graph_support_v1.png)

![skin_graph_support_v1 — 2](../../docs/benchmarks/skin_graph_support_v1/risk_coverage.png)


## 41 · skin_patch_likelihood_v1

Распределение патчей против среднего изображения; сложность не принесла универсальной точности.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_patch_likelihood_v1/report.md)

![skin_patch_likelihood_v1 — 1](figures/family_skin_patch_likelihood_v1.png)

![skin_patch_likelihood_v1 — 2](../../docs/benchmarks/skin_patch_likelihood_v1/risk_coverage.png)


## 42 · skin_support_curve_v1

6/12/18 обучающих людей при одинаковом числе обновлений: больше людей помогло, pixel adapter не убедил.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_support_curve_v1/report.md)

![skin_support_curve_v1 — 1](figures/family_skin_support_curve_v1.png)

![skin_support_curve_v1 — 2](../../docs/benchmarks/skin_support_curve_v1/learning_curve.png)


## 43 · skin_color_sampling_v1

Распределение бюджета по измеренному цвету и людям; небольшие внутренние улучшения, слабое отличие от обычной балансировки.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_color_sampling_v1/report.md)

![skin_color_sampling_v1 — 1](figures/family_skin_color_sampling_v1.png)

![skin_color_sampling_v1 — 2](../../docs/benchmarks/skin_color_sampling_v1/risk_coverage.png)


## 44 · skin_color_sampling_mass_v1

Шесть дополнительных контролей массы людей к sampling screen; включены в общий отчёт, не считать дважды.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_color_sampling_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 45 · skin_sampling_transfer_v1

Скрещивание цветовой и person/site балансировки: небольшие known gains не переносятся в обе стороны.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_sampling_transfer_v1/report.md)

![skin_sampling_transfer_v1 — 1](figures/family_skin_sampling_transfer_v1.png)

![skin_sampling_transfer_v1 — 2](../../docs/benchmarks/skin_sampling_transfer_v1/risk_coverage.png)


## 46 · skin_offset_diagnostic_v1

Привилегированные offsets по эталонам других evaluation людей; не результат без калибровки.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_offset_diagnostic_v1/report.md)

![skin_offset_diagnostic_v1 — 1](../../docs/benchmarks/skin_offset_diagnostic_v1/offset_diagnostic.png)


## 47 · skin_gradient_transfer_v1

Градиенты людей, перемешанные группы и малые шаги: уменьшение TRAIN loss не гарантирует уменьшения ΔE00.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_gradient_transfer_v1/report.md)

![skin_gradient_transfer_v1 — 1](../../docs/benchmarks/skin_gradient_transfer_v1/gradient_transfer.png)


## 48 · skin_relational_probe_v1

Сравнение участков и ложных пар; 1 421 пара не создаёт новых независимых людей или cross-camera пар.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_relational_probe_v1/report.md)

![skin_relational_probe_v1 — 1](../../docs/benchmarks/skin_relational_probe_v1/relational_probe.png)


## 49 · skin_local_reference_v1

Локальная аффинная регрессия на TRAIN leave-person-out улучшает ridge. Известный статистический механизм.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_local_reference_v1/report.md)

Числовые таблицы и/или диагностические доказательства находятся в отчёте. Отсутствующая метрика не заменена придуманной диаграммой.

## 50 · skin_local_reference_transfer_v1

Перенос локального эталонного банка на камеры не побеждает сильные нейронные контроли.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_local_reference_transfer_v1/report.md)

![skin_local_reference_transfer_v1 — 1](figures/family_skin_local_reference_transfer_v1.png)

![skin_local_reference_transfer_v1 — 2](../../docs/benchmarks/skin_local_reference_transfer_v1/local_reference_results.png)


## 51 · skin_neural_reference_v1

Более ёмкий neural reference head в лимите +200k; снижение TRAIN ошибки не перенеслось универсально.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_neural_reference_v1/report.md)

![skin_neural_reference_v1 — 1](../../docs/benchmarks/skin_neural_reference_v1/risk_coverage.png)


## 52 · skin_crossfit_correction_v1

Внутренний выигрыш 7,40% при +188 035 параметрах. Гипотеза превосходства OOF не подтвердилась.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_crossfit_correction_v1/report.md)

![skin_crossfit_correction_v1 — 1](figures/family_skin_crossfit_correction_v1.png)

![skin_crossfit_correction_v1 — 2](../../docs/benchmarks/skin_crossfit_correction_v1/correction_results.png)


## 53 · skin_correction_transfer_v1

Финальная проверка: все три correction head ухудшили оба unseen-camera направления. Исследования остановлены.

[Методика, все результаты и ограничения](../../docs/benchmarks/skin_correction_transfer_v1/report.md)

![skin_correction_transfer_v1 — 1](figures/family_skin_correction_transfer_v1.png)

![skin_correction_transfer_v1 — 2](../../docs/benchmarks/skin_correction_transfer_v1/risk_coverage.png)

