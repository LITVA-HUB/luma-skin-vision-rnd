# `scripts/build_research_archive.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/build_research_archive.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Build the public research paper's static figures and complete evidence index.

Reads archived reports/aggregate JSON only. Does not decode datasets, train models,
rescore the exposed test, or change any frozen evidence. Python >=3.11.

SHA-256 исходника: `bd1fce1c9010370686439d20af1f312da0f4d8f7fb3d60272332102897258f61`. Строк: **555**.

## Зависимости

```python
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 20](../../../../scripts/build_research_archive.py#L20)

```python
ROOT = Path(__file__).resolve().parents[1]
```

[Строка 21](../../../../scripts/build_research_archive.py#L21)

```python
OUT = ROOT / "docs" / "publication"
```

[Строка 22](../../../../scripts/build_research_archive.py#L22)

```python
FIG = OUT / "figures"
```

[Строка 40](../../../../scripts/build_research_archive.py#L40)

```python
NOTES = {
    "cc_v2": "Якорный остаточный метод: выигрыш у matched C+ на внешних камерах, регрессия на части доменов.",
    "cc_v3": "Полный цветовой базис и граф: проигрыш обычному RGB при одинаковой ёмкости.",
    "cc_v4": "Оценка исправленного цвета и повторное уточнение: механизм transport не выиграл source screen.",
    "cc_v5": "Парное обучение критика и равное число запросов: устойчивого преимущества повторных проходов нет.",
    "cc_v6": "Скрещивание канонического базиса и физического критика: улучшения не сложились.",
    "cc_v7": "Семантический teacher и виртуальный сенсор: стандартные контроли остаются конкурентными.",
    "cc_v7_external": "29 заранее зафиксированных методов; новые изображения внешних камер, с оговоркой о сценах и истории камер.",
    "fourier_ridge_v1": "Выпуклая регрессия в пространстве цветовых гистограмм проверяет необходимость пространственной CNN.",
    "fourier_ridge_v2": "Отмена глобального prior и проверка границы регуляризации; отдельный source screen.",
    "phone_v1": "Первый фиксированный Samsung/Oppo тест; строгий HDF5 loader ограничивал достижимое покрытие.",
    "phone_v1_alias": "Документированное исправление HDF5 имён: 79/88 пригодных эталонов. Это повторный, уже раскрытый тест.",
    "projector_probe": "Проверка геометрии допустимых поправок; привилегированный диагностический контроль, не skin accuracy.",
    "public_runs": "Первый реальный SimpleCube++ benchmark: классические и компактные нейросетевые контроли.",
    "source_snapshots": "Архив исходников при исправлении формата. Это служебная доказательная папка, не поколение модели.",
    "skin_appearance_inverse_v1": "Обратная задача: предсказывать распределение наблюдения по цвету и инвертировать его; универсального выигрыша нет.",
    "skin_branch_combination_v1": "Равновесные пары уже обученных моделей: source gains при удвоенной стоимости, не новый независимый результат.",
    "skin_capture_v1": "Скрытые условия съёмки и смесь четырёх гипотез. Сильный компактный source baseline: 929 297 параметров.",
    "skin_capture_support_v1": "Перемешивание реально наблюдавшихся патчей одного участка кожи; одинаковый Lab не означает регистрацию пикселей.",
    "skin_color_sampling_v1": "Распределение бюджета по измеренному цвету и людям; небольшие внутренние улучшения, слабое отличие от обычной балансировки.",
    "skin_color_sampling_mass_v1": "Шесть дополнительных контролей массы людей к sampling screen; включены в общий отчёт, не считать дважды.",
    "skin_copula_v1": "Ранги, copula и абсолютные гистограммы. Удаление абсолютного цвета ухудшает точность.",
    "skin_correction_transfer_v1": "Финальная проверка: все три correction head ухудшили оба unseen-camera направления. Исследования остановлены.",
    "skin_correspondence_v1": "Повторяемость реальных приборных измерений и многовидовые диагностики; это не известный предел ошибки модели.",
    "skin_crossfit_correction_v1": "Внутренний выигрыш 7,40% при +188 035 параметрах. Гипотеза превосходства OOF не подтвердилась.",
    "skin_distribution_v1": "Условные распределения цвета и решения по ожидаемой ошибке; универсального выигрыша над прямой регрессией нет.",
    "skin_expert_anchor_v1": "Удаление и явная супервизия экспертов; отдельный forward gain, общей победы нет.",
    "skin_gradient_transfer_v1": "Градиенты людей, перемешанные группы и малые шаги: уменьшение TRAIN loss не гарантирует уменьшения ΔE00.",
    "skin_graph_support_v1": "Комбинация графа и реальных paired patches помогает слабому transfer baseline, но не сильнейшим рецептам.",
    "skin_he_xyz_v1": "Реальные парные региональные RGB/XYZ, 20 тестовых людей. XYZ RMSE, без выдуманного ΔE00 при неизвестном белом.",
    "skin_issa_v1": "Измеренные спектры кожи: oracle compression с полным спектром на входе, не точность фотографии.",
    "skin_local_reference_v1": "Локальная аффинная регрессия на TRAIN leave-person-out улучшает ridge. Известный статистический механизм.",
    "skin_local_reference_transfer_v1": "Перенос локального эталонного банка на камеры не побеждает сильные нейронные контроли.",
    "skin_local_teacher_v1": "Локальное соответствие teacher/RGB: aligned, global и shuffled контроли, отрицательный перенос.",
    "skin_loss_field_v1": "Предсказывать поле цветовой ошибки вместо единственного ответа; 15 625 кандидатов не дали общей победы.",
    "skin_material_image_v1": "Физический спектральный декодер на реальных изображениях: диапазон представимых цветов не объясняет основной провал.",
    "skin_mskcc_pixel_ablation_v2": "Локальные/глобальные голоса и усиленная робастная итерация. Изъятие контекста не помогло.",
    "skin_mskcc_pixels_v1": "Переход от готовых региональных статистик к JPEG-to-Lab; CNN, patch votes, fusion и robust recurrence.",
    "skin_mskcc_selective_v1": "Основной независимый тест: 400 изображений / 10 новых людей. 4,457 ΔE00; обычная fusion сильнее.",
    "skin_mskcc_summary_v1": "Первый MSKCC контроль по трём опубликованным медианным image-Lab признакам, только source validation.",
    "skin_neural_reference_v1": "Более ёмкий neural reference head в лимите +200k; снижение TRAIN ошибки не перенеслось универсально.",
    "skin_nuisance_v1": "Learned graph, grid, global graph и constant bias; инвариантность не должна уничтожать полезный абсолютный сигнал.",
    "skin_offset_diagnostic_v1": "Привилегированные offsets по эталонам других evaluation людей; не результат без калибровки.",
    "skin_pair_v1": "Paired invariance, quotient, output consistency и VICReg; пара снимков не устранила зависимость от условий съёмки.",
    "skin_patch_likelihood_v1": "Распределение патчей против среднего изображения; сложность не принесла универсальной точности.",
    "skin_relational_probe_v1": "Сравнение участков и ложных пар; 1 421 пара не создаёт новых независимых людей или cross-camera пар.",
    "skin_risk_cross_v1": "Перекрёстные сочетания цвета и риска: 90 endpoints из существующих моделей, не 90 новых обучений.",
    "skin_sampling_transfer_v1": "Скрещивание цветовой и person/site балансировки: небольшие known gains не переносятся в обе стороны.",
    "skin_shared_bias_v1": "Общий bias нескольких снимков против их согласованности; согласие не гарантирует правильный цвет.",
    "skin_spatial_v1": "Свёрточные и recurrent graph ветви с геометрическими контролями. Дополнительные проходы не дали общего выигрыша.",
    "skin_spectral_probe_v1": "22 500 спектров из трёх TRAIN лиц. Представимость материала проверена отдельно от восстановления по RGB.",
    "skin_support_curve_v1": "6/12/18 обучающих людей при одинаковом числе обновлений: больше людей помогло, pixel adapter не убедил.",
    "skin_teacher_readout_v1": "Frozen foundation features, ridge и shuffled controls: перенос семантики не стал универсальной колориметрией.",
    "skin_train_branch_v1": "Граф/свёртка только при обучении; направленные gains без универсальной победы и независимого подтверждения.",
}
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `read` | FunctionDef | См. реализацию | [L97](../../../../scripts/build_research_archive.py#L97) |
| `load` | FunctionDef | См. реализацию | [L101](../../../../scripts/build_research_archive.py#L101) |
| `write` | FunctionDef | См. реализацию | [L105](../../../../scripts/build_research_archive.py#L105) |
| `save` | FunctionDef | См. реализацию | [L111](../../../../scripts/build_research_archive.py#L111) |
| `tables` | FunctionDef | См. реализацию | [L123](../../../../scripts/build_research_archive.py#L123) |
| `overview_figures` | FunctionDef | См. реализацию | [L136](../../../../scripts/build_research_archive.py#L136) |
| `report_for` | FunctionDef | См. реализацию | [L273](../../../../scripts/build_research_archive.py#L273) |
| `family_plot` | FunctionDef | Plot literal rounded report cells, never pretend they are pooled raw data. | [L292](../../../../scripts/build_research_archive.py#L292) |
| `main` | FunctionDef | См. реализацию | [L382](../../../../scripts/build_research_archive.py#L382) |
