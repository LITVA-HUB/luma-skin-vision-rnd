# Сокращение модели и разрядности

Дополнительные опыты задуманы после внутреннего подбора основной серии, до внешней проверки. Все варианты сохранены. Выбор по внешней ошибке не выполнялся.

## Число элементов

Из обученных моделей на 64 элемента взяты первые K элементов и заново аналитически рассчитаны выходные веса. Порядок, нормализация и alpha исходного подбора сохранены. В стоимость входят исходный поиск 64 элементов и дополнительный пересчёт; это не замер обучения K элементов с нуля.

| Протокол | Метод | K | Байты FP32 | Отложенные люди ΔE00 | Выбран K внутренней проверкой |
|---|---|---:|---:|---:|---|
| mixed | random_rbf | 8 | 2036 | 6.0724 |  |
| mixed | random_rbf | 16 | 3316 | 6.0090 |  |
| mixed | random_rbf | 32 | 5876 | 5.9719 |  |
| mixed | random_rbf | 64 | 10996 | 5.7521 | да |
| mixed | guided_rbf | 8 | 2036 | 6.1505 |  |
| mixed | guided_rbf | 16 | 3316 | 6.0272 |  |
| mixed | guided_rbf | 32 | 5876 | 5.8802 |  |
| mixed | guided_rbf | 64 | 10996 | 5.8013 | да |
| slr_to_ipod | random_rbf | 8 | 2036 | 10.9764 | да |
| slr_to_ipod | random_rbf | 16 | 3316 | 10.9034 |  |
| slr_to_ipod | random_rbf | 32 | 5876 | 10.7489 |  |
| slr_to_ipod | random_rbf | 64 | 10996 | 10.3636 |  |
| slr_to_ipod | guided_rbf | 8 | 2036 | 10.7806 | да |
| slr_to_ipod | guided_rbf | 16 | 3316 | 10.5416 |  |
| slr_to_ipod | guided_rbf | 32 | 5876 | 10.3036 |  |
| slr_to_ipod | guided_rbf | 64 | 10996 | 10.0280 |  |
| ipod_to_slr | random_rbf | 8 | 2036 | 8.5967 |  |
| ipod_to_slr | random_rbf | 16 | 3316 | 8.5676 |  |
| ipod_to_slr | random_rbf | 32 | 5876 | 8.3956 |  |
| ipod_to_slr | random_rbf | 64 | 10996 | 8.0618 | да |
| ipod_to_slr | guided_rbf | 8 | 2036 | 7.8165 |  |
| ipod_to_slr | guided_rbf | 16 | 3316 | 7.5348 |  |
| ipod_to_slr | guided_rbf | 32 | 5876 | 7.7687 |  |
| ipod_to_slr | guided_rbf | 64 | 10996 | 7.7193 | да |

## Разрядность хранения

FP16 хранит все числовые массивы в 16 битах. INT8 хранит веса/центры с масштабами, нормализацию — в FP32. Перед вычислением оба варианта разворачиваются в FP32. Уменьшение файла не равно уменьшению рабочей памяти или ускорению native INT8/FP16 kernels.

Drift — отклонение предсказания от исходного FP32, а не ошибка относительно прибора. Таблица содержит максимум среди изображений и seeds. Формат не выбирался по этой проверке.

| Протокол | Метод | Хранение | Числовые байты | Отложенные люди ΔE00 | Максимальный drift ΔE00 |
|---|---|---|---:|---:|---:|
| mixed | ridge | fp32_reference | 756 | 6.5389 | 0.0000 |
| mixed | ridge | fp16 | 378 | 6.5416 | 0.0220 |
| mixed | ridge | int8 | 435 | 6.5406 | 0.1851 |
| mixed | krr | fp32_reference | 114820 | 5.3984 | 0.0000 |
| mixed | krr | fp16 | 57410 | 5.4002 | 0.1311 |
| mixed | krr | int8 | 29099 | 6.3922 | 6.9998 |
| mixed | random_rbf | fp32_reference | 10996 | 5.7521 | 0.0000 |
| mixed | random_rbf | fp16 | 5498 | 5.7488 | 0.0907 |
| mixed | random_rbf | int8 | 3143 | 5.8138 | 1.3248 |
| mixed | guided_rbf | fp32_reference | 10996 | 5.8013 | 0.0000 |
| mixed | guided_rbf | fp16 | 5498 | 5.8021 | 0.0197 |
| mixed | guided_rbf | int8 | 3143 | 5.8200 | 0.3469 |
| mixed | mlp | fp32_reference | 10996 | 5.7856 | 0.0000 |
| mixed | mlp | fp16 | 5498 | 5.7867 | 0.0219 |
| mixed | mlp | int8 | 3271 | 5.7807 | 0.1491 |
| slr_to_ipod | ridge | fp32_reference | 756 | 10.8739 | 0.0000 |
| slr_to_ipod | ridge | fp16 | 378 | 10.8919 | 0.0371 |
| slr_to_ipod | ridge | int8 | 435 | 10.9782 | 0.4871 |
| slr_to_ipod | krr | fp32_reference | 50704 | 8.5845 | 0.0000 |
| slr_to_ipod | krr | fp16 | 25352 | 8.6069 | 0.1404 |
| slr_to_ipod | krr | int8 | 13070 | 8.9000 | 3.3796 |
| slr_to_ipod | random_rbf | fp32_reference | 10996 | 10.3636 | 0.0000 |
| slr_to_ipod | random_rbf | fp16 | 5498 | 10.3793 | 0.0353 |
| slr_to_ipod | random_rbf | int8 | 3143 | 10.3776 | 0.4253 |
| slr_to_ipod | guided_rbf | fp32_reference | 10996 | 10.0280 | 0.0000 |
| slr_to_ipod | guided_rbf | fp16 | 5498 | 10.0422 | 0.0400 |
| slr_to_ipod | guided_rbf | int8 | 3143 | 10.0164 | 0.3200 |
| slr_to_ipod | mlp | fp32_reference | 10996 | 9.4810 | 0.0000 |
| slr_to_ipod | mlp | fp16 | 5498 | 9.4981 | 0.0462 |
| slr_to_ipod | mlp | int8 | 3271 | 9.4930 | 0.1680 |
| ipod_to_slr | ridge | fp32_reference | 756 | 7.9557 | 0.0000 |
| ipod_to_slr | ridge | fp16 | 378 | 7.9590 | 0.0160 |
| ipod_to_slr | ridge | int8 | 435 | 7.9550 | 0.1302 |
| ipod_to_slr | krr | fp32_reference | 100624 | 8.9125 | 0.0000 |
| ipod_to_slr | krr | fp16 | 50312 | 8.8987 | 0.1294 |
| ipod_to_slr | krr | int8 | 25550 | 8.4656 | 7.1017 |
| ipod_to_slr | random_rbf | fp32_reference | 10996 | 8.0618 | 0.0000 |
| ipod_to_slr | random_rbf | fp16 | 5498 | 8.0501 | 0.0965 |
| ipod_to_slr | random_rbf | int8 | 3143 | 8.4278 | 2.6002 |
| ipod_to_slr | guided_rbf | fp32_reference | 10996 | 7.7193 | 0.0000 |
| ipod_to_slr | guided_rbf | fp16 | 5498 | 7.7192 | 0.0478 |
| ipod_to_slr | guided_rbf | int8 | 3143 | 7.8633 | 1.0449 |
| ipod_to_slr | mlp | fp32_reference | 10996 | 9.7309 | 0.0000 |
| ipod_to_slr | mlp | fp16 | 5498 | 9.7403 | 0.0369 |
| ipod_to_slr | mlp | int8 | 3271 | 9.7468 | 0.5535 |

[Протокол сокращения](../../research/skin_local_search_compact_protocol.md) · [Протокол разрядности](../../research/skin_local_search_precision_protocol.md) · [Основной отчёт](report.md)
