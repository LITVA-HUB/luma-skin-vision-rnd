# Luma ChromaSeed AS / HR · patch_small

[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)

**17,374 параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.

## Полная спецификация слоёв

| Слой | Вход → выход | Bias | Параметры |
|---|---|---|---:|
| `token1` | 18 → 32 | да | 608 |
| `token2` | 32 → 24 | да | 792 |
| `mlp1` | 60 → 96 | да | 5,856 |
| `mlp2` | 96 → 64 | да | 6,208 |
| `mlp3` | 64 → 48 | да | 3,120 |
| `head` | 48 → 3 | да | 147 |
| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |
| **Всего** | | | **17,374** |

## Прямой проход

Два token-слоя с SiLU; mean по 64 токенам; concat с color36; три MLP-слоя с SiLU; head3 и прибавление поправки к NP. Один ответ, рекуррентных шагов нет.

## Три версии выходной функции

| Версия | Формула | Новые веса |
|---|---|---:|
| unit / исходный AS | tanh(z) | 0 |
| wide / HR | 4·tanh(z/4) | 0 |
| linear / HR | z | 0 |

Изменяется residual output head. Остальные нелинейности и рекуррентный state update сохранены. Все три имеют value0/derivative1 в начале координат. HR unit использует точный AS-контроль.

## Обучение и выбор

NP заморожен; FP32 residual bank из 6 слотов (seeds17/29/43 ×LR1e-5/1e-4). Batch64, AdamW decay.01, clip5; фиксированный cosine horizon8192. Loss=.6 mean(all-pass MSE)+.4 last-pass MSE+.001 gate penalty. Выбор шага/LR только по INNER. CUDA Graph ускоряет исполнение одинакового правила; в NumPy ответ считается FP64 после FP32-нормализации.

## Все сохранённые AS-результаты этого варианта

| Роль | Выбранный шаг | LR | ΔE00 | CPU median, мкс |
|---|---:|---:|---:|---:|
| mixed | 2048 | 0.0001 | 5.636765220 | 50.4 |
| slr_to_ipod | 2048 | 0.0001 | 8.004457071 | 49.6 |
| ipod_to_slr | 2048 | 0.0001 | 8.898108833 | 50.05 |

Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.

## HR: все head / role результаты

| Head | Роль | Средняя ΔE00 |
|---|---|---:|
| linear | ipod_to_slr | 8.903616475 |
| unit | ipod_to_slr | 8.898108833 |
| wide | ipod_to_slr | 8.901583909 |
| linear | mixed | 5.637631176 |
| unit | mixed | 5.636765220 |
| wide | mixed | 5.637596861 |
| linear | slr_to_ipod | 8.295459550 |
| unit | slr_to_ipod | 8.004457071 |
| wide | slr_to_ipod | 8.295459550 |

**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).

## Реализация и тесты

- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)
- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)
- [Все тесты соответствующих серий](../EXPERIMENTS.md)

Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.
