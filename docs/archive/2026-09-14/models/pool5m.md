# Luma ChromaSeed AS / HR · pool5m

[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)

**4,851,846 параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.

## Полная спецификация слоёв

| Слой | Вход → выход | Bias | Параметры |
|---|---|---|---:|
| `token1` | 18 → 512 | да | 9,728 |
| `mlp1` | 1572 → 3072 | да | 4,832,256 |
| `head` | 3072 → 3 | да | 9,219 |
| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |
| **Всего** | | | **4,851,846** |

## Прямой проход

Token1 +ReLU; mean/std/max по 64 токенам. Concat 36+3×512=1572. MLP +ReLU, head3, прибавление ограниченной поправки к NP. Std содержит +1e-6 под корнем.

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
| mixed | 2048 | 1e-05 | 5.403851404 | 442.3 |
| slr_to_ipod | 2048 | 1e-05 | 8.410492078 | 516.8 |
| ipod_to_slr | 512 | 1e-05 | 8.565986617 | 484.1 |

Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.

## HR: все head / role результаты

| Head | Роль | Средняя ΔE00 |
|---|---|---:|
| linear | ipod_to_slr | 8.554401784 |
| unit | ipod_to_slr | 8.565986617 |
| wide | ipod_to_slr | 8.548096317 |
| linear | mixed | 5.385838982 |
| unit | mixed | 5.403851404 |
| wide | mixed | 5.387278305 |
| linear | slr_to_ipod | 8.499825992 |
| unit | slr_to_ipod | 8.410492078 |
| wide | slr_to_ipod | 8.489491865 |

**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).

## Реализация и тесты

- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)
- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)
- [Все тесты соответствующих серий](../EXPERIMENTS.md)

Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.
