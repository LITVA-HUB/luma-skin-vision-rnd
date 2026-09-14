# Luma ChromaSeed AS / HR · patch5m

[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)

**4,962,566 параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.

## Полная спецификация слоёв

| Слой | Вход → выход | Bias | Параметры |
|---|---|---|---:|
| `token1` | 18 → 384 | да | 7,296 |
| `token2` | 384 → 256 | да | 98,560 |
| `mlp1` | 292 → 1792 | да | 525,056 |
| `mlp2` | 1792 → 1536 | да | 2,754,048 |
| `mlp3` | 1536 → 1024 | да | 1,573,888 |
| `head` | 1024 → 3 | да | 3,075 |
| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |
| **Всего** | | | **4,962,566** |

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
| mixed | 2048 | 1e-05 | 5.606355472 | 954.6 |
| slr_to_ipod | 128 | 1e-05 | 8.280084259 | 959.8 |
| ipod_to_slr | 2048 | 1e-05 | 8.970359153 | 941.7 |

Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.

## HR: все head / role результаты

| Head | Роль | Средняя ΔE00 |
|---|---|---:|
| linear | ipod_to_slr | 8.978130204 |
| unit | ipod_to_slr | 8.970359153 |
| wide | ipod_to_slr | 8.977498806 |
| linear | mixed | 5.605393824 |
| unit | mixed | 5.606355472 |
| wide | mixed | 5.605472334 |
| linear | slr_to_ipod | 8.280396273 |
| unit | slr_to_ipod | 8.280084259 |
| wide | slr_to_ipod | 8.280376604 |

**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).

## Реализация и тесты

- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)
- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)
- [Все тесты соответствующих серий](../EXPERIMENTS.md)

Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.
