# Luma ChromaSeed AS / HR · soft_small

[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)

**15,246 параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.

## Полная спецификация слоёв

| Слой | Вход → выход | Bias | Параметры |
|---|---|---|---:|
| `token1` | 18 → 32 | да | 608 |
| `token2` | 32 → 24 | да | 792 |
| `context` | 60 → 48 | да | 2,928 |
| `query` | 51 → 24 | да | 1,248 |
| `key` | 24 → 24 | нет | 576 |
| `update1` | 123 → 48 | да | 5,952 |
| `update2` | 48 → 48 | да | 2,352 |
| `head` | 48 → 3 | да | 147 |
| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |
| **Всего** | | | **15,246** |

## Прямой проход

Два token-слоя с SiLU, context из concat(color36,mean(tokens)). Keys без bias вычисляются один раз. Четыре прохода общими весами: query(state,pred) → scores/√e → attention → pooled tokens → update MLP; state=.5state+.5tanh(update), pred+=.25head(state).

Soft: attention использует все токены. Общие между проходами веса не умножают параметрический размер на четыре.

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
| mixed | 512 | 0.0001 | 5.694078723 | 156.14999999999998 |
| slr_to_ipod | 128 | 1e-05 | 8.295824318 | 156.9 |
| ipod_to_slr | 512 | 0.0001 | 8.536610626 | 244.85000000000002 |

Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.

## HR: все head / role результаты

| Head | Роль | Средняя ΔE00 |
|---|---|---:|
| linear | ipod_to_slr | 8.536318078 |
| unit | ipod_to_slr | 8.536610626 |
| wide | ipod_to_slr | 8.536338567 |
| linear | mixed | 5.694202531 |
| unit | mixed | 5.694078723 |
| wide | mixed | 5.694194681 |
| linear | slr_to_ipod | 8.295824325 |
| unit | slr_to_ipod | 8.295824318 |
| wide | slr_to_ipod | 8.295824325 |

**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).

## Реализация и тесты

- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)
- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)
- [Все тесты соответствующих серий](../EXPERIMENTS.md)

Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.
