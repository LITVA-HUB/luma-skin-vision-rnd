# Luma ChromaSeed AS / HR · soft5m

[Архитектуры](../ARCHITECTURES.md) · [AS-серия](../experiments/chromaseed_architecture_scale_v1.md) · [HR: все результаты](../HR_ALL_MODELS.md)

**4,846,822 параметров**, включая frozen NP-якорь 643. Семейство residual-регрессии нативного Lab; вход color36 +64×18 токенов подготовленного участка. Это не самостоятельный face detector или система подбора косметики.

## Полная спецификация слоёв

| Слой | Вход → выход | Bias | Параметры |
|---|---|---|---:|
| `token1` | 18 → 384 | да | 7,296 |
| `token2` | 384 → 256 | да | 98,560 |
| `context` | 292 → 1120 | да | 328,160 |
| `query` | 1123 → 256 | да | 287,744 |
| `key` | 256 → 256 | нет | 65,536 |
| `update1` | 2499 → 1120 | да | 2,800,000 |
| `update2` | 1120 → 1120 | да | 1,255,520 |
| `head` | 1120 → 3 | да | 3,363 |
| Frozen NP anchor | color36 → Lab3 | В составе родителя | 643 |
| **Всего** | | | **4,846,822** |

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
| mixed | 512 | 1e-05 | 5.662450592 | 3094.9 |
| slr_to_ipod | 2048 | 1e-05 | 7.976812830 | 3182.8 |
| ipod_to_slr | 2048 | 1e-05 | 8.811416097 | 2984.55 |

Это среднее ошибок отдельных seeds на повторно используемых TRAIN-ролях, не независимая точность селфи. CPU время относится к готовым признакам; декодирование и сегментация исключены. Общий победитель определяется отдельным INNER-выбором, не минимумом внешних чисел этой карточки.

## HR: все head / role результаты

| Head | Роль | Средняя ΔE00 |
|---|---|---:|
| linear | ipod_to_slr | 8.887051967 |
| unit | ipod_to_slr | 8.811416097 |
| wide | ipod_to_slr | 8.874082882 |
| linear | mixed | 5.662636668 |
| unit | mixed | 5.662450592 |
| wide | mixed | 5.662624985 |
| linear | slr_to_ipod | 8.282167895 |
| unit | slr_to_ipod | 7.976812830 |
| wide | slr_to_ipod | 8.282128291 |

**Статус HR:** аудит качества принят; runtime оборван, финального seal нет. Все seeds, хэши и выбранные checkpoints доступны в [HR records](../evidence/chromaseed_head_range_v1/results.json).

## Реализация и тесты

- [AS: точные конструкторы, forward, fit и consumer](../modules/scripts__chromaseed_architecture_scale.md)
- [HR: unit/wide/linear](../modules/scripts__chromaseed_head_range.md)
- [Все тесты соответствующих серий](../EXPERIMENTS.md)

Таблица размерностей — документальная расшифровка specs/capacity; при сборке архива модель не создаётся и не исполняется.
