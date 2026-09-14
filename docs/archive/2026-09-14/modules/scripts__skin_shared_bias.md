# `scripts/skin_shared_bias.py`

[Архив](../README.md) · [Индекс](../SOURCE_INDEX.md) · [Полный исходник](../../../../scripts/skin_shared_bias.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Paired mean-bias objectives with a fixed single-image inference model.

SHA-256 исходника: `4c8da866175820a1ebb5771fa707f92381efb097277f33b4ca8e2b509a774561`. Строк: **24**.

## Зависимости

```python
import torch
```

## Зафиксированные конфигурации и константы

Вычисляемые значения сохранены как выражения исходника; это не результат выполнения.

[Строка 4](../../../../scripts/skin_shared_bias.py#L4)

```python
ARMS = ['individual', 'consistency', 'shared_half', 'shared_only']
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `paired_objective` | FunctionDef | См. реализацию | [L7](../../../../scripts/skin_shared_bias.py#L7) |
