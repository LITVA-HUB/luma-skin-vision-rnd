# `tests/test_skin_face_transfer_evaluate.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_skin_face_transfer_evaluate.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

Модуль исследовательского архива; назначение уточняется по определениям и связанному протоколу.

SHA-256 исходника: `de3f481be08a28012ac51f073877250934918b60de3316449e7affc84f5a7d19`. Строк: **143**.

## Зависимости

```python
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_confusion_arithmetic_and_empty_mask_conventions` | FunctionDef | См. реализацию | [L12](../../../../tests/test_skin_face_transfer_evaluate.py#L12) |
| `test_actual_inference_preserves_gapped_order_masks_and_color_coverage` | FunctionDef | См. реализацию | [L27](../../../../tests/test_skin_face_transfer_evaluate.py#L27) |
| `test_selection_recheck_rejects_late_choice_or_changed_history` | FunctionDef | См. реализацию | [L53](../../../../tests/test_skin_face_transfer_evaluate.py#L53) |
| `test_pairwise_color_uses_only_joint_coverage_and_reports_losses` | FunctionDef | См. реализацию | [L73](../../../../tests/test_skin_face_transfer_evaluate.py#L73) |
| `test_decoded_mask_metrics_detect_changed_saved_pixels` | FunctionDef | См. реализацию | [L86](../../../../tests/test_skin_face_transfer_evaluate.py#L86) |
| `test_complete_training_trace_rejects_held_indices_and_wrong_schedule` | FunctionDef | См. реализацию | [L99](../../../../tests/test_skin_face_transfer_evaluate.py#L99) |
| `test_comparison_reports_every_seed_budget_and_negative_data_effect` | FunctionDef | См. реализацию | [L122](../../../../tests/test_skin_face_transfer_evaluate.py#L122) |

## Все тестовые определения (7)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_confusion_arithmetic_and_empty_mask_conventions` · L12

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_confusion_arithmetic_and_empty_mask_conventions():
    from skin_face_transfer_evaluate import mask_summary
    counts = np.array([[1,1,2,252], [0,0,0,256]], dtype=np.int64)
    result = mask_summary(counts)
    assert result['images'] == 2
    assert result['global_iou'] == .25 and result['global_dice'] == .4
    assert result['mean_image_iou'] == .625 and result['fraction_image_iou_below_0_5'] == .5
    assert result['mean_image_precision'] == .75
    assert result['mean_image_recall'] == pytest.approx(2/3)
    with pytest.raises(ValueError):
        mask_summary(np.array([[1,0,-1,2]]))
    with pytest.raises(ValueError):
        mask_summary(counts.astype(float))
```

</details>

### `test_actual_inference_preserves_gapped_order_masks_and_color_coverage` · L27

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_actual_inference_preserves_gapped_order_masks_and_color_coverage():
    from skin_face_transfer_data import ArraySplit
    from skin_face_transfer_evaluate import evaluate_split, unpack_masks
    class Red(torch.nn.Module):
        def forward(self,x):
            return x[:,0:1]*2-1
    rgb = np.zeros((3,16,16,3), np.uint8)
    labels = np.zeros((3,16,16), np.uint8)
    rgb[2,:4,:4] = [255,128,100]
    labels[2,:4,:4] = 1
    labels[0,:4,:4] = 1
    split = ArraySplit('celeba','validation',rgb,labels,np.array([2,0]),'binary')
    result, arrays = evaluate_split(Red(),split,2,'cpu',include_color=True)
    assert arrays['global_ids'].tolist() == [2,0]
    np.testing.assert_array_equal(arrays['confusion'], [[16,0,0,240], [0,0,16,240]])
    masks = unpack_masks(arrays)
    np.testing.assert_array_equal(masks[0],labels[2].astype(bool))
    assert not masks[1].any()
    assert result['color']['valid_references'] == 2 and result['color']['valid_predictions'] == 1
    assert result['color']['available_pairs'] == 1 and result['color']['median_delta_e00_mean'] == 0
    assert np.isnan(arrays['color_error'][1]).all()
    split.role = 'train'
    with pytest.raises(ValueError, match='validation|test'):
        evaluate_split(Red(),split,2,'cpu',include_color=False)
```

</details>

### `test_selection_recheck_rejects_late_choice_or_changed_history` · L53

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_recheck_rejects_late_choice_or_changed_history():
    from skin_face_transfer_evaluate import validate_selection
    from skin_face_transfer_study import freeze_choices, recipe
    spec = recipe()
    histories = {r['id']:[dict(step=s,checkpoint={'path':str(s),'sha256':f'{s:064x}'},validation={
        'lapa':{'images':1692,'mean_image_iou':.8},
        'celeba':{'images':2992,'mean_image_iou':.7}}) for s in spec['validation_steps']]
        for r in spec['trajectories']}
    selected = freeze_choices(histories)
    validate_selection(selected,histories)
    altered = copy.deepcopy(selected)
    altered['overall']['step'] = 5976
    with pytest.raises(ValueError,match='selection'):
        validate_selection(altered,histories)
    altered = copy.deepcopy(histories)
    altered[spec['trajectories'][0]['id']][1]['validation']['celeba']['mean_image_iou'] = .99
    with pytest.raises(ValueError,match='selection'):
        validate_selection(selected,altered)
```

</details>

### `test_pairwise_color_uses_only_joint_coverage_and_reports_losses` · L73

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_pairwise_color_uses_only_joint_coverage_and_reports_losses():
    from skin_face_transfer_report import paired_color
    a = {'global_ids':np.array([1,2,3]), 'color_error':np.array([[1.,2.],[9.,8.],[np.nan,np.nan]])}
    b = {'global_ids':np.array([1,2,3]), 'color_error':np.array([[3.,4.],[np.nan,np.nan],[5.,6.]])}
    result = paired_color(a,b)
    assert result['common_images'] == 1
    assert result['left_available'] == result['right_available'] == 2
    assert result['right_minus_left_median_delta_e00'] == 2
    b['global_ids'] = np.array([3,2,1])
    with pytest.raises(ValueError,match='order'):
        paired_color(a,b)
```

</details>

### `test_decoded_mask_metrics_detect_changed_saved_pixels` · L86

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_decoded_mask_metrics_detect_changed_saved_pixels():
    from skin_face_transfer_evaluate import confusion_from_masks, verify_mask_arrays
    masks = np.zeros((2,16,16),bool)
    masks[0,2:4,2:4] = True
    truth = masks.copy()
    arrays = {'global_ids':np.array([0,1]), 'confusion':confusion_from_masks(masks,truth),
              'mask_shape':np.array([16,16]), 'packed_masks':np.packbits(masks.reshape(2,-1),axis=1,bitorder='little')}
    verify_mask_arrays(arrays,truth)
    arrays['packed_masks'][0,0] ^= 1
    with pytest.raises(ValueError,match='confusion'):
        verify_mask_arrays(arrays,truth)
```

</details>

### `test_complete_training_trace_rejects_held_indices_and_wrong_schedule` · L99

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_complete_training_trace_rejects_held_indices_and_wrong_schedule():
    import math

    from skin_face_transfer_data import PairedSampler
    from skin_face_transfer_report import validate_trace
    from skin_face_transfer_study import BUDGETS
    lapa,celeba = np.array([2,7,9]),np.array([100,110])
    sampler = PairedSampler(lapa,celeba,17,4)
    samples = [sampler.next('lapa_celeba') for _ in range(4)]
    rates = [.0001*(.1+.9*.5*(1+math.cos(math.pi*s/(BUDGETS[-1]-1)))) for s in range(4)]
    trace = dict(global_ids=np.stack([s[1] for s in samples]),source_ids=np.stack([s[0] for s in samples]),
                 loss_gradient_lr=np.array([[.1,2.,lr] for lr in rates]))
    assert validate_trace(trace,'lapa_celeba',17,lapa,celeba,steps=4,batch_size=4) == 0
    wrong = copy.deepcopy(trace)
    wrong['global_ids'][2,3] = 101
    with pytest.raises(ValueError,match='TRAIN'):
        validate_trace(wrong,'lapa_celeba',17,lapa,celeba,steps=4,batch_size=4)
    wrong = copy.deepcopy(trace)
    wrong['loss_gradient_lr'][0,2] *= 2
    with pytest.raises(ValueError,match='schedule'):
        validate_trace(wrong,'lapa_celeba',17,lapa,celeba,steps=4,batch_size=4)
```

</details>

### `test_comparison_reports_every_seed_budget_and_negative_data_effect` · L122

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_comparison_reports_every_seed_budget_and_negative_data_effect(tmp_path):
    from skin_face_transfer_evaluate import mask_summary
    from skin_face_transfer_report import comparisons
    from skin_face_transfer_run import save_npz
    from skin_face_transfer_study import ARMS, BUDGETS, INITIAL_SHA, SEEDS
    choices = [dict(arm=arm,seed=seed,budget=budget,step=498,selection_score=.8,
                    checkpoint={'path':str(tmp_path/(arm+'.pt')), 'sha256':('d' if arm == ARMS[0] else 'e')*64})
               for arm in ARMS for seed in SEEDS for budget in BUDGETS]
    selection = {'choices':choices,'overall':choices[0]}
    records = []
    for sha,tp in [(INITIAL_SHA,9),('d'*64,8),('e'*64,7)]:
        for source in ('lapa','celeba'):
            counts = np.array([[tp,0,10-tp,246]],dtype=np.int64)
            ref = save_npz(tmp_path/f'{sha}_{source}.npz',global_ids=np.array([1]),confusion=counts,color_error=np.array([[1.,2.]]))
            records.append(dict(checkpoint_sha256=sha,source=source,arrays=ref,metrics=mask_summary(counts)))
    result = comparisons(selection,{'records':records})
    assert len(result['all_choices']) == 18 and len(result['data_contrasts']) == 9
    assert len(result['longer_training_contrasts']) == 12
    assert all(r['macro_iou_points'] == pytest.approx(-10) for r in result['data_contrasts'])
    assert all(r['macro_iou_points'] == 0 for r in result['longer_training_contrasts'])
    with pytest.raises(ValueError,match='coverage'):
        comparisons(selection,{'records':records[:-1]})
```

</details>
