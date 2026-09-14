# `tests/test_cc_v3_ffcc.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_cc_v3_ffcc.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

SYNTHETIC formula fixtures for an FFCC-inspired control, no benchmark evidence.

SHA-256 исходника: `cff65664008f6d3ab3648bfaa45fe468ad148c3151ea57eb201bf09e66d565ec`. Строк: **211**.

## Зависимости

```python
import importlib.util
import math
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `core` | FunctionDef | См. реализацию | [L12](../../../../tests/test_cc_v3_ffcc.py#L12) |
| `cpu_threads` | FunctionDef | См. реализацию | [L22](../../../../tests/test_cc_v3_ffcc.py#L22) |
| `test_matlab_round_and_periodic_unit_histogram_axes_half_bins_and_wrap` | FunctionDef | См. реализацию | [L29](../../../../tests/test_cc_v3_ffcc.py#L29) |
| `test_local_deviation_uses_all_eight_neighbors_replicated_edges_and_mask_products` | FunctionDef | См. реализацию | [L41](../../../../tests/test_cc_v3_ffcc.py#L41) |
| `test_two_feature_channels_normalize_independently_and_empty_edges_stay_zero` | FunctionDef | См. реализацию | [L55](../../../../tests/test_cc_v3_ffcc.py#L55) |
| `test_average_linear_rgb_preserves_source_all_pixel_policy_and_rejects_invalid_float_input` | FunctionDef | См. реализацию | [L71](../../../../tests/test_cc_v3_ffcc.py#L71) |
| `test_fft_filter_is_circular_convolution_not_correlation_and_softmax_is_stable` | FunctionDef | См. реализацию | [L82](../../../../tests/test_cc_v3_ffcc.py#L82) |
| `test_wrapped_moment_decode_covariance_padding_units_and_rgb_sign` | FunctionDef | См. реализацию | [L101](../../../../tests/test_cc_v3_ffcc.py#L101) |
| `test_gray_world_dealias_uses_linear_rgb_anchor_and_matlab_half_ties` | FunctionDef | См. реализацию | [L116](../../../../tests/test_cc_v3_ffcc.py#L116) |
| `test_uniform_posterior_reports_undefined_mean_and_zero_confidence_with_finite_diagnostics` | FunctionDef | См. реализацию | [L127](../../../../tests/test_cc_v3_ffcc.py#L127) |
| `test_both_target_branches_and_crossentropy_match_literal_wrapped_weights` | FunctionDef | См. реализацию | [L136](../../../../tests/test_cc_v3_ffcc.py#L136) |
| `test_gaussian_nll_shift_and_nonwrapped_residual_are_exact` | FunctionDef | См. реализацию | [L153](../../../../tests/test_cc_v3_ffcc.py#L153) |
| `test_fft_quadratic_regularizer_dc_parseval_shift_and_data_mass` | FunctionDef | См. реализацию | [L166](../../../../tests/test_cc_v3_ffcc.py#L166) |
| `test_full_default_model_capacity_zero_init_ce_backward_and_empty_input_refusal` | FunctionDef | См. реализацию | [L180](../../../../tests/test_cc_v3_ffcc.py#L180) |
| `test_objective_gradcheck_through_fft_moments_covariance_and_regularizer` | FunctionDef | См. реализацию | [L198](../../../../tests/test_cc_v3_ffcc.py#L198) |

## Все тестовые определения (13)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_matlab_round_and_periodic_unit_histogram_axes_half_bins_and_wrap` · L29

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_matlab_round_and_periodic_unit_histogram_axes_half_bins_and_wrap():
    m = core()
    values = torch.tensor([-2.5, -1.5, -.5, .5, 1.5, 2.5], dtype=torch.float64)
    torch.testing.assert_close(m.round_matlab(values), torch.tensor([-3., -2., -1., 1., 2., 3.], dtype=values.dtype))
    uv = torch.tensor([[[-.5, 1.5], [3.5, -1.5], [4.5, 2.5], [100., 100.]]], dtype=values.dtype)
    hist, count = m.periodic_histogram(uv, torch.tensor([[True, True, True, False]]), n=4, h=1., lo=0.)
    expected = torch.zeros(1, 4, 4, dtype=values.dtype)
    expected[0, 3, 2] = expected[0, 0, 2] = expected[0, 1, 3] = 1 / 3
    torch.testing.assert_close(hist, expected)
    assert count.tolist() == [3]
```

</details>

### `test_local_deviation_uses_all_eight_neighbors_replicated_edges_and_mask_products` · L41

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_local_deviation_uses_all_eight_neighbors_replicated_edges_and_mask_products():
    m = core()
    rgb = torch.tensor([[[[0., .25], [.5, 1.]]]], dtype=torch.float64).expand(-1, 3, -1, -1)
    actual = m.masked_local_absolute_deviation(rgb, torch.ones(1, 2, 2, dtype=torch.bool))
    expected = torch.tensor([[[[10/32, 9/32], [9/32, 14/32]]]], dtype=rgb.dtype).expand_as(rgb)
    torch.testing.assert_close(actual, expected)
    mask = torch.tensor([[[True, True], [True, False]]])
    actual = m.masked_local_absolute_deviation(rgb, mask)
    torch.testing.assert_close(actual[0, :, 0, 0], torch.full((3,), 6/28, dtype=rgb.dtype))
    torch.testing.assert_close(actual[0, :, 0, 1], torch.full((3,), 3/24, dtype=rgb.dtype))
    torch.testing.assert_close(actual[0, :, 1, 0], torch.full((3,), 5/24, dtype=rgb.dtype))
    assert torch.isnan(actual[0, :, 1, 1]).all()
```

</details>

### `test_two_feature_channels_normalize_independently_and_empty_edges_stay_zero` · L55

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_two_feature_channels_normalize_independently_and_empty_edges_stay_zero():
    m = core()
    rgb = torch.tensor([.25, .5, .125], dtype=torch.float64)[None, :, None, None].expand(1, 3, 3, 4)
    output = m.featurize(rgb)
    assert output["feature_counts"].tolist() == [[12, 0]]
    torch.testing.assert_close(output["hist"].sum((-2, -1)), torch.tensor([[1., 0.]], dtype=rgb.dtype))
    torch.testing.assert_close(output["average_rgb"], F.normalize(torch.tensor([[.25, .5, .125]], dtype=rgb.dtype), dim=-1))
    # Two different exposures cast equal votes, and all-channel minimum excludes
    # the third pixel even though its ratios are finite.
    pixels = torch.tensor([[[[.1, .2, .001]], [[.2, .2, .2]], [[.2, .2, .2]]]], dtype=rgb.dtype)
    output = m.featurize(pixels)
    assert output["feature_counts"][0, 0].item() == 2
    occupied = output["hist"][0, 0][output["hist"][0, 0] > 0]
    torch.testing.assert_close(occupied, torch.tensor([.5, .5], dtype=rgb.dtype))
```

</details>

### `test_average_linear_rgb_preserves_source_all_pixel_policy_and_rejects_invalid_float_input` · L71

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_average_linear_rgb_preserves_source_all_pixel_policy_and_rejects_invalid_float_input():
    m = core()
    rgb = torch.tensor([[[[.1, .9]], [[.2, .3]], [[.3, .2]]]], dtype=torch.float64)
    output = m.featurize(rgb, torch.tensor([[[True, False]]]))
    torch.testing.assert_close(output["average_rgb"], F.normalize(torch.tensor([[.5, .25, .25]], dtype=rgb.dtype), dim=-1))
    with pytest.raises(ValueError, match="finite"):
        m.featurize(rgb * float("nan"))
    with pytest.raises(ValueError, match="0, 1"):
        m.featurize(rgb * 2)
```

</details>

### `test_fft_filter_is_circular_convolution_not_correlation_and_softmax_is_stable` · L82

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fft_filter_is_circular_convolution_not_correlation_and_softmax_is_stable():
    m = core()
    hist = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
    filters = torch.zeros(2, 4, 4, dtype=hist.dtype)
    hist[0, 0, 1, 2] = 1
    filters[0, 2, 3] = 7
    hist[0, 1, 0, 0] = 1
    filters[1, 0, 1] = 2
    bias = torch.zeros(4, 4, dtype=hist.dtype)
    logits, pmf = m.circular_score(hist, filters, bias)
    expected = torch.zeros_like(logits)
    expected[0, 3, 1] = 7
    expected[0, 0, 1] = 2
    torch.testing.assert_close(logits, expected)
    _, shifted = m.circular_score(hist, filters, bias + 1e6)
    torch.testing.assert_close(shifted, pmf)
    torch.testing.assert_close(pmf.sum((-2, -1)), torch.ones(1, dtype=hist.dtype))
```

</details>

### `test_wrapped_moment_decode_covariance_padding_units_and_rgb_sign` · L101

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_wrapped_moment_decode_covariance_padding_units_and_rgb_sign():
    m = core()
    pmf = torch.zeros(1, 4, 4, dtype=torch.float64)
    pmf[0, 0, 1] = pmf[0, 3, 1] = .5
    output = m.decode(pmf, lo=-.5, h=.25, eps_bins=1.)
    torch.testing.assert_close(output["mu_index"], torch.tensor([[3.5, 1.]], dtype=pmf.dtype))
    torch.testing.assert_close(output["mu_uv"], torch.tensor([[.375, -.25]], dtype=pmf.dtype))
    expected_cov = torch.tensor([[[1.25, 0.], [0., 1.]]], dtype=pmf.dtype) * .25**2
    torch.testing.assert_close(output["covariance_uv"], expected_cov)
    expected_rgb = F.normalize(torch.tensor([[math.exp(-.375), 1., math.exp(.25)]], dtype=pmf.dtype), dim=-1)
    torch.testing.assert_close(output["pred"], expected_rgb)
    torch.testing.assert_close(output["confidence"], torch.tensor([1 / math.sqrt(1.25)], dtype=pmf.dtype))
    assert output["mean_valid"].all()
```

</details>

### `test_gray_world_dealias_uses_linear_rgb_anchor_and_matlab_half_ties` · L116

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gray_world_dealias_uses_linear_rgb_anchor_and_matlab_half_ties():
    m = core()
    pmf = torch.zeros(1, 4, 4, dtype=torch.float64)
    pmf[0, 0, 0] = 1
    average = torch.tensor([[math.exp(-1.), 1., math.exp(1.)]], dtype=pmf.dtype)
    output = m.decode(pmf, lo=0., h=.5, average_rgb=average, unwrap_mode="gray_world")
    # Period2, delta(-1,+1) lies exactly on the half-period; away-from-zero
    # chooses(+2,-2), unlike ties-to-even which would leave(0,0).
    torch.testing.assert_close(output["mu_uv"], torch.tensor([[2., -2.]], dtype=pmf.dtype))
```

</details>

### `test_uniform_posterior_reports_undefined_mean_and_zero_confidence_with_finite_diagnostics` · L127

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_uniform_posterior_reports_undefined_mean_and_zero_confidence_with_finite_diagnostics():
    m = core()
    output = m.decode(torch.full((2, 64, 64), 1/4096, dtype=torch.float64))
    assert (~output["mean_valid"]).all()
    assert (output["confidence"] == 0).all()
    for key in ["pred", "mu_uv", "covariance_uv"]:
        assert torch.isfinite(output[key]).all()
```

</details>

### `test_both_target_branches_and_crossentropy_match_literal_wrapped_weights` · L136

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_both_target_branches_and_crossentropy_match_literal_wrapped_weights():
    m = core()
    uv = torch.tensor([[-.25, 3.5]], dtype=torch.float64)
    gt = torch.tensor([[math.exp(.25), 1., math.exp(-3.5)]], dtype=uv.dtype)
    target = m.target_distribution(gt, n=4, h=1., lo=0., kind="bilinear")
    expected = torch.zeros(1, 4, 4, dtype=uv.dtype)
    expected[0, 3, 3] = expected[0, 3, 0] = .125
    expected[0, 0, 3] = expected[0, 0, 0] = .375
    torch.testing.assert_close(target, expected)
    nearest = m.target_distribution(gt, n=4, h=1., lo=0., kind="nearest")
    assert nearest[0, 0, 0].item() == 1
    logits = torch.arange(16, dtype=uv.dtype).reshape(1, 4, 4) / 8
    ce = m.cross_entropy(logits, gt, h=1., lo=0., kind="bilinear")
    oracle = -(expected * logits.flatten(1).log_softmax(-1).reshape_as(logits)).sum()
    torch.testing.assert_close(ce, oracle)
```

</details>

### `test_gaussian_nll_shift_and_nonwrapped_residual_are_exact` · L153

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_gaussian_nll_shift_and_nonwrapped_residual_are_exact():
    m = core()
    output = {"mu_uv": torch.tensor([[0., 0.]], dtype=torch.float64), "covariance_uv": torch.eye(2, dtype=torch.float64)[None] * .25, "mean_valid": torch.tensor([True])}
    gt = torch.tensor([[math.exp(-1.), 1., 1.]], dtype=torch.float64)
    torch.testing.assert_close(m.gaussian_uv_nll(output, gt, h=.5, eps_bins=1.), torch.tensor(2., dtype=gt.dtype))
    larger = dict(output, covariance_uv=output["covariance_uv"] * 2)
    expected = 1 + math.log(2)
    torch.testing.assert_close(m.gaussian_uv_nll(larger, gt, h=.5), torch.tensor(expected, dtype=gt.dtype))
    invalid = dict(output, mean_valid=torch.tensor([False]))
    with pytest.raises(ValueError, match="defined"):
        m.gaussian_uv_nll(invalid, gt)
```

</details>

### `test_fft_quadratic_regularizer_dc_parseval_shift_and_data_mass` · L166

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_fft_quadratic_regularizer_dc_parseval_shift_and_data_mass():
    m = core()
    filters = torch.ones(2, 4, 4, dtype=torch.float64)
    bias = torch.ones(4, 4, dtype=filters.dtype)
    # Only DC is nonzero: three maps * |FFT(ones)[0,0]|²=3*256.
    expected = .5 * (2**-8) * 3 * 256
    torch.testing.assert_close(m.fft_regularizer(filters, bias), torch.tensor(expected, dtype=filters.dtype))
    torch.testing.assert_close(m.fft_regularizer(filters, bias, data_mass=3), torch.tensor(expected * 3, dtype=filters.dtype))
    # A single unit spatial impulse has unit power in all bins; sum A=n²/2.
    filters = torch.zeros_like(filters)
    filters[0, 0, 0] = 1
    torch.testing.assert_close(m.fft_regularizer(filters, bias * 0), torch.tensor(.5 * (4**-4 * 8 + 2**-8 * 16), dtype=filters.dtype))
```

</details>

### `test_full_default_model_capacity_zero_init_ce_backward_and_empty_input_refusal` · L180

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_full_default_model_capacity_zero_init_ce_backward_and_empty_input_refusal():
    m = core()
    model = m.FFCCInspiredNet()
    assert sum(p.numel() for p in model.parameters()) == 12288
    rgb = torch.rand(2, 3, 8, 8) * .8 + .1
    output = model(rgb)
    assert output["hist"].shape == (2, 2, 64, 64)
    assert (~output["mean_valid"]).all()
    loss = m.cross_entropy(output["logits"], torch.tensor([[.8, 1., .9], [1., .7, .8]])) + model.regularizer()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.filters.grad.abs().sum() > 0
    assert model.bias.grad.abs().sum() > 0
    empty = model(torch.zeros_like(rgb))
    assert (~empty["valid"]).all()
    assert (empty["pred"] > 0).all()
```

</details>

### `test_objective_gradcheck_through_fft_moments_covariance_and_regularizer` · L198

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_objective_gradcheck_through_fft_moments_covariance_and_regularizer():
    m = core()
    n = 4
    hist = torch.zeros(1, 2, n, n, dtype=torch.float64)
    hist[0, 0, 0, 0] = 1
    hist[0, 1, 1, 1] = 1
    filters = (torch.arange(2*n*n, dtype=torch.float64).reshape(2, n, n) * .002).requires_grad_()
    bias = torch.tensor([[0., .1, 0., 0.], [.1, 2., 1., 0.], [0., .8, .3, 0.], [0., 0., 0., 0.]], dtype=torch.float64, requires_grad=True)
    gt = torch.tensor([[.8, 1., .9]], dtype=torch.float64)
    def objective(filt, prior):
        logits, pmf = m.circular_score(hist, filt, prior)
        output = m.decode(pmf, lo=-.5, h=.25)
        return m.cross_entropy(logits, gt, lo=-.5, h=.25) + m.gaussian_uv_nll(output, gt, h=.25) + m.fft_regularizer(filt, prior)
    assert torch.autograd.gradcheck(objective, (filters, bias), eps=1e-6, atol=2e-5, rtol=2e-4)
```

</details>
