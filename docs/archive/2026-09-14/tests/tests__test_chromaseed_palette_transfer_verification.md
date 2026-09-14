# `tests/test_chromaseed_palette_transfer_verification.py`

[Архив](../README.md) · [Индекс](../TESTS.md) · [Полный исходник](../../../../tests/test_chromaseed_palette_transfer_verification.py)

> Статическая документация исходника. Наличие теста или функции не означает, что её запуск завершён успешно.

P3 verifier must detect intervention, selection and reporting mistakes.

SHA-256 исходника: `268fc7625f21c0d22c2d93371b9676463760dd3cf9d192fe2b0cbaf80b5e1a57`. Строк: **288**.

## Зависимости

```python
import sys
from pathlib import Path
import numpy as np
import pytest
```


## Определения верхнего уровня

| Имя | Вид | Назначение из docstring | Исходник |
|---|---|---|---|
| `test_independent_initialization_covers_every_weight_and_arm` | FunctionDef | См. реализацию | [L13](../../../../tests/test_chromaseed_palette_transfer_verification.py#L13) |
| `test_encoder_rejects_mismatched_seed_arm_and_corrupted_scales` | FunctionDef | См. реализацию | [L36](../../../../tests/test_chromaseed_palette_transfer_verification.py#L36) |
| `test_head_rule_checks_inner_choice_and_not_external_scores` | FunctionDef | См. реализацию | [L50](../../../../tests/test_chromaseed_palette_transfer_verification.py#L50) |
| `test_selection_preserves_stable_arms_and_control_winner` | FunctionDef | См. реализацию | [L80](../../../../tests/test_chromaseed_palette_transfer_verification.py#L80) |
| `test_export_check_accepts_unit_payload_and_catches_false_provenance` | FunctionDef | См. реализацию | [L108](../../../../tests/test_chromaseed_palette_transfer_verification.py#L108) |
| `test_all_pass_check_detects_corruption_with_unchanged_final_prediction` | FunctionDef | См. реализацию | [L128](../../../../tests/test_chromaseed_palette_transfer_verification.py#L128) |
| `test_original_payload_unwrap_and_full_bank_costs` | FunctionDef | См. реализацию | [L154](../../../../tests/test_chromaseed_palette_transfer_verification.py#L154) |
| `test_report_keeps_negative_palette_contrasts_and_never_selects_on_external` | FunctionDef | См. реализацию | [L174](../../../../tests/test_chromaseed_palette_transfer_verification.py#L174) |
| `test_final_auditor_checks_real_payloads_paths_metrics_and_every_pass` | FunctionDef | См. реализацию | [L191](../../../../tests/test_chromaseed_palette_transfer_verification.py#L191) |

## Все тестовые определения (9)

Параметризация показана дословно. Число определений не равно числу развёрнутых pytest cases; фикстуры и окружение влияют на сборку. Эти тесты не запускались при подготовке атласа.

### `test_independent_initialization_covers_every_weight_and_arm` · L13

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('variant', ['patch5m', 'soft5m', 'dynamic5m'])
def test_independent_initialization_covers_every_weight_and_arm(variant):
    import torch
    from chromaseed_palette_transfer import Bank
    from chromaseed_palette_transfer_verification import (
        encoder_content,
        encoders_for,
        initial_theta,
    )

    torch.set_num_threads(1)
    mean = np.linspace(-0.2, 0.3, 18).astype(np.float32)
    std = np.linspace(0.4, 1.2, 18).astype(np.float32)
    for arm in ("original", "aligned", "shuffled"):
        encoders = encoders_for(arm, {})
        expected = initial_theta(variant, arm, encoders, mean, std)
        net = Bank(variant, "linear", arm, encoders, mean, std)
        np.testing.assert_array_equal(expected, net.theta.detach().numpy())
        if encoders:
            assert net.encoder_digests == [encoder_content(e) for e in encoders]
        del net, expected
    assert not torch.cuda.is_initialized()
```

</details>

### `test_encoder_rejects_mismatched_seed_arm_and_corrupted_scales` · L36

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_encoder_rejects_mismatched_seed_arm_and_corrupted_scales():
    from chromaseed_palette_transfer_verification import encoders_for, validate_encoder

    encoder = encoders_for("aligned", {})[0]
    validate_encoder(encoder, 17, "aligned")
    for seed, arm in [(29, "aligned"), (17, "shuffled")]:
        with pytest.raises(AssertionError):
            validate_encoder(encoder, seed, arm)
    invalid = {k: v.copy() for k, v in encoder.items()}
    invalid["std"][0] = 0
    with pytest.raises(AssertionError):
        validate_encoder(invalid, 17, "aligned")
```

</details>

### `test_head_rule_checks_inner_choice_and_not_external_scores` · L50

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_head_rule_checks_inner_choice_and_not_external_scores():
    from chromaseed_palette_transfer_verification import PARAMETERS, ROLES, selected_heads

    selection = {"roles": {}}
    for role in ROLES:
        candidates = [
            dict(
                variant=v + "__" + mode,
                architecture=v,
                head_mode=mode,
                clean=1 if mode == "linear" else 2,
                p90=2,
                numeric_bytes=100,
                step=128,
                lr=1e-5,
            )
            for v in PARAMETERS
            for mode in ["unit", "wide", "linear"]
        ]
        per = {v: next(c for c in candidates if c["variant"] == v + "__linear") for v in PARAMETERS}
        selection["roles"][role] = dict(candidates=candidates, policies=dict(per_architecture=per))
    heads = selected_heads(selection)
    assert all(mode == "linear" for role in heads.values() for mode in role.values())
    selection["roles"]["mixed"]["policies"]["per_architecture"]["patch5m"] = selection["roles"][
        "mixed"
    ]["candidates"][0]
    with pytest.raises(AssertionError):
        selected_heads(selection)
```

</details>

### `test_selection_preserves_stable_arms_and_control_winner` · L80

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_selection_preserves_stable_arms_and_control_winner():
    from chromaseed_palette_transfer_verification import ARMS, PARAMETERS, select_policies

    candidates = [
        dict(
            variant=v + "__" + a,
            architecture=v,
            initialization=a,
            clean=1,
            p90=2,
            numeric_bytes=100,
            step=128,
            lr=1e-5,
        )
        for v in PARAMETERS
        for a in ARMS
    ]
    chosen = select_policies(candidates)
    assert len(chosen["per_pair"]) == 9 and len(chosen["per_architecture"]) == 3
    assert chosen["overall"]["variant"] == "patch5m__original"
    candidates.append(dict(variant="np", clean=0.5, p90=1, numeric_bytes=10, step=0, lr=None))
    assert select_policies(candidates)["overall"]["variant"] == "np"
    candidates[1]["clean"] = float("nan")
    with pytest.raises(ValueError):
        select_policies(candidates)
```

</details>

### `test_export_check_accepts_unit_payload_and_catches_false_provenance` · L108

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
@pytest.mark.parametrize('mode', ['unit', 'wide', 'linear'])
def test_export_check_accepts_unit_payload_and_catches_false_provenance(mode):
    from chromaseed_palette_transfer import Bank
    from chromaseed_palette_transfer_audit import check_model
    from chromaseed_palette_transfer_verification import encoders_for
    from test_chromaseed_architecture_scale import fixture

    _, tokens, _, warm = fixture()
    mean, std = (
        tokens.astype(float).mean((0, 1)).astype(np.float32),
        tokens.astype(float).std((0, 1)).astype(np.float32),
    )
    encoders = encoders_for("aligned", {})
    net = Bank("soft5m", mode, "aligned", encoders, mean, std)
    model = net.export(0, warm, tokens)
    check_model(model, "soft5m", mode, "aligned", encoders[0], warm[0], mean, std)
    model["palette_encoder_digest"] = np.asarray("incorrect")
    with pytest.raises(AssertionError):
        check_model(model, "soft5m", mode, "aligned", encoders[0], warm[0], mean, std)
```

</details>

### `test_all_pass_check_detects_corruption_with_unchanged_final_prediction` · L128

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_all_pass_check_detects_corruption_with_unchanged_final_prediction():
    import torch
    from chromaseed_head_range import predict_torch
    from chromaseed_palette_transfer import Bank
    from chromaseed_palette_transfer_audit import verify_passes
    from chromaseed_palette_transfer_verification import encoders_for
    from test_chromaseed_architecture_scale import fixture

    x, t, _, warm = fixture()
    mean, std = (
        t.astype(float).mean((0, 1)).astype(np.float32),
        t.astype(float).std((0, 1)).astype(np.float32),
    )
    net = Bank("dynamic5m", "linear", "aligned", encoders_for("aligned", {}), mean, std)
    with torch.no_grad():
        net.layers["head"].weight.fill_(0.02)
        net.layers["head"].bias.fill_(0.01)
    model = net.export(0, warm, t)
    saved = predict_torch(model, x[:2], t[:2], device="cpu", all_passes=True)
    maximum, calls = verify_passes(model, x[:2], t[:2], saved)
    assert maximum < 0.002 and calls == 2
    saved[0, 0, 0] += 0.01
    with pytest.raises(AssertionError):
        verify_passes(model, x[:2], t[:2], saved)
```

</details>

### `test_original_payload_unwrap_and_full_bank_costs` · L154

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_original_payload_unwrap_and_full_bank_costs():
    from chromaseed_palette_transfer_verification import native_costs, payload

    original = dict(model="actual.npz", output="output.npz")
    assert payload(dict(inherited_hr_record=dict(inherited_as_record=original))) == original
    banks = [
        dict(
            steps=128,
            setup_seconds=1.0,
            full_bank_seconds=10.0,
            write_and_prediction_inclusive_seconds=12.0,
            cuda_peak_allocated_bytes=100,
        )
    ]
    value = native_costs(banks, 15.0)
    assert value["native_bank_seconds"] == 10 and value["native_write_inclusive_seconds"] == 12
    assert value["native_trajectories"] == 6 and value["presentations"] == 128 * 64 * 6
    assert value["complete_end_to_end_seconds"] is None
```

</details>

### `test_report_keeps_negative_palette_contrasts_and_never_selects_on_external` · L174

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_report_keeps_negative_palette_contrasts_and_never_selects_on_external():
    from chromaseed_palette_transfer_report import comparisons
    from chromaseed_palette_transfer_verification import ARMS, PARAMETERS, ROLES

    rows = [
        dict(variant=v + "__" + arm, roles={r: dict(delta_e00=error) for r in ROLES})
        for v in PARAMETERS
        for arm, error in zip(ARMS, [2.0, 3.0, 4.0], strict=True)
    ]
    records = comparisons(rows)
    assert len(records) == 9
    assert all(
        r["aligned_minus_original"] == 1.0 and r["aligned_minus_shuffled"] == -1.0 for r in records
    )
    assert not any(r["better_than_both"] for r in records)
```

</details>

### `test_final_auditor_checks_real_payloads_paths_metrics_and_every_pass` · L191

Проверяемые условия перечислены в теле теста ниже.

<details><summary>Условия, параметризация и полное тело теста</summary>

```python
def test_final_auditor_checks_real_payloads_paths_metrics_and_every_pass(tmp_path, monkeypatch):
    import chromaseed_palette_transfer_audit as audit
    import torch
    from chromaseed_gated import flatten
    from chromaseed_head_range import predict_torch
    from chromaseed_palette_transfer import Bank
    from chromaseed_palette_transfer_verification import digest, encoders_for
    from test_chromaseed_architecture_scale import fixture

    x, tokens, y, warm = fixture()
    mean = tokens.astype(float).mean((0, 1)).astype(np.float32)
    std = tokens.astype(float).std((0, 1)).astype(np.float32)
    encoders = encoders_for("shuffled", {})
    net = Bank("dynamic5m", "unit", "shuffled", encoders, mean, std)
    with torch.no_grad():
        net.layers["head"].bias.fill_(0.01)
    models = [net.export(i, warm, tokens) for i in range(6)]
    del net
    bank = tmp_path / "bank"
    bank.mkdir()
    source = bank / "models_128.npz"
    np.savez(source, **flatten({str(i): m for i, m in enumerate(models)}))
    query = np.arange(2)
    data = dict(
        color=x,
        tokens=tokens,
        target=y,
        patient=np.asarray([f"p{i}" for i in range(len(x))]),
        site=np.zeros(len(x), int),
    )
    monkeypatch.setattr(audit, "bank_path", lambda *args: bank)
    monkeypatch.setattr(
        audit, "inspect", lambda *args: (np.arange(2, len(x)), query, warm, mean, std, encoders)
    )
    records, outputs = {}, []
    pair = "dynamic5m__shuffled"
    for si, seed in enumerate((17, 29, 43)):
        model_path, output_path = tmp_path / f"model{seed}.npz", tmp_path / f"out{seed}.npz"
        model = models[2 * si]
        np.savez(model_path, **model)
        passes = predict_torch(model, x[query], tokens[query], device="cpu", all_passes=True)
        saved = dict(row_indices=query, predictions=passes[:, -1].copy(), passes=passes)
        outputs.append(saved)
        np.savez(output_path, **saved)
        records["mixed", pair, seed] = dict(
            role="mixed",
            variant=pair,
            architecture="dynamic5m",
            head_mode="unit",
            initialization="shuffled",
            seed=seed,
            slot=2 * si,
            source_path=str(source),
            source_sha256=digest(source),
            step=128,
            lr=1e-5,
            overall=True,
            model=str(model_path),
            model_sha256=digest(model_path),
            output=str(output_path),
            output_sha256=digest(output_path),
            parameters=4846822,
            numeric_bytes=4 * 4846822 + 458,
            metrics=audit.error_summary(
                saved["predictions"], y[query], data["patient"][query], data["site"][query]
            )[0],
        )
    del models
    policies = dict(per_pair={pair: dict(step=128, lr=1e-5)}, overall=dict(variant=pair))
    keys = [
        "final_bank_payloads",
        "actual_single_calls",
        "final_models",
        "final_vectors",
        "final_banks",
    ]
    counts = dict.fromkeys(keys, 0)
    audit.audit_final(data, "mixed", "dynamic5m", "unit", "shuffled", policies, records, {}, counts)
    assert counts == dict(
        final_bank_payloads=6, actual_single_calls=6, final_models=3, final_vectors=6, final_banks=1
    )
    outputs[0]["passes"][0, 0, 0] += 0.01
    outpath = tmp_path / "out17.npz"
    np.savez(outpath, **outputs[0])
    records["mixed", pair, 17]["output_sha256"] = digest(outpath)
    with pytest.raises(AssertionError):
        audit.audit_final(
            data,
            "mixed",
            "dynamic5m",
            "unit",
            "shuffled",
            policies,
            records,
            {},
            dict.fromkeys(keys, 0),
        )
    assert not torch.cuda.is_initialized()
```

</details>
