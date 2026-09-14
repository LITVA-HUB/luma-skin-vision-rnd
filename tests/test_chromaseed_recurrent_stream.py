"""CPU-only streaming equivalence and genuine computation omission."""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from chromaseed_architecture_scale import specs  # noqa: E402
from chromaseed_head_range import Predictor  # noqa: E402
from chromaseed_recurrent_stream import StreamingPredictor  # noqa: E402

VARIANTS = ("soft_small", "dynamic_small", "soft5m", "dynamic5m")
MODES = ("unit", "wide", "linear")


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def synthetic(variant="dynamic_small", mode="unit"):
    rng = np.random.default_rng(9142601)
    theta = []
    for name, incoming, outgoing in specs(variant):
        theta.append(rng.normal(0, 0.8 / np.sqrt(incoming), incoming * outgoing))
        if name != "key":
            theta.append(rng.normal(0.1 if name != "head" else 0.7, 0.15, outgoing))
    model = dict(
        variant=np.asarray(variant), head_mode=np.asarray(mode),
        theta=np.concatenate(theta).astype(np.float32),
        t_mean=rng.normal(0, 0.2, 18).astype(np.float32),
        t_std=rng.uniform(0.5, 2, 18).astype(np.float32),
        base_x_mean=rng.normal(0, 0.2, 36).astype(np.float32),
        base_x_std=rng.uniform(0.5, 2, 36).astype(np.float32),
        base_y_mean=np.asarray([52, 14, 19], np.float32),
        base_y_std=np.asarray([5, 8, 9], np.float32),
        base_w0=rng.normal(0, 0.1, (36, 16)).astype(np.float32),
        base_b0=rng.normal(0, 0.1, 16).astype(np.float32),
        base_v0=rng.normal(0, 0.1, (16, 3)).astype(np.float32),
        base_c0=rng.normal(0, 0.1, 3).astype(np.float32),
    )
    return model, rng.normal(size=36).astype(np.float32), rng.normal(size=(64, 18)).astype(np.float32)


class Observed(StreamingPredictor):
    def __init__(self, model):
        super().__init__(model)
        self.calls = []

    def _layer(self, name, value):
        self.calls.append(name)
        return super()._layer(name, value)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("mode", MODES)
def test_every_prefix_matches_frozen_consumer_with_nonzero_weights(variant, mode):
    model, x, t = synthetic(variant, mode)
    expected = Predictor(model)(x, t, all_passes=True)
    consumer = StreamingPredictor(model)
    actual = list(consumer.iter_passes(x, t))
    np.testing.assert_array_equal(np.stack([p.prediction for p in actual]), expected)
    assert np.max(np.abs(np.diff(expected, axis=0))) > 1e-3
    assert [p.index for p in actual] == [1, 2, 3, 4]
    assert actual[0].change_l2 is None
    for i in range(1, 4):
        assert actual[i].change_l2 == pytest.approx(np.linalg.norm(expected[i] - expected[i-1]))
    assert all(4 <= p.active_tokens <= 64 for p in actual)
    if variant.startswith("soft"):
        assert all(p.active_tokens == 64 for p in actual)
    for count in (1, 2, 3, 4):
        trace = consumer.predict_trace(x, t, max_passes=count)
        assert trace.passes_executed == count and trace.stop_reason == "max_passes"
        np.testing.assert_array_equal(trace.prediction, expected[count-1])
    np.testing.assert_array_equal(consumer.predict(x, t), expected[-1])


def test_iterator_is_lazy_and_does_not_execute_discarded_passes():
    model, x, t = synthetic()
    consumer = Observed(model)
    stream = consumer.iter_passes(x, t)
    assert consumer.calls == []
    next(stream)
    first = ["token1", "token2", "context", "key", "query", "update1", "update2", "head"]
    assert consumer.calls == first
    next(stream)
    assert consumer.calls == first + ["query", "update1", "update2", "head"]
    stream.close()
    assert len(consumer.calls) == 12


def test_explicit_threshold_really_stops_and_default_does_not():
    model, x, t = synthetic()
    model["exit_threshold"] = np.asarray(1e6)  # An old payload setting is not adopted.
    consumer = Observed(model)
    expected = Predictor(model)(x, t, all_passes=True)
    trace = consumer.predict_trace(x, t, exit_threshold=1e6)
    assert trace.passes_executed == 2 and trace.stop_reason == "delta_threshold"
    np.testing.assert_array_equal(trace.prediction, expected[1])
    counts = Counter(consumer.calls)
    assert counts == Counter(token1=1, token2=1, context=1, key=1, query=2, update1=2, update2=2, head=2)
    consumer.calls.clear()
    assert consumer.predict_trace(x, t).passes_executed == 4
    assert Counter(consumer.calls)["head"] == 4
    assert consumer.predict_trace(x, t, exit_threshold=1e6, min_passes=3).passes_executed == 3
    assert consumer.predict_trace(x, t, exit_threshold=1e-12).passes_executed == 4


def test_snapshots_and_interleaved_requests_do_not_modify_recurrent_state():
    model, x, t = synthetic()
    consumer = StreamingPredictor(model)
    a = consumer.iter_passes(x, t)
    b = consumer.iter_passes(-x, -t)
    first = next(a)
    saved = first.prediction.copy()
    next(b)
    second = next(a)
    np.testing.assert_array_equal(first.prediction, saved)
    # Public output storage never aliases the private recurrent prediction.
    second.prediction.setflags(write=True)
    second.prediction[:] = 1e20
    final = list(a)[-1].prediction
    np.testing.assert_array_equal(final, consumer.predict(x, t))
    np.testing.assert_array_equal(list(b)[-1].prediction, consumer.predict(-x, -t))
    model["theta"][:] = 0
    np.testing.assert_array_equal(consumer.predict(x, t), final)


def test_dynamic_fallback_keeps_exactly_four_tied_nonpositive_tokens():
    model, x, t = synthetic()
    offset = 0
    for name, incoming, outgoing in specs("dynamic_small"):
        size = (incoming + (name != "key")) * outgoing
        if name == "key":
            model["theta"][offset:offset+size] = 0
        offset += size
    consumer = StreamingPredictor(model)
    passes = list(consumer.iter_passes(x, t))
    assert all(p.active_tokens == 4 for p in passes)
    np.testing.assert_array_equal(np.stack([p.prediction for p in passes]),
                                  Predictor(model)(x, t, all_passes=True))


@pytest.mark.parametrize("field,value", [
    ("max_passes", 0), ("max_passes", 5), ("max_passes", True), ("max_passes", 2.5),
    ("min_passes", 1), ("min_passes", 5), ("min_passes", True),
    ("exit_threshold", -1), ("exit_threshold", float("nan")),
    ("exit_threshold", float("inf")), ("exit_threshold", True),
])
def test_invalid_exit_policy_fails_before_any_layer(field, value):
    model, x, t = synthetic()
    consumer = Observed(model)
    with pytest.raises(ValueError):
        consumer.predict_trace(x, t, **{field: value})
    assert not consumer.calls


def test_invalid_features_and_payloads_are_rejected():
    model, x, t = synthetic()
    consumer = Observed(model)
    for xx, tt in ((x[None], t), (x, t[None]), (x[:-1], t), (x, t[:-1]), (x*np.nan, t)):
        with pytest.raises(ValueError):
            consumer.predict(xx, tt)
    assert not consumer.calls
    for key, value in (("variant", "patch5m"), ("head_mode", "unknown"),
                       ("theta", model["theta"][:-1]), ("base_x_std", np.zeros(36)),
                       ("t_std", np.ones((1, 18))), ("base_y_mean", [1, 2, np.inf])):
        with pytest.raises(ValueError):
            StreamingPredictor({**model, key: value})


def test_standalone_consumer_does_not_import_torch():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    code = (f"import sys; sys.path.insert(0, {scripts!r}); "
            "import chromaseed_recurrent_stream; assert 'torch' not in sys.modules")
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_qualification_checks_real_prefix_work_and_no_quality_claim():
    from chromaseed_recurrent_stream_check import qualify_case

    model, x, t = synthetic()
    result = qualify_case(model, x, t)
    assert result["all_passes_bitwise_equal"]
    assert result["max_abs_lab_drift"] == 0.0
    assert [p["linear_macs"] for p in result["prefixes"]] == [135336, 144912, 154488, 164064]
    assert result["forced_exit"]["passes_executed"] == 2
    assert result["forced_exit"]["linear_macs"] == 144912
    assert result["default_passes"] == 4
    assert result["quality_metric"] is None and result["latency_seconds"] is None


def test_qualification_rejects_an_input_that_changed(tmp_path):
    from chromaseed_recurrent_stream_check import binding, check_bindings

    source = tmp_path / "source.txt"
    source.write_text("before", encoding="utf-8")
    records = [binding(source)]
    check_bindings(records)
    source.write_text("after", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        check_bindings(records)
