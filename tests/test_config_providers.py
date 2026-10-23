"""Config loading, provider retry logic, and cost accounting."""
import pytest

from medaudit.config import ModelSpec, load_config, self_judging_overlap
from medaudit.providers import make_provider
from medaudit.providers.base import Completion, Provider, RetryableError, cost_usd

from .conftest import ROOT


def test_default_config_loads(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL_A", "my-model-x")
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert {m.provider for m in cfg.models} == {"openai", "bedrock"}
    assert cfg.models[0].model_id == "my-model-x"  # env override
    assert cfg.judges and not self_judging_overlap(cfg)


def test_disabled_models_are_skipped(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: t\nmodels:\n"
                 "  - {name: A, provider: mock, model_id: a}\n"
                 "  - {name: B, provider: mock, model_id: b, enabled: false}\n"
                 "judges:\n  - {name: J, provider: mock, model_id: a}\n")
    cfg = load_config(p)
    assert [m.name for m in cfg.models] == ["A"]
    assert self_judging_overlap(cfg) == ["a"]


def test_unknown_keys_and_providers_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run_name: t\nmodels:\n  - {name: A, provider: mock, model_id: a, tempurature: 0}\n"
                 "judges:\n  - {name: J, provider: mock, model_id: j}\n")
    with pytest.raises(ValueError, match="unknown model keys"):
        load_config(p)
    p.write_text("run_name: t\nmodels:\n  - {name: A, provider: azure, model_id: a}\n"
                 "judges:\n  - {name: J, provider: mock, model_id: j}\n")
    with pytest.raises(ValueError, match="provider"):
        load_config(p)


def test_cost_usd():
    spec = ModelSpec(name="m", provider="mock", model_id="m", price_input_per_mtok=3.0, price_output_per_mtok=15.0)
    assert cost_usd(spec, 1_000_000, 100_000) == pytest.approx(4.5)
    assert cost_usd(ModelSpec(name="m", provider="mock", model_id="m"), 10, 10) is None


class Flaky(Provider):
    def __init__(self, fails, exc=RetryableError):
        super().__init__(ModelSpec(name="f", provider="mock", model_id="f"), max_retries=4)
        self.fails, self.exc, self.calls = fails, exc, 0

    def _call(self, messages, system, temperature, max_tokens):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc("transient")
        return Completion("ok", 1, 1, 0.0, 0)


def test_retries_transient_errors(monkeypatch):
    monkeypatch.setattr("medaudit.providers.base.time.sleep", lambda s: None)
    p = Flaky(fails=2)
    c = p.complete([{"role": "user", "content": "hi"}])
    assert c.text == "ok" and c.attempts == 3


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("medaudit.providers.base.time.sleep", lambda s: None)
    with pytest.raises(RetryableError):
        Flaky(fails=10).complete([{"role": "user", "content": "hi"}])


def test_non_retryable_errors_propagate_immediately():
    p = Flaky(fails=1, exc=ValueError)
    with pytest.raises(ValueError):
        p.complete([{"role": "user", "content": "hi"}])
    assert p.calls == 1


def test_mock_provider_is_deterministic():
    spec = ModelSpec(name="m", provider="mock", model_id="m", mock_behavior="mixed")
    p = make_provider(spec)
    msgs = [{"role": "user", "content": "question"}]
    assert p.complete(msgs).text == p.complete(msgs).text
