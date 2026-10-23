"""YAML configuration: which models to audit, which judge(s), and run settings."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelSpec:
    name: str                          # label used in tables/plots
    provider: str                      # openai | bedrock | mock
    model_id: str                      # provider model ID / Bedrock model or inference-profile ID
    region: str | None = None          # Bedrock region
    temperature: float | None = 0.0    # None = omit (some reasoning models reject it)
    max_tokens: int = 1024
    system_prompt: str | None = None
    price_input_per_mtok: float | None = None   # USD per 1M input tokens
    price_output_per_mtok: float | None = None  # USD per 1M output tokens
    max_concurrency: int = 4
    extra_params: dict = field(default_factory=dict)
    mock_behavior: str | None = None   # mock provider only: safe | unsafe | mixed | refuse


@dataclass
class JudgeSpec:
    model: ModelSpec
    temperature: float = 0.0


@dataclass
class RunConfig:
    run_name: str
    output_dir: Path
    models: list[ModelSpec]
    judges: list[JudgeSpec]
    samples_per_prompt: int = 1
    categories: list[str] | None = None
    seed_ids: list[str] | None = None
    limit: int | None = None
    max_retries: int = 5
    request_timeout_s: float = 120.0
    bootstrap_iters: int = 2000
    random_seed: int = 17
    human_sample_size: int = 120


_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand_env(obj):
    if isinstance(obj, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _model(d: dict) -> ModelSpec:
    known = ModelSpec.__dataclass_fields__
    unknown = set(d) - set(known)
    if unknown:
        raise ValueError(f"unknown model keys {sorted(unknown)} in {d.get('name')}")
    if d.get("provider") not in ("openai", "bedrock", "mock"):
        raise ValueError(f"model {d.get('name')}: provider must be openai, bedrock, or mock")
    return ModelSpec(**d)


def load_config(path: str | Path, overrides: dict | None = None) -> RunConfig:
    raw = _expand_env(yaml.safe_load(Path(path).read_text()))
    raw.update(overrides or {})
    models = [_model({k: v for k, v in m.items() if k != "enabled"})
              for m in raw["models"] if m.get("enabled", True)]
    judges = [JudgeSpec(model=_model({k: v for k, v in j.items() if k != "judge_temperature"}),
                        temperature=j.get("judge_temperature", 0.0)) for j in raw["judges"]]
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique")
    if not judges:
        raise ValueError("at least one judge is required")
    return RunConfig(
        run_name=raw["run_name"],
        output_dir=Path(raw.get("output_dir", "runs")),
        models=models,
        judges=judges,
        samples_per_prompt=int(raw.get("samples_per_prompt", 1)),
        categories=raw.get("categories"),
        seed_ids=raw.get("seed_ids"),
        limit=raw.get("limit"),
        max_retries=int(raw.get("max_retries", 5)),
        request_timeout_s=float(raw.get("request_timeout_s", 120)),
        bootstrap_iters=int(raw.get("bootstrap_iters", 2000)),
        random_seed=int(raw.get("random_seed", 17)),
        human_sample_size=int(raw.get("human_sample_size", 120)),
    )


def self_judging_overlap(cfg: RunConfig) -> list[str]:
    """Model IDs that are both audited and used as judge (self-preference bias risk)."""
    judged = {m.model_id for m in cfg.models}
    return sorted(j.model.model_id for j in cfg.judges if j.model.model_id in judged)
