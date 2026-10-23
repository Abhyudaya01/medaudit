"""Provider interface: one call -> text + usage + latency + cost."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

from ..config import ModelSpec


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float          # wall-clock for the successful attempt
    attempts: int
    stop_reason: str | None = None
    provider_latency_ms: float | None = None  # server-reported where available (Bedrock)


class RetryableError(Exception):
    """Rate limits, timeouts, 5xx — safe to retry."""


class Provider:
    def __init__(self, spec: ModelSpec, timeout_s: float = 120.0, max_retries: int = 5):
        self.spec = spec
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    # Subclasses implement one attempt; raise RetryableError for transient failures.
    def _call(self, messages: list[dict], system: str | None, temperature: float | None,
              max_tokens: int) -> Completion:
        raise NotImplementedError

    def complete(self, messages: list[dict], system: str | None = None,
                 temperature: float | None = "default", max_tokens: int | None = None) -> Completion:
        temp = self.spec.temperature if temperature == "default" else temperature
        mt = max_tokens or self.spec.max_tokens
        sys_prompt = system if system is not None else self.spec.system_prompt
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                out = self._call(messages, sys_prompt, temp, mt)
                out.latency_s = time.perf_counter() - t0
                out.attempts = attempt
                return out
            except RetryableError:
                if attempt == self.max_retries:
                    raise
                time.sleep(min(60.0, (2 ** attempt) + random.random()))
        raise RuntimeError("unreachable")

    def cost_usd(self, c: Completion) -> float | None:
        return cost_usd(self.spec, c.input_tokens, c.output_tokens)


def cost_usd(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float | None:
    if spec.price_input_per_mtok is None or spec.price_output_per_mtok is None:
        return None
    return (input_tokens * spec.price_input_per_mtok + output_tokens * spec.price_output_per_mtok) / 1e6
