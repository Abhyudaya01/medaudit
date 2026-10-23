"""OpenAI Chat Completions provider."""
from __future__ import annotations

from .base import Completion, Provider, RetryableError


class OpenAIProvider(Provider):
    def __init__(self, spec, **kw):
        super().__init__(spec, **kw)
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'medaudit[openai]'") from e
        self._openai = openai
        # Reads OPENAI_API_KEY from the environment; retries handled here, not by the SDK.
        self.client = openai.OpenAI(timeout=self.timeout_s, max_retries=0)

    def _call(self, messages, system, temperature, max_tokens):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        params = {"model": self.spec.model_id, "messages": msgs,
                  "max_completion_tokens": max_tokens, **self.spec.extra_params}
        if temperature is not None:
            params["temperature"] = temperature
        oa = self._openai
        try:
            r = self.client.chat.completions.create(**params)
        except (oa.RateLimitError, oa.APITimeoutError, oa.APIConnectionError, oa.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        u = r.usage
        return Completion(
            text=r.choices[0].message.content or "",
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,  # includes reasoning tokens
            latency_s=0.0, attempts=0,
            stop_reason=r.choices[0].finish_reason,
        )
