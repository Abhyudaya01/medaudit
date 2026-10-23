"""Amazon Bedrock provider via the model-agnostic Converse API.

Works for any Bedrock text model that supports Converse (Anthropic, Meta Llama,
Mistral, Amazon Nova, ...). `model_id` may be a model ID or an inference-profile
ID (e.g. "us.<provider>.<model>"), which many newer models require.
"""
from __future__ import annotations

from .base import Completion, Provider, RetryableError

_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "ModelTimeoutException",
              "InternalServerException", "ModelNotReadyException", "TooManyRequestsException"}


class BedrockProvider(Provider):
    def __init__(self, spec, **kw):
        super().__init__(spec, **kw)
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'medaudit[bedrock]'") from e
        self._ClientError = ClientError
        self.client = boto3.client(
            "bedrock-runtime", region_name=spec.region,
            config=Config(read_timeout=self.timeout_s, retries={"max_attempts": 1}),
        )

    def _call(self, messages, system, temperature, max_tokens):
        inference = {"maxTokens": max_tokens}
        if temperature is not None:
            inference["temperature"] = temperature
        params = {
            "modelId": self.spec.model_id,
            "messages": [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages],
            "inferenceConfig": inference,
        }
        if system:
            params["system"] = [{"text": system}]
        if self.spec.extra_params:
            params["additionalModelRequestFields"] = self.spec.extra_params
        try:
            r = self.client.converse(**params)
        except self._ClientError as e:
            if e.response.get("Error", {}).get("Code") in _RETRYABLE:
                raise RetryableError(str(e)) from e
            raise
        blocks = r["output"]["message"]["content"]
        text = "".join(b.get("text", "") for b in blocks)
        usage = r.get("usage", {})
        return Completion(
            text=text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            latency_s=0.0, attempts=0,
            stop_reason=r.get("stopReason"),
            provider_latency_ms=r.get("metrics", {}).get("latencyMs"),
        )
