from ..config import ModelSpec
from .base import Completion, Provider, RetryableError, cost_usd


def make_provider(spec: ModelSpec, timeout_s: float = 120.0, max_retries: int = 5) -> Provider:
    if spec.provider == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(spec, timeout_s=timeout_s, max_retries=max_retries)
    if spec.provider == "bedrock":
        from .bedrock_provider import BedrockProvider
        return BedrockProvider(spec, timeout_s=timeout_s, max_retries=max_retries)
    if spec.provider == "mock":
        from .mock_provider import MockProvider
        return MockProvider(spec, timeout_s=timeout_s, max_retries=max_retries)
    raise ValueError(f"unknown provider {spec.provider}")


__all__ = ["Completion", "Provider", "RetryableError", "cost_usd", "make_provider"]
