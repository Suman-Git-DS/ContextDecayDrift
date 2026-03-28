"""LLM provider wrappers with built-in drift tracking."""

from context_decay_drift.providers.base import BaseProvider, DriftAwareResponse
from context_decay_drift.providers.generic import GenericDriftTracker

__all__ = [
    "BaseProvider",
    "DriftAwareResponse",
    "GenericDriftTracker",
]

# Lazy imports for optional provider dependencies
def __getattr__(name: str):
    if name == "OpenAIDriftWrapper":
        from context_decay_drift.providers.openai_provider import OpenAIDriftWrapper
        return OpenAIDriftWrapper
    if name == "AnthropicDriftWrapper":
        from context_decay_drift.providers.anthropic_provider import AnthropicDriftWrapper
        return AnthropicDriftWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
