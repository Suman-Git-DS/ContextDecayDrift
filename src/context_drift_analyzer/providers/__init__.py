"""LLM provider wrappers with built-in drift tracking.

Note: Provider-specific wrappers (OpenAI, Anthropic) have been replaced
by the unified DriftTracker. Use DriftTracker instead — it works with
any LLM pipeline.

The GenericDriftTracker is kept for backward compatibility.
"""

from context_drift_analyzer.providers.base import BaseProvider, DriftAwareResponse
from context_drift_analyzer.providers.generic import GenericDriftTracker

__all__ = [
    "BaseProvider",
    "DriftAwareResponse",
    "GenericDriftTracker",
]
