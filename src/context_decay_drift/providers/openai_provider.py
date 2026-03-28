"""OpenAI provider wrapper with drift tracking.

Usage:
    from openai import OpenAI
    from context_decay_drift.providers.openai_provider import OpenAIDriftWrapper

    client = OpenAI()
    wrapper = OpenAIDriftWrapper(
        client=client,
        model="gpt-4o",
        system_prompt="You are a helpful Python tutor...",
    )

    result = wrapper.chat("How do I use list comprehensions?")
    print(result.content)
    print(f"Drift: {result.drift_score:.1f} ({result.drift_verdict})")
"""

from __future__ import annotations

from typing import Any, Optional

from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.session import Session
from context_decay_drift.providers.base import BaseProvider, DriftAwareResponse
from context_decay_drift.strategies.base import BaseStrategy


class OpenAIDriftWrapper(BaseProvider):
    """Wraps an OpenAI client with automatic drift tracking.

    Args:
        client: An initialized OpenAI client instance.
        model: Model name to use (e.g. "gpt-4o", "gpt-4o-mini").
        system_prompt: The system prompt to track drift against.
        analyzer: Optional pre-configured DriftAnalyzer.
        strategies: Optional list of strategies.
        session: Optional pre-existing session.
        decay_rate: Exponential decay rate.
        window_size: Turn window size.
        **chat_kwargs: Additional default kwargs passed to chat.completions.create.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        system_prompt: str,
        analyzer: Optional[DriftAnalyzer] = None,
        strategies: Optional[list[BaseStrategy]] = None,
        session: Optional[Session] = None,
        decay_rate: float = 0.95,
        window_size: int = 5,
        **chat_kwargs: Any,
    ):
        super().__init__(
            system_prompt=system_prompt,
            analyzer=analyzer,
            strategies=strategies,
            session=session,
            decay_rate=decay_rate,
            window_size=window_size,
        )
        self.client = client
        self.model = model
        self.chat_kwargs = chat_kwargs

    def chat(
        self,
        user_message: str,
        **kwargs: Any,
    ) -> DriftAwareResponse:
        """Send a message and get a drift-aware response.

        Args:
            user_message: The user's message text.
            **kwargs: Override any chat completion kwargs for this call.

        Returns:
            DriftAwareResponse with the LLM response and drift score.
        """
        self.session.add_user_message(user_message)

        # Build messages list
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.session.turns:
            messages.append({"role": turn.role, "content": turn.content})

        # Merge default kwargs with per-call overrides
        call_kwargs = {**self.chat_kwargs, **kwargs}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **call_kwargs,
        )

        # Extract content
        content = response.choices[0].message.content or ""
        self.session.add_assistant_message(content)

        # Compute drift
        drift = self.analyzer.analyze(self.session)

        return DriftAwareResponse(
            response=response,
            content=content,
            drift=drift,
        )
