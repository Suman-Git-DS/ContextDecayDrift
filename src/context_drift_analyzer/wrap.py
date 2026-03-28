"""Drop-in LLM client wrapper with drift tracking.

Wraps any LLM client so drift tracking happens transparently —
no need to manually call record_turn(). Just use the wrapper
like you'd use the original client.

Usage with OpenAI:
    from openai import OpenAI
    from context_drift_analyzer import wrap

    client = OpenAI()
    tracked = wrap(client, system_prompt="You are a Python tutor.")

    # Use exactly like the original client
    response = tracked.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Python tutor."},
            {"role": "user", "content": "How do loops work?"},
        ],
    )

    # Drift score is attached to the response
    print(response._drift.score)        # 85.2
    print(response._drift_explanation)   # "Context well-preserved..."

    # Or check on demand
    report = tracked.drift_check()

Usage with Anthropic:
    from anthropic import Anthropic
    from context_drift_analyzer import wrap

    client = Anthropic()
    tracked = wrap(client, system_prompt="You are a Python tutor.")

    response = tracked.messages.create(
        model="claude-haiku-4-5-20251001",
        system="You are a Python tutor.",
        messages=[{"role": "user", "content": "How do loops work?"}],
        max_tokens=200,
    )

    print(response._drift.score)
    report = tracked.drift_check()
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from context_drift_analyzer.core.session import FewShotExample
from context_drift_analyzer.tracker import DriftReport, DriftTracker


class _OpenAIChatProxy:
    """Proxy for client.chat that intercepts completions.create()."""

    def __init__(self, original_chat: Any, wrapper: DriftClientWrapper):
        self._original = original_chat
        self._wrapper = wrapper
        self.completions = _OpenAICompletionsProxy(
            original_chat.completions, wrapper
        )


class _OpenAICompletionsProxy:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, original: Any, wrapper: DriftClientWrapper):
        self._original = original
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        # Extract user message from messages list
        messages = kwargs.get("messages", [])
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Call the original
        response = self._original.create(**kwargs)

        # Extract response content
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""

        # Record turn and attach drift
        result = self._wrapper._tracker.record_turn(user_msg, content)
        response._drift = result.drift
        response._drift_explanation = result.explanation
        response._managed_context = result.managed_context

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _AnthropicMessagesProxy:
    """Proxy for client.messages that intercepts create()."""

    def __init__(self, original: Any, wrapper: DriftClientWrapper):
        self._original = original
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        # Extract user message
        messages = kwargs.get("messages", [])
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_msg = content
                elif isinstance(content, list):
                    # Anthropic content blocks
                    user_msg = " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                break

        # Call the original
        response = self._original.create(**kwargs)

        # Extract response content
        content = ""
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

        # Record turn and attach drift
        result = self._wrapper._tracker.record_turn(user_msg, content)
        response._drift = result.drift
        response._drift_explanation = result.explanation
        response._managed_context = result.managed_context

        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class DriftClientWrapper:
    """Drop-in wrapper around any LLM client.

    Intercepts API calls, records turns, and attaches drift scores
    to every response. The original client is fully accessible.

    Args:
        client: The LLM client (OpenAI, Anthropic, or any).
        system_prompt: System prompt to track drift against.
        few_shot_examples: Optional few-shot pairs.
        mode: "always" or "ondemand".
        persist: Save to .session_memory file.
        persist_path: Path for persistence file.
        max_summary_sessions: Session summaries to retain.
        **tracker_kwargs: Additional kwargs passed to DriftTracker.
    """

    def __init__(
        self,
        client: Any,
        system_prompt: str,
        few_shot_examples: Optional[list[FewShotExample]] = None,
        mode: str = "always",
        persist: bool = False,
        persist_path: str = ".session_memory",
        max_summary_sessions: int = 3,
        **tracker_kwargs: Any,
    ):
        self._client = client
        self._tracker = DriftTracker(
            system_prompt=system_prompt,
            few_shot_examples=few_shot_examples,
            mode=mode,
            persist=persist,
            persist_path=persist_path,
            max_summary_sessions=max_summary_sessions,
            **tracker_kwargs,
        )

        # Detect client type and set up proxies
        client_type = type(client).__name__.lower()
        client_module = type(client).__module__ or ""

        if "openai" in client_module or "openai" in client_type:
            self.chat = _OpenAIChatProxy(client.chat, self)
        elif "anthropic" in client_module or "anthropic" in client_type:
            self.messages = _AnthropicMessagesProxy(client.messages, self)

    def drift_check(self) -> DriftReport:
        """On-demand drift check."""
        return self._tracker.check()

    def end_session(self) -> Optional[DriftReport]:
        """End current session and summarize."""
        return self._tracker.end_session()

    def get_managed_context(self) -> str:
        """Get the managed context for LLM calls."""
        return self._tracker.get_managed_context()

    def reset(self) -> None:
        """Full reset."""
        self._tracker.reset()

    def freeze_context(self) -> None:
        self._tracker.freeze_context()

    def unfreeze_context(self) -> None:
        self._tracker.unfreeze_context()

    def clear_history(self) -> None:
        self._tracker.clear_history()

    @property
    def tracker(self) -> DriftTracker:
        """Access the underlying DriftTracker."""
        return self._tracker

    def __getattr__(self, name: str) -> Any:
        """Fall through to the original client for any other attribute."""
        return getattr(self._client, name)


def wrap(
    client: Any,
    system_prompt: str,
    few_shot_examples: Optional[list[FewShotExample]] = None,
    mode: str = "always",
    persist: bool = False,
    persist_path: str = ".session_memory",
    max_summary_sessions: int = 3,
    **tracker_kwargs: Any,
) -> DriftClientWrapper:
    """Wrap an LLM client with drift tracking.

    This is the recommended way to add drift tracking to your existing
    LLM pipeline. It's a drop-in replacement — use the wrapped client
    exactly like the original, and drift scores are attached automatically.

    Args:
        client: Any LLM client (OpenAI, Anthropic, etc.)
        system_prompt: System prompt to track drift against.
        few_shot_examples: Optional few-shot pairs.
        mode: "always" or "ondemand".
        persist: Save to .session_memory file.
        persist_path: Path for persistence file.
        max_summary_sessions: Session summaries to retain.
        **tracker_kwargs: Additional kwargs for DriftTracker.

    Returns:
        A DriftClientWrapper that behaves like the original client.

    Example:
        from openai import OpenAI
        from context_drift_analyzer import wrap

        client = OpenAI()
        tracked = wrap(client, system_prompt="You are a Python tutor.")

        response = tracked.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response._drift.score)  # Drift score attached!
    """
    return DriftClientWrapper(
        client=client,
        system_prompt=system_prompt,
        few_shot_examples=few_shot_examples,
        mode=mode,
        persist=persist,
        persist_path=persist_path,
        max_summary_sessions=max_summary_sessions,
        **tracker_kwargs,
    )
