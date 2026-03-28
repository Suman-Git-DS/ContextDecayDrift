"""Tests for Session v2 features (FewShotExample, initial_context)."""

import pytest
from context_drift_analyzer.core.session import Session, FewShotExample


class TestFewShotExample:
    def test_create(self):
        ex = FewShotExample(user="Hello", assistant="Hi there!")
        assert ex.user == "Hello"
        assert ex.assistant == "Hi there!"


class TestSessionInitialContext:
    def test_initial_context_no_few_shots(self):
        s = Session(system_prompt="You are a helpful bot.")
        assert s.initial_context == "You are a helpful bot."

    def test_initial_context_with_few_shots(self):
        s = Session(
            system_prompt="You are a bot.",
            few_shot_examples=[
                FewShotExample(user="What is 2+2?", assistant="4"),
                FewShotExample(user="Hi", assistant="Hello!"),
            ],
        )
        ctx = s.initial_context
        assert "You are a bot." in ctx
        assert "User: What is 2+2?" in ctx
        assert "Assistant: 4" in ctx
        assert "User: Hi" in ctx
        assert "Assistant: Hello!" in ctx

    def test_initial_context_preserves_order(self):
        s = Session(
            system_prompt="System.",
            few_shot_examples=[
                FewShotExample(user="First", assistant="A1"),
                FewShotExample(user="Second", assistant="A2"),
            ],
        )
        ctx = s.initial_context
        assert ctx.index("First") < ctx.index("Second")
        assert ctx.index("A1") < ctx.index("A2")

    def test_reset_preserves_few_shots(self):
        s = Session(
            system_prompt="Test",
            few_shot_examples=[FewShotExample(user="Q", assistant="A")],
        )
        s.add_user_message("Hello")
        s.add_assistant_message("Hi")
        s.reset()
        assert len(s.few_shot_examples) == 1
        assert s.turn_count == 0

    def test_empty_few_shots_by_default(self):
        s = Session(system_prompt="Test")
        assert s.few_shot_examples == []
