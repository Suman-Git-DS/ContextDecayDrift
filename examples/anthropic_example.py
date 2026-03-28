"""Example: Using context-decay-drift with Anthropic.

Install:
    pip install context-decay-drift[anthropic]

Set your API key:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from anthropic import Anthropic
from context_decay_drift.providers.anthropic_provider import AnthropicDriftWrapper

client = Anthropic()

wrapper = AnthropicDriftWrapper(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt=(
        "You are a Python programming tutor. Help students learn Python concepts "
        "including variables, functions, loops, classes, and error handling. "
        "Always provide code examples and explain step by step."
    ),
    max_tokens=512,
    decay_rate=0.95,
)

# Simulate a conversation
questions = [
    "How do I use classes in Python?",
    "Explain inheritance with an example",
    "What's a good pasta recipe?",           # off-topic
    "How tall is the Eiffel Tower?",          # off-topic
]

for question in questions:
    result = wrapper.chat(question)
    print(f"User: {question}")
    print(f"  Drift: {result.drift_score:.1f}/100 ({result.drift_verdict})")

    if result.drift.needs_reset:
        print("  WARNING: Context has drifted too far. Consider resetting.")
        wrapper.reset_session()
        print("  Session reset!")
    print()
