"""Example: Using context-decay-drift with OpenAI.

Install:
    pip install context-decay-drift[openai]

Set your API key:
    export OPENAI_API_KEY="sk-..."
"""

from openai import OpenAI
from context_decay_drift.providers.openai_provider import OpenAIDriftWrapper

client = OpenAI()

wrapper = OpenAIDriftWrapper(
    client=client,
    model="gpt-4o-mini",
    system_prompt=(
        "You are a Python programming tutor. Help students learn Python concepts "
        "including variables, functions, loops, classes, and error handling. "
        "Always provide code examples and explain step by step."
    ),
    decay_rate=0.95,
    window_size=5,
)

# Simulate a multi-turn conversation
questions = [
    "How do I define a function in Python?",
    "What are list comprehensions?",
    "Can you explain decorators?",
    "What's the weather like today?",       # off-topic
    "Tell me a joke about cats",             # off-topic
    "What's the best restaurant in NYC?",    # off-topic
]

for question in questions:
    result = wrapper.chat(question)
    print(f"User: {question}")
    print(f"Assistant: {result.content[:100]}...")
    print(f"  Drift Score: {result.drift_score:.1f}/100 ({result.drift_verdict})")
    print(f"  Effective: {result.drift.is_effective} | Needs Reset: {result.drift.needs_reset}")
    print()
