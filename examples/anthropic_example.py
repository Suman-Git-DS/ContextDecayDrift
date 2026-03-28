"""Example: Drop-in wrapper for Anthropic — banking chatbot with drift tracking.

Install:
    pip install context-drift-analyzer[anthropic]

Set your API key:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from anthropic import Anthropic
from context_drift_analyzer import wrap, FewShotExample

client = Anthropic()

# Wrap the client — use it exactly like the original
tracked = wrap(
    client,
    system_prompt=(
        "You are a banking assistant for Acme Bank. Help customers with savings accounts, "
        "credit cards, loans, and account inquiries. Always provide accurate financial "
        "information and guide customers to the right products."
    ),
    few_shot_examples=[
        FewShotExample(
            user="How do I open a savings account?",
            assistant="Visit any Acme Bank branch with your government-issued ID and proof of address. You can also open one online at acmebank.com. Minimum opening deposit is $25.",
        ),
    ],
    mode="always",
    persist=True,
)

# Banking questions followed by off-topic drift
questions = [
    "What are your mortgage rates?",
    "Can I set up automatic bill pay?",
    "What's a good pasta recipe?",              # off-topic
    "How tall is the Eiffel Tower?",            # off-topic
]

for question in questions:
    response = tracked.messages.create(
        model="claude-haiku-4-5-20251001",
        system="You are a banking assistant for Acme Bank.",
        messages=[{"role": "user", "content": question}],
        max_tokens=512,
    )

    content = response.content[0].text
    print(f"User: {question}")
    print(f"  Drift: {response._drift.score:.1f}/100 ({response._drift.verdict.value})")
    print(f"  Explanation: {response._drift_explanation}")

    if response._drift.needs_reset:
        print("  WARNING: Context has drifted too far. Consider resetting.")
        tracked.reset()
        print("  Session reset!")
    print()
