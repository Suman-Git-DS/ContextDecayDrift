"""Example: Drop-in wrapper for OpenAI — banking chatbot with drift tracking.

Install:
    pip install context-drift-analyzer[openai]

Set your API key:
    export OPENAI_API_KEY="sk-..."
"""

from openai import OpenAI
from context_drift_analyzer import wrap, FewShotExample

client = OpenAI()

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
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. Premium savings offers 5.1% APY for balances over $10,000.",
        ),
    ],
    mode="always",
    persist=True,
)

# Banking-related questions followed by off-topic ones
questions = [
    "What credit cards do you offer?",
    "How do I apply for a home loan?",
    "Can I open a joint checking account?",
    "What's the weather like today?",           # off-topic
    "Tell me a joke about cats",                # off-topic
    "What's the best restaurant in NYC?",       # off-topic
]

for question in questions:
    response = tracked.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a banking assistant for Acme Bank."},
            {"role": "user", "content": question},
        ],
    )

    content = response.choices[0].message.content
    print(f"User: {question}")
    print(f"Assistant: {content[:100]}...")
    print(f"  Drift: {response._drift.score:.1f}/100 ({response._drift.verdict.value})")
    print(f"  Explanation: {response._drift_explanation}")
    print()

# End session
report = tracked.end_session()
print(f"--- Session ended (final score: {report.drift.score:.1f}) ---")
