"""Example: Basic drift tracking for a banking chatbot.

No extra dependencies needed:
    pip install context-drift-analyzer
"""

from context_drift_analyzer import DriftTracker, FewShotExample

tracker = DriftTracker(
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
    decay_rate=0.93,
    max_summary_sessions=3,
)

# Simulate a multi-turn conversation that progressively drifts from banking
conversations = [
    (
        "What credit cards do you offer?",
        "We offer three credit cards: Acme Rewards (1.5% cashback), Acme Travel (3x points on travel), and Acme Business (2% on business expenses). All have no annual fee for the first year.",
    ),
    (
        "How do I apply for a home loan?",
        "You can apply for a home loan online at acmebank.com/loans or visit any branch. You'll need your ID, proof of income, 2 years of tax returns, and a down payment of at least 3.5% for FHA or 20% for conventional.",
    ),
    (
        "What's a good pasta recipe?",
        "Try pasta carbonara! Boil spaghetti, fry pancetta until crispy, mix with eggs and parmesan, and toss with the hot pasta. Season with black pepper.",
    ),
    (
        "Tell me about the best travel destinations",
        "Bali is amazing for beaches, Tokyo for culture, and Switzerland for mountains. Book flights 3 months ahead for the best deals.",
    ),
    (
        "How tall is the Eiffel Tower?",
        "The Eiffel Tower stands 330 meters tall including antennas. It was built in 1889 and attracts about 7 million visitors annually.",
    ),
]

print("Turn | Score | Verdict    | Explanation")
print("-" * 90)

for user_msg, assistant_msg in conversations:
    result = tracker.record_turn(user_msg, assistant_msg)
    if result.drift:
        print(
            f"  {result.drift.turn_number:2d} | "
            f"{result.drift.score:5.1f} | "
            f"{result.drift.verdict.value:10s} | "
            f"{result.explanation[:60] if result.explanation else ''}"
        )

# End session and see managed context
report = tracker.end_session()
print(f"\n--- Session ended (score: {report.drift.score:.1f}) ---")
print(f"\nManaged context for next session:\n{tracker.get_managed_context()}")
