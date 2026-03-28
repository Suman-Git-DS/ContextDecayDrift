"""Example: Using context-drift-analyzer with any LLM — banking chatbot.

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
    window_size=5,
    max_summary_sessions=3,
)

# Simulate turns from any LLM pipeline — banking questions that gradually drift
conversations = [
    (
        "What credit cards do you have?",
        "We offer Acme Rewards (1.5% cashback on everything), Acme Travel (3x points on flights and hotels), and Acme Business (2% on business purchases). All cards have no annual fee the first year.",
    ),
    (
        "How do I set up direct deposit?",
        "Log in to online banking, go to Account Settings, and select Direct Deposit. You'll find your routing number and account number there. Share these with your employer's payroll department.",
    ),
    (
        "Tell me about the latest movies",
        "The latest sci-fi blockbuster just broke box office records with stunning visual effects. Critics gave it 4 out of 5 stars and audiences loved the plot twists.",
    ),
    (
        "What's the best workout routine?",
        "A balanced routine includes 3 days of strength training, 2 days of cardio, and 2 rest days. Start with compound movements like squats, deadlifts, and bench press.",
    ),
    (
        "Any good book recommendations?",
        "Try 'Atomic Habits' by James Clear for productivity, 'Project Hail Mary' for sci-fi, or 'Educated' by Tara Westover for a powerful memoir.",
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

# End session
report = tracker.end_session()
print(f"\n--- Session ended (score: {report.drift.score:.1f}) ---")
print(f"\nManaged context for next session:\n{tracker.get_managed_context()}")
