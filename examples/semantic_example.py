"""Example: Semantic drift detection for a banking chatbot using sentence-transformers.

Install:
    pip install context-drift-analyzer[semantic]

No API key needed — runs entirely locally.
"""

from context_drift_analyzer import DriftAnalyzer, Session, FewShotExample
from context_drift_analyzer.strategies.sentence_transformer import SentenceTransformerStrategy

# Define the banking assistant context
session = Session(
    system_prompt=(
        "You are a banking assistant for Acme Bank. Help customers with savings accounts, "
        "credit cards, loans, and account inquiries. Always provide accurate financial "
        "information and guide customers to the right products."
    ),
    few_shot_examples=[
        FewShotExample(
            user="What interest rate do your savings accounts offer?",
            assistant="Our standard savings account offers 4.5% APY. Premium savings "
            "offers 5.1% APY for balances over $10,000. Both are FDIC insured.",
        ),
        FewShotExample(
            user="How do I apply for a credit card?",
            assistant="You can apply online at acmebank.com/cards or visit any branch. "
            "You'll need your ID, proof of income, and SSN. Approval typically takes "
            "1-2 business days.",
        ),
    ],
)

# Use sentence-transformers for semantic drift detection
analyzer = DriftAnalyzer(
    strategies=[SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")],
    decay_rate=0.95,
    window_size=5,
)

# Simulate a conversation that starts on-topic then drifts
conversations = [
    ("What are the requirements for a home loan?",
     "For a home loan at Acme Bank, you'll need a credit score of 620+, proof of income, "
     "2 years of tax returns, and a down payment of at least 3.5% for FHA loans or 20% "
     "for conventional loans. Current rates start at 6.2% APR."),

    ("Can I refinance my existing mortgage?",
     "Yes! Refinancing can lower your monthly payments or shorten your loan term. "
     "You'll need your current mortgage statement, recent pay stubs, and a home appraisal. "
     "Our refinance rates start at 5.8% APR for qualified borrowers."),

    ("What about your business checking accounts?",
     "Our business checking offers unlimited transactions, free online banking, and "
     "integrated payroll. Basic plan is free with $5,000 minimum balance, Premium plan "
     "is $25/month with cash management tools."),

    ("What's a good recipe for dinner tonight?",
     "Try making grilled salmon with a lemon herb butter! Season with salt, pepper, and "
     "garlic. Serve with roasted asparagus and rice pilaf. Takes about 25 minutes."),

    ("Tell me about the best vacation spots",
     "Bali offers stunning beaches and temples, Japan has incredible food and culture, "
     "and Iceland has breathtaking northern lights. Book flights 3 months ahead for deals."),
]

print("Turn | Score  | Verdict    | Effective | Topic")
print("-" * 70)

for user_msg, assistant_msg in conversations:
    session.add_user_message(user_msg)
    session.add_assistant_message(assistant_msg)

    result = analyzer.analyze(session)
    print(
        f"  {result.turn_number:2d} | "
        f"{result.score:5.1f}  | "
        f"{result.verdict.value:10s} | "
        f"{'Yes' if result.is_effective else 'NO ':3s}       | "
        f"{user_msg[:40]}"
    )

    if result.needs_reset:
        print("     ^ ALERT: Context drift is critical. Recommend resetting session.")

print("\n--- Detailed Final Report ---")
final = analyzer.analyze(session)
for key, value in final.to_dict().items():
    print(f"  {key}: {value}")
