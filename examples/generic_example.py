"""Example: Using context-decay-drift with any LLM (generic tracker).

No extra dependencies needed:
    pip install context-decay-drift
"""

from context_decay_drift.providers.generic import GenericDriftTracker

tracker = GenericDriftTracker(
    system_prompt=(
        "You are a Python programming tutor. Help students learn Python concepts "
        "including variables, functions, loops, classes, and error handling. "
        "Always provide code examples and explain step by step."
    ),
    decay_rate=0.93,
    window_size=5,
)

# Simulate turns from any LLM pipeline
conversations = [
    ("How do I use loops?", "In Python, you can use for and while loops. For example: for i in range(10): print(i). This loops through numbers 0 to 9 step by step."),
    ("What about functions?", "Functions are defined with the def keyword. Example: def add(a, b): return a + b. Functions help organize your Python code into reusable blocks."),
    ("Tell me about movies", "The latest blockbuster movie has amazing special effects. Critics gave it 4 out of 5 stars."),
    ("What's for dinner?", "You could try making grilled salmon with roasted vegetables. Season with lemon, garlic, and herbs."),
    ("Any travel tips?", "When traveling to Europe, pack light and use public transportation. Book accommodations early for better rates."),
]

print("Turn | Score | Verdict    | Effective | Content Preview")
print("-" * 75)

for user_msg, assistant_msg in conversations:
    drift = tracker.record_turn(user_msg, assistant_msg)
    print(
        f"  {drift.turn_number:2d} | "
        f"{drift.score:5.1f} | "
        f"{drift.verdict.value:10s} | "
        f"{'Yes' if drift.is_effective else 'NO ':3s}       | "
        f"{user_msg[:35]}"
    )

# Show detailed final state
print("\n--- Final Drift Report ---")
final = tracker.get_drift()
report = final.to_dict()
for key, value in report.items():
    print(f"  {key}: {value}")
