"""Example: Semantic drift detection with sentence-transformers.

Install:
    pip install context-decay-drift[semantic]

No API key needed — runs entirely locally.
"""

from context_decay_drift import DriftAnalyzer, Session, FewShotExample
from context_decay_drift.strategies.sentence_transformer import SentenceTransformerStrategy

# Define the full initial context
session = Session(
    system_prompt=(
        "You are a Python programming tutor. Help students learn Python concepts "
        "including variables, functions, loops, classes, and error handling. "
        "Always provide code examples and explain step by step."
    ),
    few_shot_examples=[
        FewShotExample(
            user="What is a variable?",
            assistant="A variable in Python stores data values. You create one by "
            "assigning a value: x = 5. Variables can hold strings, numbers, lists, "
            "and more. Example: name = 'Alice'",
        ),
        FewShotExample(
            user="How do loops work?",
            assistant="Python has two main loop types: for and while. A for loop "
            "iterates over a sequence: for i in range(5): print(i). A while loop "
            "runs until a condition is false: while x > 0: x -= 1",
        ),
    ],
)

# Use sentence-transformers for semantic drift detection
analyzer = DriftAnalyzer(
    strategies=[SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")],
    decay_rate=0.95,
    window_size=5,
)

# Simulate a multi-turn conversation that progressively drifts
conversations = [
    ("How do I define a function?",
     "Use the def keyword. Example: def greet(name): return f'Hello {name}'. "
     "Functions help organize code into reusable blocks."),

    ("What about classes?",
     "Classes are blueprints for objects. Example: class Dog: def __init__(self, name): "
     "self.name = name. Use classes for object-oriented programming in Python."),

    ("Can you explain error handling?",
     "Use try/except blocks. Example: try: result = 10/0 except ZeroDivisionError: "
     "print('Cannot divide by zero'). Always handle specific exceptions."),

    ("What's a good recipe for dinner?",
     "Try making pasta carbonara! Boil spaghetti, fry pancetta, mix with eggs and parmesan. "
     "Season with black pepper and serve immediately."),

    ("Tell me about the weather",
     "Today's forecast shows sunny skies with a high of 75 degrees. "
     "Tomorrow might bring afternoon showers."),
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
