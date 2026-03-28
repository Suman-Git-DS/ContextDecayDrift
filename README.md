<p align="center">
  <h1 align="center">context-decay-drift</h1>
  <p align="center">
    Measure and monitor how LLM conversations drift from their system prompt over time.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="PyPI" src="https://img.shields.io/pypi/v/context-decay-drift"></a>
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/context-decay-drift"></a>
  <a href="https://github.com/Suman-Git-DS/ContextDecayDrift/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Suman-Git-DS/ContextDecayDrift"></a>
</p>

---

## The Problem

LLM-powered chatbots lose focus over long conversations. After several turns, the model "forgets" its system prompt, leading to:

- Off-topic responses that ignore the bot's intended persona
- Reduced accuracy as the conversation context window fills up
- Poor user experience with no visibility into when the bot becomes unreliable

**context-decay-drift** solves this by giving you a **real-time drift score (0-100)** that tracks how far a conversation has drifted from its original system prompt. Wrap it around any LLM client and get actionable signals for when to reset, warn, or intervene.

## How It Works

```
Session 1 (Turn 1-2):  Score 95  [FRESH]     - Bot is on-topic
Session 2 (Turn 3-6):  Score 78  [MILD]      - Slight drift detected
Session 3 (Turn 7-12): Score 58  [MODERATE]   - Noticeable drift
Session 4 (Turn 13+):  Score 34  [SEVERE]     - Bot is off-rails, reset recommended
```

The library combines multiple strategies (keyword tracking, token overlap analysis) with exponential decay to produce a single drift score. No external API calls needed for scoring - it runs entirely locally.

## Installation

```bash
# Core package (no dependencies)
pip install context-decay-drift

# With OpenAI wrapper
pip install context-decay-drift[openai]

# With Anthropic wrapper
pip install context-decay-drift[anthropic]

# Everything
pip install context-decay-drift[all]
```

## Quick Start

### Generic Tracker (Any LLM)

Works with any LLM pipeline. Just feed in the turns manually:

```python
from context_decay_drift.providers.generic import GenericDriftTracker

tracker = GenericDriftTracker(
    system_prompt="You are a Python tutor. Always provide code examples.",
    decay_rate=0.95,    # How fast context decays (0-1, lower = faster)
    window_size=5,      # Recent turns to evaluate
)

# After each LLM call in your pipeline:
drift = tracker.record_turn(
    user_message="How do I use list comprehensions?",
    assistant_response="List comprehensions provide a concise way to create lists..."
)

print(f"Drift Score: {drift.score:.1f}/100")  # e.g., 82.3/100
print(f"Verdict: {drift.verdict.value}")       # e.g., "mild"
print(f"Still Effective: {drift.is_effective}") # True/False
print(f"Needs Reset: {drift.needs_reset}")     # True/False
```

### OpenAI Integration

Wraps the OpenAI Python SDK. Drift is computed automatically after each response:

```python
from openai import OpenAI
from context_decay_drift.providers.openai_provider import OpenAIDriftWrapper

client = OpenAI()  # Uses OPENAI_API_KEY env var

wrapper = OpenAIDriftWrapper(
    client=client,
    model="gpt-4o",
    system_prompt="You are a Python tutor. Always provide code examples.",
)

result = wrapper.chat("How do I define a function?")

print(result.content)                           # The LLM response text
print(f"Drift: {result.drift_score:.1f}/100")   # 85.2/100
print(f"Verdict: {result.drift_verdict}")        # "mild"
print(result.response)                           # Original OpenAI response object
```

### Anthropic Integration

Same pattern for Anthropic's Claude:

```python
from anthropic import Anthropic
from context_decay_drift.providers.anthropic_provider import AnthropicDriftWrapper

client = Anthropic()  # Uses ANTHROPIC_API_KEY env var

wrapper = AnthropicDriftWrapper(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a Python tutor. Always provide code examples.",
    max_tokens=1024,
)

result = wrapper.chat("Explain decorators in Python")

print(result.content)
print(f"Drift: {result.drift_score:.1f}/100")

# Auto-reset when drift is critical
if result.drift.needs_reset:
    wrapper.reset_session()
```

## Drift Score Reference

| Score Range | Verdict      | Meaning                                           |
|-------------|-------------|---------------------------------------------------|
| 90 - 100    | `FRESH`     | Context well-preserved, bot is on-topic           |
| 75 - 89     | `MILD`      | Minor drift, still effective                      |
| 55 - 74     | `MODERATE`  | Noticeable drift, monitor closely                 |
| 35 - 54     | `SEVERE`    | Significant drift, consider resetting context     |
| 0 - 34      | `CRITICAL`  | Context largely lost, reset recommended           |

## Configuration

### Decay Rate

Controls how fast the score decays per conversation turn. Range: `(0, 1]`.

```python
# Slow decay - forgiving, good for long conversations
DriftAnalyzer(decay_rate=0.98)

# Default - balanced
DriftAnalyzer(decay_rate=0.95)

# Fast decay - strict, flags drift early
DriftAnalyzer(decay_rate=0.90)
```

### Window Size

How many recent assistant turns to evaluate. `0` means evaluate all turns.

```python
# Only look at last 3 responses (responsive to recent changes)
DriftAnalyzer(window_size=3)

# Look at last 10 responses (smoother, less noisy)
DriftAnalyzer(window_size=10)

# Evaluate entire conversation history
DriftAnalyzer(window_size=0)
```

### Custom Strategies

Mix and weight different drift detection strategies:

```python
from context_decay_drift import DriftAnalyzer
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy
from context_decay_drift.strategies.composite import CompositeStrategy

# Custom composite with 70/30 weighting
strategy = CompositeStrategy(
    strategies=[
        KeywordStrategy(top_n=50),           # Track top 50 system prompt keywords
        TokenOverlapStrategy(),               # Cosine similarity of term frequencies
    ],
    weights=[0.7, 0.3],  # Keyword matching weighted higher
)

analyzer = DriftAnalyzer(strategies=[strategy])
```

## Advanced Usage

### Direct Analyzer API

Use the analyzer directly without a provider wrapper:

```python
from context_decay_drift import DriftAnalyzer, Session

session = Session(system_prompt="You are a helpful coding assistant.")
analyzer = DriftAnalyzer(decay_rate=0.93)

# Record turns
session.add_user_message("How do I sort a list?")
session.add_assistant_message("Use the sorted() function or list.sort() method...")

# Get drift score
result = analyzer.analyze(session)
print(result.to_dict())
# {
#   "score": 78.5,
#   "verdict": "mild",
#   "is_effective": True,
#   "needs_reset": False,
#   "turn_number": 2,
#   "session_id": "a1b2c3d4e5f6",
#   "strategy_scores": {"keyword": 72.0, "token_overlap": 85.0, "composite": 78.5},
#   "metadata": {"raw_score": 80.1, "decay_factor": 0.98, ...}
# }
```

### Session Management

```python
from context_decay_drift import Session

session = Session(
    system_prompt="You are a Python tutor.",
    session_id="user-123-session-1",  # Custom ID for tracking
)

session.add_user_message("Hello")
session.add_assistant_message("Hi! Let's learn Python.")

print(session.turn_count)        # 2
print(session.assistant_turns)   # [Turn(role='assistant', ...)]
print(session.get_recent_context(n=3))  # Last 3 turns

session.reset()  # Clear turns, keep system prompt
```

### Building Custom Strategies

Extend `BaseStrategy` to implement your own drift detection logic:

```python
from context_decay_drift.strategies.base import BaseStrategy

class SentimentStrategy(BaseStrategy):
    """Detect drift via sentiment shift from system prompt tone."""

    @property
    def name(self) -> str:
        return "sentiment"

    def score(self, system_prompt: str, assistant_responses: list[str]) -> tuple[float, dict[str, float]]:
        # Your custom logic here
        score = compute_sentiment_alignment(system_prompt, assistant_responses)
        return score, {self.name: score}
```

## Project Structure

```
context_decay_drift/
  src/context_decay_drift/
    core/
      analyzer.py       # Central drift analysis engine
      scorer.py         # DriftScore and DriftVerdict data structures
      session.py        # Session and Turn management
    strategies/
      base.py           # BaseStrategy abstract class
      keyword.py        # Keyword hit-rate strategy
      token_overlap.py  # Cosine similarity strategy
      composite.py      # Weighted multi-strategy combiner
    providers/
      base.py           # BaseProvider and DriftAwareResponse
      openai_provider.py    # OpenAI SDK wrapper
      anthropic_provider.py # Anthropic SDK wrapper
      generic.py        # Provider-agnostic tracker
    utils/
      text.py           # Tokenization, TF, cosine similarity
  tests/                # 75 tests covering all modules
  examples/             # Ready-to-run examples for each provider
```

## Running Tests

```bash
git clone https://github.com/Suman-Git-DS/ContextDecayDrift.git
cd ContextDecayDrift
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [PyPI Package](https://pypi.org/project/context-decay-drift/)
- [GitHub Repository](https://github.com/Suman-Git-DS/ContextDecayDrift)
- [Issue Tracker](https://github.com/Suman-Git-DS/ContextDecayDrift/issues)
