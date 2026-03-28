<p align="center">
  <h1 align="center">context-decay-drift</h1>
  <p align="center">
    Measure and monitor how LLM conversations drift from their initial context over time — using semantic embeddings.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="PyPI" src="https://img.shields.io/pypi/v/context-decay-drift"></a>
  <a href="https://pypi.org/project/context-decay-drift/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/context-decay-drift"></a>
  <a href="https://github.com/Suman-Git-DS/ContextDecayDrift/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/Suman-Git-DS/ContextDecayDrift"></a>
</p>

---

## The Problem

LLM-powered chatbots lose focus over long conversations. After several turns, the model "forgets" its system prompt and few-shot examples, leading to:

- Off-topic responses that ignore the bot's intended persona
- Reduced accuracy as the conversation context window fills up
- Poor user experience with no visibility into when the bot becomes unreliable

**context-decay-drift** solves this by giving you a **real-time drift score (0-100)** that tracks how far a conversation has semantically drifted from its initial context (system prompt + few-shot examples). Wrap it around any LLM client and get actionable signals for when to reset, warn, or intervene.

## How It Works

```
Session 1 (Turn 1-2):  Score 95  [FRESH]     - Bot is on-topic
Session 2 (Turn 3-6):  Score 78  [MILD]      - Slight drift detected
Session 3 (Turn 7-12): Score 58  [MODERATE]   - Noticeable drift
Session 4 (Turn 13+):  Score 34  [SEVERE]     - Bot is off-rails, reset recommended
```

The library embeds your initial context (system prompt + few-shot examples) and compares it against recent assistant responses using **cosine similarity of embedding vectors**. You choose the embedding backend:

| Backend | Cost | Quality | Install |
|---------|------|---------|---------|
| Sentence Transformers (default) | Free, local | Great | `pip install context-decay-drift[semantic]` |
| OpenAI Embeddings | ~$0.02/1M tokens | Excellent | `pip install context-decay-drift[openai]` |
| Any custom model | Varies | You decide | `pip install context-decay-drift` |
| Keyword/TF fallback | Free, local | Basic | `pip install context-decay-drift` |

## Installation

```bash
# Core + Sentence Transformers (recommended)
pip install context-decay-drift[semantic]

# Core + OpenAI embeddings
pip install context-decay-drift[openai]

# Core + Anthropic wrapper
pip install context-decay-drift[anthropic]

# Everything
pip install context-decay-drift[all]

# Core only (keyword/TF strategies, or bring your own embedder)
pip install context-decay-drift
```

## Quick Start

### Sentence Transformers (Recommended — Free, Local, Semantic)

```python
from context_decay_drift import DriftAnalyzer, Session, FewShotExample
from context_decay_drift.strategies.sentence_transformer import SentenceTransformerStrategy

# Define your initial context: system prompt + few-shot examples
session = Session(
    system_prompt="You are a Python programming tutor. Always provide code examples.",
    few_shot_examples=[
        FewShotExample(
            user="What is a variable?",
            assistant="A variable stores data. Example: x = 5"
        ),
        FewShotExample(
            user="How do loops work?",
            assistant="Loops iterate over sequences. Example: for i in range(10): print(i)"
        ),
    ],
)

# Use sentence-transformers for semantic drift detection
analyzer = DriftAnalyzer(
    strategies=[SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")],
    decay_rate=0.95,
)

# Simulate conversation turns
session.add_user_message("How do I define a function?")
session.add_assistant_message(
    "Use the def keyword. Example: def greet(name): return f'Hello {name}'"
)

result = analyzer.analyze(session)
print(f"Drift: {result.score:.1f}/100 ({result.verdict.value})")
# Drift: 88.2/100 (mild)

# Later... conversation drifts off topic
session.add_user_message("What's a good pasta recipe?")
session.add_assistant_message("Try carbonara: eggs, parmesan, pancetta, spaghetti.")

result = analyzer.analyze(session)
print(f"Drift: {result.score:.1f}/100 ({result.verdict.value})")
# Drift: 41.5/100 (severe)
print(f"Still effective: {result.is_effective}")  # False
print(f"Needs reset: {result.needs_reset}")       # True
```

### OpenAI Embeddings (Paid API, High Quality)

```python
from openai import OpenAI
from context_decay_drift import DriftAnalyzer, Session
from context_decay_drift.strategies.openai_embedding import OpenAIEmbeddingStrategy

client = OpenAI()  # Uses OPENAI_API_KEY

analyzer = DriftAnalyzer(
    strategies=[OpenAIEmbeddingStrategy(client=client, model="text-embedding-3-small")],
)

session = Session(system_prompt="You are a Python tutor. Always explain with examples.")
session.add_user_message("Explain decorators")
session.add_assistant_message("Decorators wrap functions. Example: @my_decorator...")

result = analyzer.analyze(session)
print(f"Drift: {result.score:.1f}/100")
```

### Bring Your Own Embedder (Cohere, Voyage, Google, etc.)

```python
from context_decay_drift import DriftAnalyzer, Session
from context_decay_drift.strategies.callable_embedding import CallableEmbeddingStrategy

# Example: Cohere
import cohere
co = cohere.Client("your-api-key")

def cohere_embed(text: str) -> list[float]:
    response = co.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
    return response.embeddings[0]

analyzer = DriftAnalyzer(
    strategies=[CallableEmbeddingStrategy(embed_fn=cohere_embed, strategy_name="cohere")]
)

# Example: Google Vertex AI
def vertex_embed(text: str) -> list[float]:
    # your Vertex AI embedding logic here
    ...

analyzer = DriftAnalyzer(
    strategies=[CallableEmbeddingStrategy(embed_fn=vertex_embed, strategy_name="vertex")]
)
```

### OpenAI Chat Wrapper (Drift Score in Every Response)

```python
from openai import OpenAI
from context_decay_drift.providers.openai_provider import OpenAIDriftWrapper

client = OpenAI()

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

# Full drift details
print(result.drift.to_dict())
```

### Anthropic Chat Wrapper

```python
from anthropic import Anthropic
from context_decay_drift.providers.anthropic_provider import AnthropicDriftWrapper

client = Anthropic()

wrapper = AnthropicDriftWrapper(
    client=client,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a Python tutor. Always provide code examples.",
    max_tokens=1024,
)

result = wrapper.chat("Explain decorators in Python")
print(f"Drift: {result.drift_score:.1f}/100")

if result.drift.needs_reset:
    wrapper.reset_session()
```

### Generic Tracker (Any LLM Pipeline)

```python
from context_decay_drift.providers.generic import GenericDriftTracker

tracker = GenericDriftTracker(
    system_prompt="You are a Python tutor. Always provide code examples.",
    decay_rate=0.95,
    window_size=5,
)

# After each LLM call in your pipeline:
drift = tracker.record_turn(
    user_message="How do I use list comprehensions?",
    assistant_response="List comprehensions provide a concise way to create lists..."
)

print(f"Drift Score: {drift.score:.1f}/100")
print(f"Verdict: {drift.verdict.value}")
print(f"Still Effective: {drift.is_effective}")
print(f"Needs Reset: {drift.needs_reset}")
```

## Few-Shot Examples as Initial Context

The initial context your drift is measured against is not just the system prompt — it includes few-shot examples too. This is critical because many production bots rely on few-shot pairs to define behavior:

```python
from context_decay_drift import Session, FewShotExample

session = Session(
    system_prompt="You are a customer support agent for Acme Corp.",
    few_shot_examples=[
        FewShotExample(
            user="I can't log in",
            assistant="I'm sorry to hear that. Let me help you reset your password. Please go to acme.com/reset."
        ),
        FewShotExample(
            user="How do I upgrade my plan?",
            assistant="You can upgrade anytime at acme.com/billing. Would you like me to walk you through it?"
        ),
    ],
)

# session.initial_context now contains the full reference text:
# system prompt + all few-shot pairs combined
# This is what drift is measured against
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
DriftAnalyzer(decay_rate=0.98)  # Slow decay — forgiving for long conversations
DriftAnalyzer(decay_rate=0.95)  # Default — balanced
DriftAnalyzer(decay_rate=0.90)  # Fast decay — strict, flags drift early
```

### Window Size

How many recent assistant turns to evaluate. `0` means evaluate all turns.

```python
DriftAnalyzer(window_size=3)   # Responsive to recent changes
DriftAnalyzer(window_size=10)  # Smoother, less noisy
DriftAnalyzer(window_size=0)   # Evaluate entire history
```

### Composite Strategies

Mix semantic and lexical strategies with custom weights:

```python
from context_decay_drift.strategies.sentence_transformer import SentenceTransformerStrategy
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.composite import CompositeStrategy

strategy = CompositeStrategy(
    strategies=[
        SentenceTransformerStrategy(),   # Semantic meaning
        KeywordStrategy(top_n=50),       # Lexical keyword presence
    ],
    weights=[0.8, 0.2],  # 80% semantic, 20% keyword
)

analyzer = DriftAnalyzer(strategies=[strategy])
```

### Sentence Transformer Models

Choose based on your speed/quality tradeoff:

```python
# Fast, 80MB, good quality (default)
SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2")

# Best quality, 420MB, slower
SentenceTransformerStrategy(model_name="all-mpnet-base-v2")

# Fastest, 60MB, decent quality
SentenceTransformerStrategy(model_name="paraphrase-MiniLM-L3-v2")

# GPU acceleration
SentenceTransformerStrategy(model_name="all-MiniLM-L6-v2", device="cuda")
```

## How Scoring Works

```
                    Initial Context
                   (System Prompt +
                    Few-Shot Examples)
                          |
                     [Embed Once]
                          |
                    Reference Vector
                          |
    Turn 1 ───────────────┤ cosine_similarity(ref, response) → 0.92
    Turn 2 ───────────────┤ cosine_similarity(ref, response) → 0.85
    Turn 3 ───────────────┤ cosine_similarity(ref, response) → 0.61
    Turn 4 ───────────────┤ cosine_similarity(ref, response) → 0.38
                          |
                   Apply Decay Factor
                 (decay_rate ^ turns/2)
                          |
                   Final Score (0-100)
```

1. **Embed the initial context** (system prompt + few-shot pairs) into a reference vector
2. **Embed recent assistant responses** into a current vector
3. **Cosine similarity** between reference and current = raw alignment score
4. **Exponential decay** applied based on conversation length
5. **Clamp to 0-100** and classify into verdict

The reference embedding is **cached** — it's computed once and reused for every turn.

## Project Structure

```
context_decay_drift/
  src/context_decay_drift/
    core/
      analyzer.py          # Central drift analysis engine
      scorer.py            # DriftScore and DriftVerdict data structures
      session.py           # Session, Turn, and FewShotExample management
    strategies/
      base.py              # BaseStrategy abstract class
      embedding_base.py    # EmbeddingStrategy base (shared scoring logic)
      sentence_transformer.py  # HuggingFace sentence-transformers backend
      openai_embedding.py  # OpenAI embedding API backend
      callable_embedding.py    # Bring-your-own embedding function
      keyword.py           # Keyword hit-rate strategy (lexical fallback)
      token_overlap.py     # TF cosine similarity (lexical fallback)
      composite.py         # Weighted multi-strategy combiner
    providers/
      base.py              # BaseProvider and DriftAwareResponse
      openai_provider.py   # OpenAI chat SDK wrapper
      anthropic_provider.py    # Anthropic chat SDK wrapper
      generic.py           # Provider-agnostic tracker
    utils/
      text.py              # Tokenization, TF vectors, cosine similarity
  tests/                   # 97 tests covering all modules
  examples/                # Ready-to-run examples for each provider
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
