# Contributing to context-drift-analyzer

Thanks for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Suman-Git-DS/ContextDriftAnalyzer.git
cd ContextDriftAnalyzer

# Install in development mode with all dev dependencies
pip install -e ".[dev,semantic]"

# Run the test suite
pytest tests/ -v
```

## Project Structure

```
src/context_drift_analyzer/
  tracker.py               # Main entry point — DriftTracker
  wrap.py                  # Drop-in client wrapper
  core/                    # Analyzer, scorer, session management
  context/                 # Context window management + explainer
  persistence/             # .session_memory file handling
  strategies/              # Drift detection strategies (embedding, keyword, etc.)
  cli/                     # Command-line interface
  utils/                   # Text processing, markdown stripping
tests/                     # Test suite (188 tests)
examples/                  # Banking chatbot examples
```

## How to Contribute

### Reporting Bugs

- Open an issue at [GitHub Issues](https://github.com/Suman-Git-DS/ContextDriftAnalyzer/issues)
- Include: Python version, OS, package version, steps to reproduce, expected vs actual behavior
- If drift scores seem wrong, include the system prompt and response that produced the unexpected score

### Suggesting Features

- Open an issue with the `enhancement` label
- Describe the use case, not just the solution
- Examples of where the feature would help (e.g., "When monitoring a customer support bot, I need to...")

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Write tests** for new functionality — we aim for high coverage
4. **Run the full test suite** before submitting:
   ```bash
   pytest tests/ -v
   ```
5. **Submit a pull request** with a clear description of what changed and why

### Code Guidelines

- **Keep it simple.** No premature abstractions. Three similar lines > one clever utility.
- **Tests are required** for new features and bug fixes.
- **Type hints** are used throughout — maintain them in new code.
- **No new dependencies** in the core package. Optional dependencies go in `[project.optional-dependencies]`.
- **Docstrings** for public APIs. Internal code should be self-explanatory.

### Adding a New Embedding Strategy

The most common contribution is a new embedding backend. Here's how:

1. Create `src/context_drift_analyzer/strategies/your_backend.py`
2. Subclass `EmbeddingStrategy` and implement `embed(text) -> list[float]`
3. Add tests in `tests/test_your_backend.py`
4. Add the dependency to `pyproject.toml` under `[project.optional-dependencies]`
5. Add a lazy import in `__init__.py`

```python
from context_drift_analyzer.strategies.embedding_base import EmbeddingStrategy

class YourStrategy(EmbeddingStrategy):
    name = "your-backend"

    def embed(self, text: str) -> list[float]:
        # Your embedding logic here
        ...
```

### Writing Tests

- Tests go in `tests/` and follow the `test_*.py` naming convention
- Use `pytest` fixtures for shared setup
- Mock external APIs (OpenAI, Anthropic) — don't make real API calls in tests
- For embedding strategies, test with small deterministic vectors

## Release Process

Releases are managed by the maintainer:

1. Version bump in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Tag the release: `git tag v0.x.0`
4. Publish to PyPI: `python -m build && twine upload dist/*`

## Code of Conduct

Be respectful, constructive, and inclusive. We're all here to build better tools.

## Questions?

Open an issue or start a discussion on [GitHub](https://github.com/Suman-Git-DS/ContextDriftAnalyzer).
