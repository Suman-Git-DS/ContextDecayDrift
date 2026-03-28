# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-03-28

### Changed
- **Renamed package** from `context-decay-drift` to `context-drift-analyzer`
- All examples and docs now use a **banking chatbot** theme for consistency
- Version bump to 0.4.0

### Added
- **Drop-in client wrapper** (`wrap()`) — wrap any LLM client (OpenAI, Anthropic) and get drift scores attached to every response automatically
  - `response._drift.score` — drift score on every API response
  - `response._drift_explanation` — human-readable explanation
  - `response._managed_context` — managed context string
- `DriftClientWrapper` class with `drift_check()`, `end_session()`, `freeze_context()`, `get_managed_context()`
- `CONTRIBUTING.md` — contributor guidelines
- `CHANGELOG.md` — this file
- 24 new tests for the client wrapper (`test_wrap.py`)

### Fixed
- Banking-themed examples for all supported providers

## [0.3.0] - 2026-03-27

### Added
- **Context management** — original system prompt + few-shot examples are always preserved, with compressed session summaries appended
- **Drift explanation** — every drift score includes a 1-2 line human-readable explanation of why drift occurred
- **Persistence** (`.session_memory` file) — drift data survives restarts and deploys
- **On-demand mode** — choose between `mode="always"` (score every turn) and `mode="ondemand"` (score on request)
- **CLI tool** — `context-drift-analyzer status`, `history`, `reset`, `freeze`, `unfreeze`
- **Session management** — `end_session()` summarizes and preserves context, `freeze_context()` / `unfreeze_context()` for control
- Custom `summarize_fn` and `explain_fn` callbacks for LLM-powered summarization and explanation

### Changed
- Removed provider-specific wrappers (OpenAI/Anthropic providers) in favor of generic tracker
- `DriftTracker` is now the single main entry point

## [0.2.0] - 2026-03-26

### Added
- **Semantic embedding strategies** — `SentenceTransformerStrategy`, `OpenAIEmbeddingStrategy`, `CallableEmbeddingStrategy`
- **Composite strategy** — weighted combination of multiple strategies
- **Calibrated scoring** — cosine similarity between instruction text and response text is scaled from [0, 0.55] to [0, 100] for meaningful scores
- Auto-detection of `sentence-transformers` as default strategy when installed
- Reference embedding caching for performance

### Fixed
- **Critical scoring bug**: on-topic responses (e.g., "How do loops work?" to a Python tutor) were scoring 0.0 due to lexical strategies having zero word overlap
- Markdown stripping now preserves code content (keeps `def hello()` instead of removing it)
- Explainer rewritten to use score-based messages instead of misleading keyword percentages

## [0.1.0] - 2026-03-25

### Added
- Initial release
- `DriftTracker` with keyword and token-overlap strategies
- `DriftScore` with verdicts: FRESH, MILD, MODERATE, SEVERE, CRITICAL
- `Session` and `FewShotExample` for conversation tracking
- Exponential decay scoring
- Markdown stripping before embedding
- Basic test suite (90+ tests)
