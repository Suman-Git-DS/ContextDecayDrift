"""Drift measurement strategies.

Available strategies:
    - KeywordStrategy: Lexical keyword hit-rate (zero dependencies)
    - TokenOverlapStrategy: TF cosine similarity (zero dependencies)
    - SentenceTransformerStrategy: Semantic via sentence-transformers (requires [semantic])
    - OpenAIEmbeddingStrategy: Semantic via OpenAI API (requires [openai])
    - CallableEmbeddingStrategy: Bring your own embedding function
    - CompositeStrategy: Weighted combination of any strategies above
"""
