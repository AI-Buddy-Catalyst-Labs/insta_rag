# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Insta RAG** is a modular, configuration-driven Python library for building advanced RAG (Retrieval-Augmented Generation) pipelines. It abstracts document processing, embedding, and hybrid retrieval behind a clean client, allowing developers to ship RAG applications faster.

Key features:
- **Semantic Chunking** – splits docs on topic boundaries to preserve context
- **Hybrid Retrieval** – semantic vectors + BM25 keyword search
- **HyDE Query Transform** – synthesizes hypothetical answers to improve recall
- **Reranking** – optional integration with SOTA rerankers (e.g., Cohere, Novita AI)
- **Pluggable by Design** – swap chunkers, embedders, rerankers, and vector DBs
- **Hybrid Storage** – Qdrant for vectors and MongoDB for flexible content storage

**Status**: Beta (v0.1.1-beta.3)
**Python**: 3.9+
**Package Manager**: uv (with fallback to pip)

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git
- uv (fast Python package manager): `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Initial Setup
```bash
# Install the package in editable mode with dev dependencies
uv pip install -e . --group dev

# Install pre-commit hooks (includes ruff, commitizen, detect-secrets, etc.)
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push
```

## Common Development Commands

### Code Quality
```bash
# Lint and auto-fix issues
ruff check . --fix

# Format code (Black-compatible style)
ruff format .

# Run all pre-commit hooks manually
pre-commit run --all-files

# Check for secrets in code
detect-secrets scan --baseline .secrets.baseline --only-verified
```

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src/insta_rag

# Run a specific test file
pytest tests/smoke_test.py

# Run tests matching a pattern
pytest -k "test_rag_client"

# Run with verbose output
pytest -v
```

### Version Management (using Commitizen)
```bash
# Bump a pre-release version (for beta releases)
cz bump --prerelease beta

# Bump patch version (0.1.0 -> 0.1.1)
cz bump --patch

# Bump minor version (0.1.0 -> 0.2.0)
cz bump --minor

# Bump major version (0.1.0 -> 1.0.0)
cz bump --major

# After bumping, push commits and tags
git push origin your-branch
git push origin --tags
```

## Project Structure

### Source Code (`src/insta_rag/`)

- **`core/`** – Main RAG client and configuration
  - `client.py` – `RAGClient` class, orchestrates all RAG operations (add, retrieve, update, delete documents)
  - `config.py` – Configuration management with dataclasses for VectorDB, Embedding, LLM, Chunking, etc.
  - `retrieval_method.py` – Enum for retrieval method selection (vector, keyword, hybrid)

- **`models/`** – Data models using Pydantic
  - `document.py` – `DocumentInput` for ingestion, supports text/PDF/URL sources
  - `chunk.py` – `Chunk` model for semantic units
  - `response.py` – Response models for API returns

- **`chunking/`** – Document splitting strategies
  - `semantic.py` – `SemanticChunker` uses topic boundaries for context preservation
  - Also includes fallback chunkers for various document types

- **`embedding/`** – Embedding providers
  - `openai.py` – OpenAI embeddings (text-embedding-3-large, etc.)
  - `base.py` – Abstract base for custom embedders
  - Support for Azure OpenAI and other providers via config

- **`retrieval/`** – Core retrieval logic
  - `vector_search.py` – Semantic search using vectors
  - `keyword_search.py` – BM25 keyword search
  - `query_generator.py` – HyDE query transformation (synthesizes hypothetical answers)
  - `reranker.py` – Reranking using cross-encoders (Cohere, Novita AI, etc.)
  - `base.py` – Base retriever interface

- **`vectordb/`** – Vector database integrations
  - `qdrant.py` – Qdrant vector store implementation
  - `base.py` – Abstract base for custom vector DBs

- **`utils/`** – Utility functions
  - `pdf_processing.py` – PDF text extraction using pdfplumber/PyPDF2
  - `exceptions.py` – Custom exceptions (ConfigurationError, ValidationError, VectorDBError)

### Testing
- **`tests/`** – Test suite
  - `conftest.py` – Pytest fixtures (loads .env.test)
  - `smoke_test.py` – Basic integration tests for RAGClient
- **`examples/`** – Example usage and integration patterns

## Architecture Concepts

### RAG Pipeline Flow
```
Documents → Chunking → Embedding → Vector DB
     ↓          ↓           ↓          ↓
   Input   Semantic    OpenAI      Qdrant
           Splitting   (3072D)
              ↓
          Vector Store + Metadata Storage (MongoDB optional)
                ↓
           Retrieval & Reranking
                ↓
            Results (Top K chunks)
```

### Key Architectural Patterns

1. **Configuration-driven** – All behavior is controlled via `RAGConfig` (environment variables + dataclasses)
2. **Pluggable components** – Base classes (`BaseEmbedder`, `BaseRetriever`, etc.) allow swapping implementations
3. **Lazy initialization** – Components are initialized only when needed
4. **Hybrid retrieval** – Combines BM25 (keyword) and vector search with configurable weights
5. **Metadata filtering** – All chunks support arbitrary metadata for filtering during retrieval

### Configuration Flow
Environment variables (`.env`) are parsed into config dataclasses in this order:
1. `RAGConfig.from_env()` reads environment variables
2. Individual config classes validate their inputs (e.g., `VectorDBConfig`, `EmbeddingConfig`)
3. `RAGClient.__init__()` initializes components based on config

## Code Quality Standards

### Pre-commit Hooks (Enforced on commit)
- **ruff-check** – Linting with auto-fix
- **ruff-format** – Code formatting (88-char line length, double quotes)
- **uv-lock/export** – Dependency management
- **trailing-whitespace** – Removes trailing spaces
- **mixed-line-ending** – Enforces LF line endings
- **mdformat** – Markdown formatting
- **commitizen** – Validates conventional commit messages
- **detect-secrets** – Detects hardcoded secrets

### Commit Conventions
Use conventional commits (enforced by pre-commit):
```bash
# Use commitizen interactive prompt (recommended)
cz commit

# Or write manually:
git commit -m "feat: add HyDE query transformation for improved recall"
git commit -m "fix: handle PDF extraction errors gracefully"
git commit -m "docs: update retrieval configuration examples"
```

Format: `type(scope): subject`
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

### Branch Naming
- `feat/description` – New features
- `fix/description` – Bug fixes
- `docs/description` – Documentation
- `refactor/description` – Code refactoring

## Key Configuration Environment Variables

```dotenv
# Vector Store (Qdrant)
INSTA_RAG_QDRANT_URL=https://your-qdrant:6333
INSTA_RAG_QDRANT_API_KEY=...

# Embedding Model (OpenAI)
INSTA_RAG_EMBED_MODEL=text-embedding-3-large
OPENAI_API_KEY=...

# Optional: MongoDB for content storage
INSTA_RAG_MONGODB_URI=mongodb+srv://...
INSTA_RAG_MONGODB_DB=insta_rag

# Retrieval Options
INSTA_RAG_HYBRID_ENABLED=true
INSTA_RAG_BM25_WEIGHT=0.35
INSTA_RAG_VECTOR_WEIGHT=0.65

# HyDE Query Transformation
INSTA_RAG_HYDE_ENABLED=true
INSTA_RAG_HYDE_MODEL=gpt-4o-mini

# Reranking (optional)
INSTA_RAG_RERANKER=cohere-rerank-3
COHERE_API_KEY=...
```

See `src/insta_rag/core/config.py` for full list of config options.

## Important Files to Know

- **`pyproject.toml`** – Dependencies, build config, tool settings (ruff, pytest, commitizen)
- **`.pre-commit-config.yaml`** – Pre-commit hook definitions
- **`CONTRIBUTING.md`** – Full contributor guide with setup details
- **`README.md`** – User-facing documentation with examples
- **`.env`** – Runtime configuration (excluded from git)
- **`uv.lock`** – Locked dependency versions (commit this)

## Testing Notes

- Tests are in `tests/` directory
- Uses pytest with fixtures defined in `conftest.py`
- Loads `.env.test` for test-specific configuration
- Smoke tests in `smoke_test.py` validate basic RAGClient initialization
- Use `pytest-mock` for mocking external services (OpenAI, Qdrant)

## Dependency Management with uv

```bash
# Add a new dependency
uv pip install package-name

# Install with groups
uv pip install -e . --group dev

# Lock dependencies
uv lock

# Export requirements
uv export > requirements.txt
```

Note: Pre-commit hooks automatically manage `uv.lock` and `requirements.txt`.

## Roadmap & Known Limitations

- Roadmap items: built-in summarization, more rerankers, CLI GA, LangChain/LlamaIndex adapters, streaming/tracing hooks
- Currently beta status – API may change before v1.0
- Reranking is optional but recommended for long-tail queries
- Vector DB integration focuses on Qdrant (extensible for others)
