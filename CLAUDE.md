
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
- **Graph RAG** *(NEW)* – Knowledge graph-based RAG using Neo4j and Graphiti for structured entity/relationship extraction

**Status**: Beta (v0.1.1-beta.3)
**Python**: 3.9+
**Package Manager**: uv (with fallback to pip)

### Graph RAG (Optional Feature)

Graph RAG is an optional feature that coexists with the existing vector-based RAG. Both systems can operate independently or complement each other.

**When to use Graph RAG:**
- Complex knowledge with many interconnected entities and relationships
- Need for explicit entity/relationship extraction and discovery
- Temporal awareness of facts and relationships
- Natural language understanding of "who", "what", "when" relationships

**Components:**
- `GraphRAGClient` – Independent async client for knowledge graph operations
- Neo4j backend – Graph database for storing entities and relationships
- Graphiti library – LLM-driven entity/relationship extraction
- No breaking changes to existing RAGClient API

**Usage pattern:**
```python
# Vector RAG (existing) - unchanged
rag_client = RAGClient(RAGConfig.from_env())

# Graph RAG (new) - separate, optional
graph_client = GraphRAGClient()
await graph_client.initialize()

# Both can coexist and be used independently or together
```

See `Graph RAG Architecture` section below for details.

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

- **`graph_rag/`** *(NEW)* – Graph RAG module (independent, optional)
  - `client.py` – `GraphRAGClient` class for knowledge graph operations
  - `graph_builder.py` – Converts documents to graph episodes, manages entity/relationship extraction
  - `graph_retriever.py` – Hybrid search on knowledge graph (semantic + BM25)
  - `neo4j_driver.py` – Neo4j connection wrapper using Graphiti
  - `models.py` – Data models for graph nodes, edges, and retrieval results

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

## Graph RAG Architecture

### Graph RAG Pipeline Flow
```
Documents → Extract Content → Graphiti Processing → Neo4j Graph
     ↓            ↓                  ↓                  ↓
  Input    Text/PDF/URL      Entity Extraction    Knowledge Graph
                                  ↓
                            Relationship Extraction
                                  ↓
                        Hybrid Search (Semantic + BM25)
                                  ↓
                         Retrieved Facts & Entities
```

### Key Components

1. **GraphRAGClient** – Async-based client (independent from RAGClient)
   - Manages Neo4j connection lifecycle
   - Orchestrates document ingestion and retrieval
   - Supports context manager pattern (`async with`)

2. **GraphBuilder** – Knowledge graph construction
   - Converts documents to Graphiti episodes
   - Triggers LLM-based entity/relationship extraction
   - Maintains group-based organization of graph data

3. **GraphRetriever** – Knowledge graph search
   - Hybrid search combining semantic and BM25
   - Optional distance-based reranking from center nodes
   - Filters by entity labels and temporal validity

4. **Neo4jGraphDriver** – Connection management
   - Wraps Graphiti client for Neo4j interaction
   - Handles indices and constraint creation
   - Manages connection lifecycle

5. **Graphiti** – External library for LLM-driven extraction
   - Automatically extracts entities and relationships
   - Supports temporal awareness (valid_at, invalid_at)
   - Tracks episodes (document sources) for provenance

### Graph RAG vs Vector RAG

| Aspect | Vector RAG | Graph RAG |
|--------|-----------|-----------|
| **Storage** | Qdrant vector DB | Neo4j graph DB |
| **Client** | `RAGClient` | `GraphRAGClient` |
| **Async** | Synchronous | Asynchronous |
| **Retrieval** | Semantic similarity | Fact/relationship queries |
| **Entity Extraction** | Not explicit | LLM-driven, explicit |
| **Metadata** | Chunk-level | Node/edge properties |
| **Use Cases** | General similarity search | Structured knowledge discovery |
| **Initialization** | One-shot in `__init__` | Async with `initialize()` |

### Data Models

**GraphRAGConfig** (in `core/config.py`):
- NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
- GRAPHITI_LLM_MODEL, GRAPHITI_EMBEDDING_MODEL
- Group ID for organizing graph data

**GraphNode** (in `graph_rag/models.py`):
- UUID, name, labels, summary, properties
- Group ID for collection management

**GraphEdge** (in `graph_rag/models.py`):
- UUID, source/target node UUIDs, fact, relationship type
- Score, validity timestamps, properties

**GraphRetrievalResult**:
- List of edges, nodes, total count, query, timing metrics

### Important Limitations & Notes

1. **Async-only** – GraphRAGClient uses async/await exclusively
2. **No collection deletion** – Neo4j doesn't support collection-level deletion via Graphiti; use raw Neo4j queries if needed
3. **Episode-based organization** – Graph data is organized as episodes (document sources); explicit management required
4. **LLM cost** – Entity extraction uses LLM calls; consider batching for large documents
5. **Phase 1 feature** – Hybrid retrieval (merging graph + vector results) is planned for Phase 2

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

### Vector RAG (Existing)
```dotenv
# Vector Store (Qdrant)
QDRANT_URL=https://your-qdrant:6333
QDRANT_API_KEY=...

# Embedding Model (OpenAI)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# Optional: MongoDB for content storage
MONGO_CONNECTION_STRING=mongodb://...

# Retrieval Options
INSTA_RAG_HYBRID_ENABLED=true
INSTA_RAG_BM25_WEIGHT=0.35
INSTA_RAG_VECTOR_WEIGHT=0.65

# HyDE Query Transformation
INSTA_RAG_HYDE_ENABLED=true
AZURE_LLM_DEPLOYMENT=gpt-4

# Reranking (optional)
BGE_RERANKER_API_KEY=...
BGE_RERANKER_URL=https://api.novita.ai/openai/v1/rerank
```

### Graph RAG (New, Optional)
```dotenv
# Neo4j Graph Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=insta_rag_graph

# Graphiti Configuration (for entity extraction)
GRAPHITI_LLM_MODEL=gpt-4
GRAPHITI_EMBEDDING_MODEL=text-embedding-3-large
GRAPHITI_GROUP_ID=insta_rag
```

See `src/insta_rag/core/config.py` for full list of config options.

## Important Files to Know

- **`pyproject.toml`** – Dependencies, build config, tool settings (ruff, pytest, commitizen)
- **`.pre-commit-config.yaml`** – Pre-commit hook definitions
- **`CONTRIBUTING.md`** – Full contributor guide with setup details
- **`README.md`** – User-facing documentation with examples
- **`.env`** – Runtime configuration (excluded from git)
- **`uv.lock`** – Locked dependency versions (commit this)

## Graph RAG Usage Examples

### Basic Graph RAG Usage
```python
import asyncio
from insta_rag.graph_rag import GraphRAGClient
from insta_rag import DocumentInput

async def main():
    # Initialize (async only)
    client = GraphRAGClient()
    await client.initialize()

    try:
        # Add documents to knowledge graph
        docs = [
            DocumentInput.from_text("Alice works at TechCorp as an engineer"),
            DocumentInput.from_text("TechCorp builds AI products")
        ]
        results = await client.add_documents(docs, collection_name="company")

        # Retrieve from knowledge graph
        result = await client.retrieve(
            query="Who works at TechCorp?",
            collection_name="company",
            k=10
        )
        for fact in result.edges:
            print(f"Fact: {fact.fact}")

    finally:
        await client.close()

asyncio.run(main())
```

### Using as Context Manager
```python
async with GraphRAGClient() as client:
    results = await client.add_documents(docs, "collection")
    retrieval = await client.retrieve("query", "collection", k=5)
```

### Combined Usage (Graph + Vector RAG)
```python
# Both clients can coexist
vector_client = RAGClient(RAGConfig.from_env())  # Sync
async with GraphRAGClient() as graph_client:  # Async
    # Use both independently - no conflicts
    vector_result = vector_client.retrieve(query, k=10)
    graph_result = await graph_client.retrieve(query, k=10)
```

See `examples/graph_rag_usage.py` for comprehensive examples.

## Testing Notes

- Tests are in `tests/` directory
- Uses pytest with fixtures defined in `conftest.py`
- Loads `.env.test` for test-specific configuration
- Smoke tests in `smoke_test.py` validate basic RAGClient initialization
- Graph RAG tests in `test_graph_rag.py` test async operations with mocks
- Use `pytest-mock` for mocking external services (OpenAI, Qdrant, Neo4j)

### Testing Graph RAG
- All GraphRAGClient tests use `@pytest.mark.asyncio` decorator
- Mock Neo4jGraphDriver.initialize() for unit tests (avoids Neo4j dependency)
- Integration tests can use running Neo4j instance if available
- Graphiti client is mocked in most tests to avoid LLM calls

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

### Implemented Features
- ✅ **Vector RAG** – Semantic chunking, hybrid retrieval, HyDE, reranking
- ✅ **Graph RAG** – Knowledge graph-based retrieval with Neo4j and Graphiti (Phase 1)
- ✅ **Hybrid Storage** – Qdrant for vectors, MongoDB for content

### Roadmap (Phase 2+)
- Graph RAG scoring – Implement semantic similarity + BM25 scoring for edges
- Hybrid retrieval merging – Combine graph + vector results intelligently
- Built-in summarization – Answer synthesis from retrieved chunks
- More rerankers – Additional SOTA reranker integrations
- CLI GA – General availability of command-line interface
- LangChain/LlamaIndex adapters – Deep integration with popular frameworks
- Streaming/tracing hooks – OpenTelemetry support, real-time monitoring

### Known Limitations

#### Vector RAG
- Reranking is optional but recommended for long-tail queries
- Vector DB integration focuses on Qdrant (extensible for others)

#### Graph RAG
- **Phase 1 limitation** – Edge scoring currently returns 0.0; Phase 2 will implement semantic similarity + BM25 scoring
- **Async-only** – GraphRAGClient must be used with async/await
- **No collection deletion** – Neo4j doesn't support collection-level deletion via Graphiti API
- **Episode-based organization** – Graph data organized by document source; requires explicit management
- **LLM cost** – Entity extraction incurs LLM API calls; consider batching large documents

### General
- Currently beta status – API may change before v1.0
- Schema RAG (pure graph-only, no vector components) not yet supported
