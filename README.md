# Insta RAG

> **Build production‑grade Retrieval‑Augmented Generation in minutes — not months.**
>
> Plug‑and‑play RAG that you configure, not hand‑wire.

<p align="center">
  <a href="https://pypi.org/project/insta-rag/"><img src="https://img.shields.io/pypi/v/insta-rag.svg" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/status-beta-FF9F1C.svg" alt="Beta">
</p>

Insta RAG (a.k.a. **insta_rag**) is a **modular, configuration‑driven** Python library for building **advanced RAG pipelines**. It abstracts document processing, embedding, and hybrid retrieval behind a clean client so you can ship faster — and tune later.

- **Semantic Chunking** → splits docs on topic boundaries to preserve context.
- **Hybrid Retrieval** → semantic vectors + BM25 keyword search.
- **HyDE Query Transform** → synthesizes hypothetical answers to improve recall.
- **Reranking** → optional integration with SOTA rerankers (e.g., Cohere) to reorder results.
- **Pluggable by Design** → swap chunkers, embedders, rerankers, and vector DBs.
- **Hybrid Storage** → keep **Qdrant** lean for vectors and use **MongoDB** for cheap, flexible content storage.
- **Graph RAG** *(NEW)* → Knowledge graph-based retrieval using **Neo4j** and **Graphiti** for structured entity/relationship extraction and discovery.

---

## Contents

- [Why Insta RAG](#why-insta-rag)
- [Quick Start](#quick-start)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Core API](#core-api)
- [Convenience "Rack" API](#convenience-rack-api)
- [Decorators (syntactic sugar)](#decorators-syntactic-sugar)
- [Advanced Retrieval Recipes](#advanced-retrieval-recipes)
- [Graph RAG (NEW)](#graph-rag-new)
- [FastAPI Example](#fastapi-example)
- [CLI (preview)](#cli-preview)
- [Guides & Docs](#guides--docs)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why Insta RAG

Most RAG stacks feel like soldering a radio: a tangle of chunkers, embedders, retrievers, rerankers, and caches. **Insta RAG makes it a plug‑and‑play client.** Configure once, swap pieces at will, and keep the door open for the latest techniques.

```
┌──────────┐   ┌────────┐   ┌──────────┐   ┌───────────┐   ┌────────┐
│ Documents├─▶│Chunking │─▶│ Embedding│─▶│ Retrieval  │─▶│ Rerank │─▶ Results
└──────────┘   └────────┘   └──────────┘   └───────────┘   └────────┘
                     ^             ^               ^
                  pluggable     pluggable       pluggable
```

---

## Quick Start

### 1) Install

```bash
# Recommended: using uv
uv pip install insta-rag

# Or with pip
pip install insta-rag
```

### 2) Minimal example

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Load configuration from environment variables (.env supported)
config = RAGConfig.from_env()
client = RAGClient(config)

# 1) Add documents to a collection
client.add_documents(
    [DocumentInput.from_text("Your first document content.")],
    collection_name="my_docs",
)

# 2) Retrieve relevant information
resp = client.retrieve(
    query="What is this document about?",
    collection_name="my_docs",
)

# Print the top chunk
if resp.chunks:
    print(resp.chunks[0].content)
```

> **Tip:** Start simple. You can turn on HyDE, hybrid retrieval, and reranking later via config.

---

## Concepts

- **Collection**: named corpus (e.g., `"my_docs"`).
- **Chunker**: splits raw docs into semantically coherent chunks.
- **Embedder**: turns chunks into vectors for semantic lookup.
- **Retriever**: finds candidates using vector search, BM25, or both.
- **Reranker**: reorders candidates using a cross‑encoder (optional).
- **Rack**: shorthand in this project for your **knowledge base**.

---

## Configuration

Declare your stack in a `.env` or environment variables. Common options:

```dotenv
# Vector store
INSTA_RAG_QDRANT_URL=https://your-qdrant:6333
INSTA_RAG_QDRANT_API_KEY=...

# Hybrid storage (optional)
INSTA_RAG_MONGODB_URI=mongodb+srv://...
INSTA_RAG_MONGODB_DB=insta_rag

# Embeddings / LLMs
INSTA_RAG_EMBED_MODEL=text-embedding-3-large
OPENAI_API_KEY=...

# HyDE
INSTA_RAG_HYDE_ENABLED=true
INSTA_RAG_HYDE_MODEL=gpt-4o-mini

# Hybrid retrieval
INSTA_RAG_HYBRID_ENABLED=true
INSTA_RAG_BM25_WEIGHT=0.35
INSTA_RAG_VECTOR_WEIGHT=0.65

# Reranking (optional)
INSTA_RAG_RERANKER=cohere-rerank-3
COHERE_API_KEY=...

# Other
INSTA_RAG_DEFAULT_COLLECTION=my_docs
```

> See **[Guides & Docs](#guides--docs)** for a full catalog of settings.

---

## Core API

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

config = RAGConfig.from_env()
client = RAGClient(config)

# Add
docs = [
    DocumentInput.from_text(
        "Payments: To get a refund, contact support within 30 days.",
        metadata={"source": "faq.md"},
    ),
]
client.add_documents(docs, collection_name="my_docs")

# Retrieve
resp = client.retrieve(
    query="How do I get a refund?",
    collection_name="my_docs",
    k=8,                       # number of candidates
    use_hyde=True,             # HyDE query transformation
    use_hybrid=True,           # BM25 + vectors
    rerank=True,               # apply reranker if configured
)

for ch in resp.chunks:
    print(f"score={ch.score:.3f}", ch.content[:80])
```

---

## Convenience “Rack” API

For teams that want **ultra‑simple, CRUD‑style operations** on the knowledge base, Insta RAG ships a tiny convenience layer that wraps the core client methods. (It’s sugar; you can ignore it.)

```python
from insta_rag import RAGClient, RAGConfig
from insta_rag.rack import Rack   # sugar over client.add/update/remove

client = RAGClient(RAGConfig.from_env())
rack = Rack(client, collection="my_docs")

# Push (create)
rack.push(
    id="doc-1",
    text="Return policy: 30‑day refunds via support@acme.com",
    metadata={"source": "policy.pdf", "lang": "en"},
)

# Update (replace text)
rack.update(id="doc-1", text="Return policy updated: 45 days.")

# Remove
rack.remove(id="doc-1")

# Ask (retrieve only; you format the answer)
chunks = rack.ask("What is the return window?", k=5)
print(chunks[0].content)
```

---

## Decorators (syntactic sugar)

Prefer functions over boilerplate? Use decorators to **bind a collection** and **configure retrieval** at the call site. These live in `insta_rag.decorators` and are **optional**.

```python
from insta_rag import RAGClient, RAGConfig
from insta_rag.decorators import rack, use_retrieval

client = RAGClient(RAGConfig.from_env())

@rack(client, collection="my_docs")         # binds the knowledge base
@use_retrieval(hyde=True, hybrid=True, k=8, rerank=True)
def top_chunk(query, retrieve):
    """retrieve is injected: chunks = retrieve(query)"""
    chunks = retrieve(query)
    return chunks[0]

best = top_chunk("Summarize the refund policy")
print(best.content)
```

> The decorator layer is intentionally thin so you can remove it without touching your business logic.

---

## Advanced Retrieval Recipes

### 1) Metadata filtering
```python
resp = client.retrieve(
    query="refunds",
    collection_name="my_docs",
    filters={"lang": "en", "source": {"$in": ["policy.pdf", "faq.md"]}},
)
```

### 2) Balanced hybrid retrieval
```python
resp = client.retrieve(
    query="PCI requirements for card storage",
    collection_name="my_docs",
    use_hybrid=True,
    bm25_weight=0.5,
    vector_weight=0.5,
)
```

### 3) HyDE + rerank for long‑tail questions
```python
resp = client.retrieve(
    query="Could I still cancel after partial shipment?",
    collection_name="my_docs",
    use_hyde=True,
    rerank=True,
    k=12,
)
```

---

## Graph RAG (NEW)

**Graph RAG** extracts entities and relationships from documents to build a **knowledge graph** in Neo4j. Perfect for discovering connections, understanding context, and answering relationship-based questions.

### When to Use Graph RAG

- Complex knowledge with many interconnected entities (e.g., organizations, people, locations)
- Need explicit entity/relationship extraction and discovery
- Temporal awareness (when facts became relevant or expired)
- Natural language queries like "Who works at X?" or "What are Y's relationships?"

### Quick Start with Graph RAG

#### 1) Configure Neo4j

Add to `.env`:

```dotenv
# Neo4j Graph Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

# Graphiti (Entity Extraction)
GRAPHITI_LLM_MODEL=gpt-4.1
GRAPHITI_EMBEDDING_MODEL=text-embedding-3-large
```

#### 2) Basic Usage

```python
import asyncio
from insta_rag.graph_rag import GraphRAGClient
from insta_rag import DocumentInput

async def main():
    # Initialize Graph RAG (async-only)
    client = GraphRAGClient()
    await client.initialize()

    try:
        # Add documents and extract entities/relationships
        docs = [
            DocumentInput.from_text("Alice works at TechCorp as a Senior Engineer"),
            DocumentInput.from_text("TechCorp builds AI products for enterprises"),
        ]

        results = await client.add_documents(
            docs,
            collection_name="company_knowledge"
        )
        print(f"✓ Extracted {results[0].nodes_created} entities, {results[0].edges_created} relationships")

        # Query the knowledge graph
        retrieval = await client.retrieve(
            query="Who works at TechCorp?",
            collection_name="company_knowledge",
            k=10
        )

        print(f"\n📊 Found {len(retrieval.edges)} facts:")
        for fact in retrieval.edges:
            print(f"  • {fact.fact}")

    finally:
        await client.close()

asyncio.run(main())
```

#### 3) Using as Context Manager

```python
async with GraphRAGClient() as client:
    # Automatically handles initialization and cleanup
    results = await client.add_documents(docs, "knowledge")
    retrieval = await client.retrieve("query", "knowledge", k=5)
```

#### 4) Combining Vector RAG + Graph RAG

```python
from insta_rag import RAGClient, RAGConfig
from insta_rag.graph_rag import GraphRAGClient

async def hybrid_retrieval():
    # Both systems can coexist independently
    vector_client = RAGClient(RAGConfig.from_env())  # Sync

    async with GraphRAGClient() as graph_client:  # Async
        # Vector search for semantic similarity
        vector_results = vector_client.retrieve(
            query="AI products",
            collection_name="docs",
            k=10
        )

        # Graph search for relationships
        graph_results = await graph_client.retrieve(
            query="Who works at TechCorp?",
            collection_name="company",
            k=10
        )

        # Combine insights
        print(f"Vector results: {len(vector_results.chunks)} chunks")
        print(f"Graph results: {len(graph_results.edges)} facts")
```

### Graph RAG API Reference

| Method | Purpose |
|--------|---------|
| `await client.initialize()` | Connect to Neo4j and setup indices |
| `await client.add_documents(docs, collection_name)` | Extract entities/relationships and add to graph |
| `await client.retrieve(query, collection_name, k)` | Search graph using hybrid semantic + BM25 |
| `await client.retrieve_with_reranking(query, collection_name, center_node)` | Retrieve with distance-based reranking from center node |
| `await client.get_entity_context(entity_name, collection_name, depth)` | Get entity and related facts (up to depth levels) |
| `await client.close()` | Cleanup Neo4j connection |

### Graph RAG vs Vector RAG

| Aspect | Vector RAG | Graph RAG |
|--------|-----------|-----------|
| **Storage** | Qdrant vectors | Neo4j graph database |
| **Client** | `RAGClient` (sync) | `GraphRAGClient` (async) |
| **Retrieval** | Semantic similarity | Fact/relationship queries |
| **Entity Extraction** | Not explicit | LLM-driven, explicit |
| **Use Cases** | General similarity search | Structured knowledge discovery |
| **Best For** | Content search | Relationship queries |

---

## FastAPI Example

```python
from fastapi import FastAPI, Query
from insta_rag import RAGClient, RAGConfig

app = FastAPI()
rag = RAGClient(RAGConfig.from_env())

@app.get("/ask")
async def ask(query: str = Query(...), collection: str = "my_docs"):
    resp = rag.retrieve(query=query, collection_name=collection, use_hyde=True, use_hybrid=True, rerank=True)
    return {
        "matches": [
            {"score": ch.score, "content": ch.content, "metadata": ch.metadata}
            for ch in resp.chunks
        ]
    }
```

---

## CLI (preview)

> Optional add‑on for simple ops. Install with `pip install insta-rag[cli]`.

```bash
# Ingest
insta-rag add --collection my_docs ./data/*.pdf

# Update by id
insta-rag update --collection my_docs --id doc-1 --file updated.txt

# Remove by id
insta-rag remove --collection my_docs --id doc-1

# Ask (JSON response)
insta-rag ask --collection my_docs --query "What is the refund window?"
```

---

## Guides & Docs

- **Installation Guide** – Python versions, optional extras, uv vs pip
- **Quickstart** – end‑to‑end in 5 minutes
- **Document Management** – ingestion patterns, chunking strategies
- **Advanced Retrieval** – hybrid knobs, HyDE, reranking, filters
- **Storage Backends** – Qdrant setup, MongoDB sizing tips

> Looking for something specific? See the **Full Documentation** (link your site here).

---

## Contributing

We welcome contributions! Please check out the **Contributing Guide** for:

- Dev environment setup (`uv`, `poetry`, or `pip`)
- Code quality: `ruff`, `black`, `mypy`, `pytest`, `pre-commit`
- Commit conventions: Conventional Commits
- Branching model: `main` (stable) / `develop` (active)
- Versioning: SemVer
- PR checklist & CI matrix

---

## Roadmap

### Implemented
- [x] **Graph RAG** – Knowledge graph-based retrieval with Neo4j and Graphiti
- [x] **Hybrid Storage** – Qdrant vectors + MongoDB content
- [x] **Hybrid Retrieval** – Semantic + BM25 search
- [x] **HyDE & Reranking** – Query transformation and SOTA reranking

### Coming Soon (Phase 2+)
- [ ] Graph RAG Scoring – Semantic similarity + BM25 for edges
- [ ] Built‑in summarization & answer synthesis helpers
- [ ] More rerankers (open‑source options)
- [ ] CLI GA
- [ ] LangChain/LlamaIndex adapters
- [ ] Streaming & tracing hooks (OpenTelemetry)
- [ ] Native PDF/HTML loaders with auto‑chunk profiles

---




## Documentation

For detailed guides on installation, configuration, and advanced features, please see the **[Full Documentation](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki)**.

Key sections include:

- **[Installation Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/installation)**
- **[Quickstart Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/quickstart)**
- **Guides**
  - [Document Management](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/document-management)
  - [Advanced Retrieval](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/retrieval)
  - [Storage Backends](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki/guides/storage-backends)

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/CONTRIBUTING.md) for details on:

- Setting up your development environment
- Code quality tools and pre-commit hooks
- Commit and branch naming conventions
- Version management
- Pull request process

## License

This project is licensed under the [MIT License](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/blob/main/LICENSE).



### Shout‑outs

Insta RAG packages the **most effective, modern RAG techniques** into a clean DX. You focus on your product; we keep the rack updated as the ecosystem evolves.
lets rock

