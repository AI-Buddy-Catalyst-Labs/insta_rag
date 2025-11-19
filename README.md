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

---

## Contents

- [Why Insta RAG](#why-insta-rag)
- [Quick Start](#quick-start)
- [Concepts](#concepts)
- [Configuration](#configuration)
- [Core API](#core-api)
- [Convenience “Rack” API](#convenience-rack-api)
- [Decorators (syntactic sugar)](#decorators-syntactic-sugar)
- [Advanced Retrieval Recipes](#advanced-retrieval-recipes)
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

