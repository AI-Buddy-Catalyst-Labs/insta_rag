# Insta RAG - Technical Architecture & Features

> **Version:** 0.1.1-beta.4 | **Python:** 3.10+ | **Status:** Beta

---

## Executive Summary

**Insta RAG** is a modular, configuration-driven Python library implementing state-of-the-art RAG (Retrieval-Augmented Generation) patterns. It provides production-ready abstractions for document processing, embedding, hybrid retrieval, and knowledge graph construction.

### Key Technical Differentiators

| Feature | Implementation | Competitive Advantage |
|---------|----------------|----------------------|
| Semantic Chunking | Embedding-based topic boundary detection | Preserves context vs. naive fixed-size splitting |
| Hybrid Retrieval | Vector search + BM25 fusion with configurable weights | Higher recall than single-method approaches |
| HyDE Query Transform | LLM-generated hypothetical documents | 20-30% retrieval improvement (research-backed) |
| Multi-stage Reranking | BGE/Cohere rerankers with LLM fallback | Resilient ranking pipeline |
| Knowledge Graph RAG | Neo4j + Graphiti entity extraction | Structured entity/relationship queries |

---

## Architecture Overview

### System Architecture Diagram

```
                                    INSTA RAG ARCHITECTURE
    =====================================================================================

    +------------------+     +-------------------+     +------------------+
    |   INPUT LAYER    |     |  PROCESSING LAYER |     |   STORAGE LAYER  |
    +------------------+     +-------------------+     +------------------+
    |                  |     |                   |     |                  |
    | DocumentInput    |     | SemanticChunker   |     | Qdrant VectorDB  |
    |  - from_file()   |---->|  - topic boundary |---->|  - upsert()      |
    |  - from_text()   |     |  - overlap mgmt   |     |  - search()      |
    |  - from_binary() |     |                   |     |  - delete()      |
    |                  |     | OpenAIEmbedder    |     |                  |
    | Supported:       |     |  - batch embed    |     | Neo4j GraphDB    |
    |  - PDF           |     |  - 3072D vectors  |     |  - entities      |
    |  - TXT/MD        |     |                   |     |  - relationships |
    |  - Raw text      |     | HyDEQueryGen      |     |                  |
    +------------------+     |  - hypothetical   |     | MongoDB (opt)    |
                             |    doc generation |     |  - content store |
    +------------------+     +-------------------+     +------------------+
    | RETRIEVAL LAYER  |
    +------------------+
    |                  |
    | VectorSearch     |     +-------------------+
    |  - cosine sim    |     |   RERANKING LAYER |
    |  - metadata filt |     +-------------------+
    |                  |     |                   |
    | BM25Searcher     |---->| BGEReranker       |---->  Final Results
    |  - keyword match |     |  - Novita AI API  |      (top-k chunks)
    |  - TF-IDF scoring|     |                   |
    |                  |     | CohereReranker    |
    | GraphRetriever   |     |  - enterprise alt |
    |  - entity search |     |                   |
    |  - relationship  |     | LLMReranker       |
    |    traversal     |     |  - fallback mode  |
    +------------------+     +-------------------+
```

---

## Module Deep Dive

### 1. Core Module (`src/insta_rag/core/`)

#### `RAGClient` - Main Orchestrator

**File:** `client.py` (1268 lines)

**Responsibilities:**
- Document ingestion pipeline orchestration
- Component initialization and lifecycle management
- Hybrid retrieval coordination

**Key Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_documents()` | `(documents: List[DocumentInput], collection_name: str, metadata: Dict, batch_size: int) -> AddDocumentsResponse` | Full ingestion pipeline: load -> extract -> chunk -> embed -> store |
| `update_documents()` | `(collection_name: str, update_strategy: str, filters: Dict, ...) -> UpdateDocumentsResponse` | CRUD operations with strategies: `replace`, `append`, `delete`, `upsert` |
| `retrieve()` | `(query: str, collection_name: str, top_k: int, ...) -> RetrievalResponse` | Full retrieval pipeline with HyDE + BM25 + reranking |
| `search()` | `(query: str, collection_name: str, top_k: int, filters: Dict) -> RetrievalResponse` | Simple vector search without advanced features |

**Pipeline Flow for `retrieve()`:**
```
1. Query Generation (HyDE)
   - Generate optimized standard query + hypothetical document

2. Dual Vector Search
   - Search with standard query (25 results)
   - Search with HyDE query (25 results)

3. Keyword Search (BM25)
   - BM25Okapi search (50 results)

4. Combine & Deduplicate
   - Merge all results, keep highest score per chunk_id

5. Reranking
   - BGE/Cohere reranker or LLM fallback

6. Selection & Formatting
   - Top-k selection, score threshold filtering
   - Convert to RetrievedChunk objects
```

#### `RAGConfig` - Configuration Management

**File:** `config.py` (449 lines)

**Configuration Hierarchy:**

```python
RAGConfig
+-- VectorDBConfig      # Qdrant connection settings
+-- EmbeddingConfig     # OpenAI/Azure embedding params
+-- RerankingConfig     # BGE/Cohere + LLM fallback settings
+-- LLMConfig           # Query generation LLM settings
+-- ChunkingConfig      # Semantic chunking parameters
+-- PDFConfig           # PDF parser selection
+-- RetrievalConfig     # Search behavior settings
```

**Environment Variable Mapping:**

| Config Class | Required Variables | Optional Variables |
|--------------|-------------------|-------------------|
| `VectorDBConfig` | `QDRANT_URL`, `QDRANT_API_KEY` | - |
| `EmbeddingConfig` | `AZURE_OPENAI_API_KEY` or `OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT`, `AZURE_EMBEDDING_DEPLOYMENT` |
| `RerankingConfig` | `BGE_RERANKER_API_KEY` or `COHERE_API_KEY` | `GPT_OSS_ENDPOINT`, `GPT_OSS_API_KEY` |
| `LLMConfig` | Inherits from embedding | `AZURE_LLM_DEPLOYMENT` |

---

### 2. Chunking Module (`src/insta_rag/chunking/`)

#### `SemanticChunker` - Topic-Aware Document Splitting

**File:** `semantic.py` (299 lines)

**Algorithm:**

```
INPUT: Raw document text

1. Token Check
   - If total_tokens <= max_chunk_size: return single chunk

2. Sentence Splitting
   - Regex-based sentence boundary detection
   - Handles abbreviations, decimals, edge cases

3. Sentence Embedding
   - Batch embed all sentences (OpenAI text-embedding-3-large)

4. Similarity Calculation
   - Cosine similarity between consecutive sentence embeddings
   - similarities[i] = cos_sim(embed[i], embed[i+1])

5. Breakpoint Detection
   - threshold = percentile(similarities, 100 - threshold_percentile)
   - breakpoints = indices where similarity < threshold

6. Chunk Assembly
   - Split sentences at breakpoints
   - Merge adjacent sentences between breakpoints

7. Token Limit Enforcement
   - If chunk > max_chunk_size: recursive token-based split

8. Overlap Addition
   - Add overlap_percentage of previous chunk to each chunk

OUTPUT: List[Chunk] with metadata
```

**Configuration Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_size` | 1000 tokens | Maximum tokens per chunk |
| `min_chunk_size` | 100 tokens | Minimum viable chunk size |
| `overlap_percentage` | 0.2 (20%) | Context overlap between chunks |
| `threshold_percentile` | 95 | Similarity percentile for breakpoints |

**Token Counting:**
- Uses `tiktoken` with `cl100k_base` encoding (GPT-4/3.5 tokenizer)
- Accurate token counting vs. word estimation

---

### 3. Embedding Module (`src/insta_rag/embedding/`)

#### `OpenAIEmbedder` - Embedding Provider

**File:** `openai.py`

**Supported Providers:**
- OpenAI API (standard)
- Azure OpenAI (enterprise)

**Specifications:**

| Model | Dimensions | Max Tokens | Batch Size |
|-------|------------|------------|------------|
| `text-embedding-3-large` | 3072 | 8191 | 100 (configurable) |
| `text-embedding-3-small` | 1536 | 8191 | 100 |

**Methods:**

```python
def embed(texts: List[str]) -> List[List[float]]
    """Batch embed multiple texts with automatic batching."""

def embed_query(query: str) -> List[float]
    """Single query embedding (optimized path)."""

def get_dimensions() -> int
    """Return embedding dimensionality for collection creation."""
```

---

### 4. Retrieval Module (`src/insta_rag/retrieval/`)

#### `HyDEQueryGenerator` - Query Enhancement

**File:** `query_generator.py` (150 lines)

**Theory:**
HyDE (Hypothetical Document Embeddings) improves retrieval by generating a hypothetical document that would answer the query, then using that document's embedding for search. This bridges the lexical gap between queries and documents.

**Reference:** "Precise Zero-Shot Dense Retrieval without Relevance Labels" (Gao et al., 2022)

**Implementation:**

```python
def generate_queries(query: str) -> Dict[str, str]:
    """
    Returns:
        {
            "standard": "optimized query keywords",
            "hyde": "A 2-3 sentence hypothetical answer..."
        }
    """
```

**LLM Prompt Strategy:**
- System prompt defines output format (JSON mode)
- Standard query: keyword expansion, stop word removal, synonym addition
- HyDE query: 2-3 sentence technical answer in target domain language

#### `BM25Searcher` - Keyword Search

**File:** `keyword_search.py` (205 lines)

**Algorithm:** BM25Okapi (Best Matching 25 - Okapi variant)

**Implementation Details:**
- Uses `rank-bm25` library
- Builds in-memory corpus from Qdrant collection
- Supports MongoDB content fetching for external storage mode
- Metadata filtering post-retrieval

**Corpus Building:**
```python
# Scroll through all Qdrant points
# Tokenize: content.lower().split()
# Build BM25Okapi index
```

#### `BGEReranker` - Neural Reranking

**File:** `reranker.py` (305 lines)

**Model:** `BAAI/bge-reranker-v2-m3`

**API Endpoint:** Novita AI (`https://api.novita.ai/openai/v1/rerank`)

**Score Transformation:**
```python
# Novita returns: 0.0 to 1.0
# Internal format: -10.0 to +10.0
transformed_score = (novita_score * 20.0) - 10.0
```

**Fallback Strategy:**

```
Primary: BGE Reranker
    | (on failure)
    v
Fallback: LLM Reranker (gpt-oss-120b or configured)
    | (on failure)
    v
Final Fallback: Vector score sorting
```

---

### 5. Vector Database Module (`src/insta_rag/vectordb/`)

#### `QdrantVectorDB` - Vector Storage

**File:** `qdrant.py` (732 lines)

**Connection Options:**
- HTTPS/HTTP auto-detection from URL
- gRPC support (disabled by default for compatibility)
- SSL verification toggle for self-signed certificates

**Key Operations:**

| Method | Complexity | Description |
|--------|------------|-------------|
| `create_collection()` | O(1) | Create with vector params |
| `upsert()` | O(n) batched | Insert/update with deterministic UUIDs |
| `search()` | O(log n) | ANN search with metadata filtering |
| `delete()` | O(n) | By chunk_ids or filters |
| `delete_by_document_ids()` | O(1) filter-based | Efficient bulk delete |
| `update_metadata()` | O(n) | Metadata-only updates |

**Point ID Generation:**
```python
# Deterministic UUID from chunk_id for idempotent upserts
point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
```

**Content Storage Modes:**
1. **Qdrant Payload** (`store_content=True`): Content in Qdrant payload
2. **External Storage** (`store_content=False`): Metadata only, content in MongoDB

---

### 6. Graph RAG Module (`src/insta_rag/graph_rag/`)

#### `GraphRAGClient` - Knowledge Graph Operations

**File:** `client.py` (317 lines)

**Architecture:**
- **Async-only API** (Python `asyncio`)
- **Independent from RAGClient** (can coexist)
- **Neo4j backend** via Graphiti library

**Dependencies:**
```
GraphRAGClient
+-- Neo4jGraphDriver (connection management)
+-- GraphBuilder (entity extraction)
+-- GraphRetriever (graph search)
```

**Initialization Flow:**

```python
async def initialize(self) -> "GraphRAGClient":
    # 1. Connect to Neo4j via Graphiti
    self._graphiti = await self.driver.initialize()

    # 2. Initialize builder (document -> graph)
    self._builder = GraphBuilder(self._graphiti, self.group_id)

    # 3. Initialize retriever (graph search)
    self._retriever = GraphRetriever(self._graphiti, self.group_id)
```

**Azure OpenAI Integration:**

```python
# Uses Graphiti's native Azure classes
from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
from graphiti_core.embedder.azure_openai import AzureOpenAIEmbedderClient

# Configured via environment:
# AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
# AZURE_LLM_DEPLOYMENT, AZURE_EMBEDDING_DEPLOYMENT
```

#### `GraphBuilder` - Entity Extraction

**File:** `graph_builder.py` (284 lines)

**Graphiti Episode Pipeline:**

```
Document -> add_episode() -> LLM Entity Extraction -> Neo4j Storage
                                      |
                                      v
                            Extracted: nodes[], edges[]
                            Episode metadata, timestamps
```

**Output Model:**

```python
@dataclass
class GraphAddResult:
    episode_uuid: str
    nodes_created: int
    edges_created: int
    group_id: str
    processing_time_ms: float
    extracted_nodes: List[GraphNode]
    extracted_edges: List[GraphEdge]
```

#### `GraphRetriever` - Graph Search

**Retrieval Methods:**

| Method | Use Case |
|--------|----------|
| `retrieve()` | Hybrid semantic + BM25 search on graph |
| `retrieve_with_reranking()` | Distance-based reranking from center node |
| `get_entity_context()` | Get all facts for a specific entity |

---

## Data Models

### Document Processing Models

**File:** `models/document.py`

```python
class SourceType(Enum):
    FILE = "file"      # Path to file
    TEXT = "text"      # Raw string
    BINARY = "binary"  # Bytes (PDF binary)

@dataclass
class DocumentInput:
    source: Union[str, Path, bytes]
    source_type: SourceType
    metadata: Dict[str, Any]
    custom_chunking: Optional[Dict[str, Any]]

    @classmethod
    def from_file(cls, path, metadata=None) -> "DocumentInput"
    @classmethod
    def from_text(cls, text, metadata=None) -> "DocumentInput"
    @classmethod
    def from_binary(cls, content, metadata=None) -> "DocumentInput"
```

### Chunk Model

**File:** `models/chunk.py`

```python
@dataclass
class ChunkMetadata:
    document_id: str
    source: str
    chunk_index: int
    total_chunks: int
    token_count: int
    char_count: int
    chunking_method: str  # "semantic", "semantic_fallback", "semantic_single"
    extraction_date: datetime
    custom_fields: Dict[str, Any]

@dataclass
class Chunk:
    chunk_id: str           # "{document_id}_chunk_{index}"
    content: str            # Actual text content
    metadata: ChunkMetadata
    embedding: Optional[List[float]]  # Populated after embedding
```

### Response Models

**File:** `models/response.py`

```python
@dataclass
class RetrievalResponse:
    success: bool
    query_original: str
    queries_generated: Dict[str, str]  # {"original", "standard", "hyde"}
    chunks: List[RetrievedChunk]
    retrieval_stats: RetrievalStats
    sources: List[SourceInfo]
    errors: List[str]

@dataclass
class RetrievalStats:
    query_generation_time_ms: float
    vector_search_time_ms: float
    keyword_search_time_ms: float
    reranking_time_ms: float
    total_time_ms: float
    vector_search_chunks: int
    keyword_search_chunks: int
    chunks_after_dedup: int
    chunks_after_reranking: int
    total_chunks_retrieved: int
```

---

## Technology Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >= 1.12.0 | Embedding & LLM API client |
| `qdrant-client` | >= 1.7.0 | Vector database client |
| `pdfplumber` | >= 0.10.3 | PDF text extraction (primary) |
| `PyPDF2` | >= 3.0.1 | PDF text extraction (fallback) |
| `tiktoken` | >= 0.5.2 | Token counting (GPT tokenizer) |
| `numpy` | >= 1.24.0 | Numerical operations |
| `pydantic` | >= 2.5.0 | Data validation & serialization |
| `rank-bm25` | >= 0.2.2 | BM25 keyword search |
| `graphiti-core` | >= 0.1.0 | Knowledge graph operations |
| `cohere` | >= 4.47.0 | Reranking API client |
| `requests` | >= 2.32.5 | HTTP client for BGE reranker |

### External Services

| Service | Purpose | Required |
|---------|---------|----------|
| **Qdrant** | Vector storage & search | Yes |
| **OpenAI / Azure OpenAI** | Embeddings & LLM | Yes |
| **Novita AI** | BGE reranking | Optional (recommended) |
| **Cohere** | Alternative reranking | Optional |
| **Neo4j** | Knowledge graph storage | Optional (Graph RAG only) |
| **MongoDB** | Content storage | Optional |

---

## Performance Characteristics

### Benchmarks (Approximate)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Document chunking (1000 tokens) | 50-100ms | Embedding-bound |
| Embedding generation (100 chunks) | 200-500ms | Batch API call |
| Vector search (1M vectors) | 10-50ms | Qdrant HNSW |
| BM25 search (10K documents) | 5-20ms | In-memory index |
| BGE reranking (50 chunks) | 100-300ms | API call |
| Full retrieve pipeline | 500-1500ms | All components |

### Scaling Considerations

1. **Corpus Size:**
   - BM25 index is in-memory (rebuild per request)
   - For large collections (>100K docs), consider caching BM25 index

2. **Concurrent Requests:**
   - RAGClient is not thread-safe by design
   - Use connection pooling for Qdrant client
   - Consider request queuing for reranker API

3. **Cost Optimization:**
   - Batch embeddings (default batch_size=100)
   - Cache HyDE queries for repeated searches
   - Disable reranking for lower-priority queries

---

## Configuration Reference

### Complete Environment Variables

```bash
# === REQUIRED ===
QDRANT_URL=https://your-qdrant-instance:6333
QDRANT_API_KEY=your-qdrant-api-key

# Option A: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_LLM_DEPLOYMENT=gpt-4

# Option B: OpenAI
OPENAI_API_KEY=your-openai-key

# === RECOMMENDED ===
BGE_RERANKER_API_KEY=your-novita-key
BGE_RERANKER_URL=https://api.novita.ai/openai/v1/rerank

# === OPTIONAL ===
# Cohere reranking (alternative to BGE)
COHERE_API_KEY=your-cohere-key

# LLM fallback reranking
GPT_OSS_ENDPOINT=https://your-llm-endpoint
GPT_OSS_API_KEY=your-llm-key
GPT_OSS_MODEL=gpt-oss-120b
GPT_OSS_FALLBACK_ENABLED=true

# Graph RAG (optional feature)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=insta_rag_graph
GRAPHITI_LLM_MODEL=gpt-4
GRAPHITI_EMBEDDING_MODEL=text-embedding-3-large
GRAPHITI_GROUP_ID=insta_rag
```

---

## API Usage Examples

### Basic Usage

```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Initialize from environment
config = RAGConfig.from_env()
client = RAGClient(config)

# Add documents
docs = [
    DocumentInput.from_file("document.pdf", {"category": "policy"}),
    DocumentInput.from_text("Raw text content here", {"category": "faq"}),
]
response = client.add_documents(docs, collection_name="knowledge_base")
print(f"Added {response.total_chunks} chunks")

# Retrieve with full pipeline
result = client.retrieve(
    query="What is the vacation policy?",
    collection_name="knowledge_base",
    top_k=10,
    enable_hyde=True,
    enable_keyword_search=True,
    enable_reranking=True,
)

for chunk in result.chunks:
    print(f"Score: {chunk.relevance_score:.4f}")
    print(f"Content: {chunk.content[:200]}...")
```

### Graph RAG Usage

```python
import asyncio
from insta_rag.graph_rag import GraphRAGClient
from insta_rag import DocumentInput

async def main():
    async with GraphRAGClient() as client:
        # Add to knowledge graph
        docs = [
            DocumentInput.from_text("Alice works at TechCorp as an engineer."),
            DocumentInput.from_text("TechCorp builds AI products."),
        ]
        results = await client.add_documents(docs, "company")

        # Query relationships
        result = await client.retrieve(
            query="Who works at TechCorp?",
            collection_name="company",
            k=10,
        )
        for edge in result.edges:
            print(f"Fact: {edge.fact}")

asyncio.run(main())
```

### Document Updates

```python
# Replace documents matching filter
client.update_documents(
    collection_name="knowledge_base",
    update_strategy="replace",
    filters={"category": "policy"},
    new_documents=[DocumentInput.from_file("new_policy.pdf")],
)

# Upsert (update or insert)
client.update_documents(
    collection_name="knowledge_base",
    update_strategy="upsert",
    new_documents=[
        DocumentInput.from_text("Updated FAQ", {"document_id": "faq-001"}),
    ],
)
```

---

## Development & Quality

### Code Quality Tools

| Tool | Purpose | Config Location |
|------|---------|-----------------|
| **Ruff** | Linting & formatting | `pyproject.toml` |
| **Pre-commit** | Git hook automation | `.pre-commit-config.yaml` |
| **Commitizen** | Conventional commits | `pyproject.toml` |
| **Pytest** | Testing framework | `pyproject.toml` |
| **detect-secrets** | Secret scanning | `.secrets.baseline` |

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src/insta_rag

# Specific module
pytest tests/test_chunking.py -v
```

### CI/CD

- GitHub Actions for automated testing
- Pre-commit hooks enforce code quality
- Semantic versioning via Commitizen

---

## Limitations & Roadmap

### Current Limitations

1. **Graph RAG Scoring:** Edge scores return 0.0 (Phase 2 will implement semantic + BM25 scoring)
2. **BM25 Index:** Rebuilt per request (no caching)
3. **No Streaming:** Retrieval is synchronous
4. **Single Vector DB:** Qdrant only (extensible via `BaseVectorDB`)

### Roadmap

- [ ] Graph + Vector hybrid retrieval merging
- [ ] Answer synthesis from retrieved chunks
- [ ] LangChain/LlamaIndex adapters
- [ ] OpenTelemetry tracing
- [ ] CLI for document management
- [ ] Streaming retrieval API

---

## License & Support

- **License:** MIT
- **Repository:** [github.com/AI-Buddy-Catalyst-Labs/insta_rag](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag)
- **Issues:** [GitHub Issues](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/issues)

---

*Built for production RAG applications.*
