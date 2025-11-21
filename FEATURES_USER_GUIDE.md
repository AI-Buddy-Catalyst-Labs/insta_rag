# Insta RAG - User Guide

> **Version:** 0.1.1-beta.4 | **Last Updated:** November 2025

---

## What is Insta RAG?

**Insta RAG** is a powerful library that helps you build smart document search systems. Instead of simple keyword matching, Insta RAG understands the *meaning* behind your documents and questions, delivering more relevant results.

**Think of it like this:** Traditional search is like looking for a word in a dictionary. Insta RAG is like having a research assistant who reads all your documents, understands them, and finds exactly what you need - even if you don't use the exact same words.

---

## Who Is This For?

- **Developers** building AI-powered search features
- **Companies** needing intelligent document retrieval for internal knowledge bases
- **AI/ML Engineers** implementing RAG (Retrieval-Augmented Generation) pipelines
- **Product Teams** adding "search your documents" features to applications

---

## Key Features at a Glance

| Feature | What It Does | Why It Matters |
|---------|--------------|----------------|
| **Smart Document Chunking** | Automatically splits documents into meaningful sections | Keeps context intact, better search results |
| **Hybrid Search** | Combines meaning-based + keyword-based search | Finds both similar concepts AND exact terms |
| **HyDE Query Enhancement** | Improves your search questions automatically | Up to 30% better retrieval accuracy |
| **Intelligent Reranking** | Re-sorts results by relevance | Top results are truly the most relevant |
| **Knowledge Graphs** | Maps relationships between concepts | "Who works at TechCorp?" type queries |
| **Multiple Document Types** | Handles PDF, TXT, MD, and raw text | Works with your existing documents |

---

## Feature Details

### 1. Intelligent Document Processing

**What you get:**
- Upload documents in multiple formats (PDF, text files, markdown)
- Documents are automatically split into smart chunks that preserve meaning
- Each chunk is analyzed and indexed for fast retrieval

**Supported Input Types:**
- `.pdf` - PDF documents (text extraction included)
- `.txt` - Plain text files
- `.md` - Markdown files
- Raw text strings (programmatic input)

**How it works for you:**
```
Your Document --> Smart Chunking --> Meaning Analysis --> Ready for Search
     |                  |                   |
   "Upload"      "Auto-split by          "Each chunk
                  topic boundaries"       understood"
```

---

### 2. Hybrid Search (Best of Both Worlds)

**The problem with traditional search:**
- Keyword search misses synonyms ("car" won't find "automobile")
- Pure AI search might miss exact terms you're looking for

**How Insta RAG solves this:**
Combines **semantic search** (understanding meaning) with **BM25 keyword search** (finding exact terms), then merges the results intelligently.

**Real-world example:**

| Query | Keyword-Only Result | Semantic-Only Result | Hybrid Result (Insta RAG) |
|-------|---------------------|----------------------|---------------------------|
| "employee benefits policy" | Documents with exact phrase | Documents about "staff compensation packages" | Both! Ranked by relevance |

**Configurable weights:** You control the balance (default: 65% semantic, 35% keyword)

---

### 3. HyDE Query Enhancement

**What is HyDE?**
HyDE (Hypothetical Document Embeddings) automatically improves your search queries by generating a "hypothetical answer" and using that to find better matches.

**Example:**

| Your Query | What Insta RAG Does | Why It Helps |
|------------|---------------------|--------------|
| "How do I reset password?" | Generates: "To reset your password, go to Settings > Security > Reset Password..." | The hypothetical answer is more likely to match your actual documentation |

**Result:** Research shows 20-30% improvement in retrieval quality.

---

### 4. Intelligent Reranking

**The problem:** Initial search might return 50 relevant results, but which 10 are MOST relevant?

**The solution:** After initial search, Insta RAG uses advanced AI models to re-score and re-order results.

**Supported Rerankers:**
| Reranker | Provider | Strength |
|----------|----------|----------|
| BGE Reranker v2 | Novita AI | Fast, accurate, cost-effective |
| Cohere Rerank | Cohere | Enterprise-grade accuracy |
| LLM Fallback | Any OpenAI-compatible | Automatic backup if primary fails |

**Automatic fallback:** If the primary reranker is unavailable, Insta RAG automatically switches to backup.

---

### 5. Knowledge Graph RAG (Advanced)

**What it adds:**
Instead of just finding relevant text chunks, Knowledge Graph RAG understands **entities** (people, places, things) and **relationships** between them.

**Best for queries like:**
- "Who works at TechCorp?"
- "What products does Company X make?"
- "When did Alice join the engineering team?"

**How it works:**

```
Documents --> Entity Extraction --> Relationship Mapping --> Graph Database
                 |                        |
           "Alice, TechCorp,         "Alice WORKS_AT TechCorp"
            Engineer"                "TechCorp BUILDS AI_Products"
```

**Query result:**
```
Question: "Who works at TechCorp?"
Answer: "Alice works at TechCorp as an engineer."
```

---

### 6. Document Management

**Full CRUD operations:**

| Operation | What You Can Do |
|-----------|-----------------|
| **Add** | Upload new documents to any collection |
| **Update** | Replace documents, update metadata, or append new content |
| **Delete** | Remove specific documents or filter by criteria |
| **Upsert** | Automatically update if exists, insert if new |

**Flexible organization:** Group documents into "collections" (like folders) for different use cases.

---

## Common Use Cases

### Use Case 1: Internal Knowledge Base
**Scenario:** Your company has hundreds of policy documents, FAQs, and guides.

**With Insta RAG:**
- Upload all documents once
- Employees search in natural language: "What's the vacation policy for remote workers?"
- System returns the most relevant sections, not just documents with matching keywords

### Use Case 2: Customer Support AI
**Scenario:** Build an AI chatbot that answers customer questions using your documentation.

**With Insta RAG:**
- Index product manuals, FAQs, troubleshooting guides
- When customer asks a question, retrieve the best context
- Pass context to your LLM (GPT-4, Claude, etc.) for a helpful answer

### Use Case 3: Research Document Analysis
**Scenario:** Analyze large collections of research papers or reports.

**With Insta RAG:**
- Index thousands of documents
- Find connections across papers using Knowledge Graph
- Query: "What papers discuss neural networks AND climate modeling?"

---

## Getting Started (Quick Overview)

### Step 1: Install
```bash
pip install insta-rag
```

### Step 2: Configure
Set your API keys for:
- **Qdrant** (vector database)
- **OpenAI** or **Azure OpenAI** (embeddings & LLM)
- **Novita AI** (optional, for BGE reranking)

### Step 3: Use
```python
from insta_rag import RAGClient, RAGConfig, DocumentInput

# Initialize
client = RAGClient(RAGConfig.from_env())

# Add documents
docs = [DocumentInput.from_file("policy.pdf")]
client.add_documents(docs, collection_name="company_policies")

# Search
results = client.retrieve("What is the vacation policy?", collection_name="company_policies")
```

---

## FAQ

**Q: How accurate is the search?**
A: With HyDE and reranking enabled, users report 20-30% better results than keyword search alone.

**Q: What document sizes can it handle?**
A: Documents are automatically chunked, so there's no practical size limit. Large documents are split into ~1000 token chunks.

**Q: Can I use my existing Qdrant database?**
A: Yes! Insta RAG works with any Qdrant instance (cloud or self-hosted).

**Q: What about sensitive documents?**
A: Insta RAG doesn't store documents externally by default. You control where data lives.

**Q: Is there a UI?**
A: Insta RAG is a Python library (backend). You integrate it into your own applications.

---

## Support & Resources

- **Documentation:** [GitHub Wiki](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/wiki)
- **Issues:** [GitHub Issues](https://github.com/AI-Buddy-Catalyst-Labs/insta_rag/issues)
- **License:** MIT (free for commercial use)

---

*Insta RAG - Ship RAG applications faster.*
