## 0.1.0-beta.0 (2025-10-20)

### Features

- **Initial Release of `insta_rag` library**:
  - Introduced a modular, plug-and-play Python library for building advanced Retrieval-Augmented Generation (RAG) pipelines.
  - Core features include:
    - Semantic Chunking
    - Hybrid Retrieval (Vector Search + Keyword Search)
    - Query Transformation (HyDE)
    - Reranking with Cohere
    - Pluggable architecture for chunkers, embedders, and vector databases.
    - Hybrid storage with Qdrant and MongoDB.

## v0.1.1-beta.6 (2025-11-25)

## v0.1.1-beta.5 (2025-11-21)

## v0.1.1-beta.4 (2025-11-20)

## v0.1.1-beta.3 (2025-10-25)

## v0.1.1-beta.2 (2025-10-24)

## v0.1.1-beta.0 (2025-10-21)

### Fix

- resolve import path errors causing module load failures (#14)
- update import paths for exceptions to use the correct module

### Refactor

- clean up code formatting and improve readability across multiple files

## v0.1.0-beta.2 (2025-10-20)

### Feat

- add docs submodule for project documentation

### Refactor

- revert author metadata to multiple authors

## v0.1.0-beta.1 (2025-10-20)

### Feat

- add pre-bump hooks for dependency management
- add additional metadata to pyproject.toml

### Fix

- correct author metadata for pypi

## v0.1.0-beta.0 (2025-10-20)

### Feat

- add GitHub Actions workflow for publishing to PyPI
- add integration and smoke tests for RAGClient with environment variable checks
- initial release of insta_rag library with comprehensive documentation and structure updates (#8)

### Fix

- update tag format in commitizen configuration to include 'v' prefix
- update version retrieval to use importlib.metadata for dynamic versioning
- add license files and update classifiers in pyproject.toml

### Refactor

- update project structure and clean up unused files (#5)
- move utility scripts into utils directory
- update exception imports to use utils.exceptions module
- remove unused ssl import in QdrantVectorDB
- move documentation into docs directory
