"""Configuration management for insta_rag library."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from insta_rag.utils.exceptions import ConfigurationError


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    url: str
    api_key: str
    provider: str = "qdrant"
    timeout: int = 30
    prefer_grpc: bool = False  # Changed to False to avoid connection issues
    https: Optional[bool] = None  # Auto-detect from URL if None
    verify_ssl: bool = False  # Set to False for self-signed certificates

    def validate(self) -> None:
        """Validate vector database configuration."""
        if not self.url:
            raise ConfigurationError("Vector database URL is required")
        if not self.api_key:
            raise ConfigurationError("Vector database API key is required")


@dataclass
class EmbeddingConfig:
    """Embedding provider configuration."""

    provider: str = "openai"  # openai, azure_openai, cohere
    model: str = "text-embedding-3-large"
    dimensions: int = 3072
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure OpenAI
    batch_size: int = 100

    def validate(self) -> None:
        """Validate embedding configuration."""
        if self.provider == "openai":
            if not self.api_key:
                raise ConfigurationError("OpenAI API key is required")
        elif self.provider == "azure_openai":
            if not self.api_key or not self.api_base or not self.deployment_name:
                raise ConfigurationError(
                    "Azure OpenAI requires api_key, api_base, and deployment_name"
                )
        elif self.provider == "cohere":
            if not self.api_key:
                raise ConfigurationError("Cohere API key is required")


@dataclass
class RerankingConfig:
    """Reranking configuration."""

    provider: str = "bge"  # bge, cohere, cross_encoder
    model: str = "BAAI/bge-reranker-v2-m3"
    api_key: Optional[str] = None
    api_url: Optional[str] = (
        "https://api.novita.ai/openai/v1/rerank"  # For BGE reranker
    )
    top_k: int = 20
    enabled: bool = True
    normalize: bool = False  # For BGE reranker
    timeout: int = 30  # Request timeout in seconds

    # LLM Fallback Configuration
    fallback_enabled: bool = False
    fallback_endpoint: Optional[str] = None
    fallback_api_key: Optional[str] = None
    fallback_model: str = "gpt-oss-120b"
    fallback_timeout: int = 60

    def validate(self) -> None:
        """Validate reranking configuration."""
        if self.enabled:
            if self.provider == "cohere" and not self.api_key:
                raise ConfigurationError("Cohere API key is required for reranking")
            elif self.provider == "bge" and not self.api_key:
                raise ConfigurationError("BGE reranker API key is required")
            elif self.provider == "bge" and not self.api_url:
                raise ConfigurationError("BGE reranker API URL is required")

        if self.fallback_enabled:
            if not self.fallback_endpoint:
                raise ConfigurationError(
                    "LLM fallback endpoint is required when fallback is enabled"
                )
            if not self.fallback_api_key:
                raise ConfigurationError(
                    "LLM fallback API key is required when fallback is enabled"
                )


@dataclass
class LLMConfig:
    """LLM configuration for query generation."""

    provider: str = "openai"  # openai, azure_openai, anthropic
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None  # For Azure OpenAI
    temperature: float = 0.0

    def validate(self) -> None:
        """Validate LLM configuration."""
        if self.provider == "openai":
            if not self.api_key:
                raise ConfigurationError("OpenAI API key is required")
        elif self.provider == "azure_openai":
            if not self.api_key or not self.api_base or not self.deployment_name:
                raise ConfigurationError(
                    "Azure OpenAI requires api_key, api_base, and deployment_name"
                )


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""

    method: str = "semantic"  # semantic, recursive, fixed
    max_chunk_size: int = 1000  # tokens
    overlap_percentage: float = 0.2
    semantic_threshold_percentile: int = 95
    min_chunk_size: int = 100  # tokens

    def validate(self) -> None:
        """Validate chunking configuration."""
        if self.max_chunk_size <= 0:
            raise ConfigurationError("max_chunk_size must be positive")
        if not 0 <= self.overlap_percentage < 1:
            raise ConfigurationError("overlap_percentage must be between 0 and 1")


@dataclass
class PDFConfig:
    """PDF processing configuration."""

    parser: str = "pdfplumber"  # pdfplumber, pypdf2, chunkr, unstructured
    extract_images: bool = False
    extract_tables: bool = False
    validate_text: bool = True

    def validate(self) -> None:
        """Validate PDF configuration."""
        valid_parsers = {"pdfplumber", "pypdf2", "chunkr", "unstructured"}
        if self.parser not in valid_parsers:
            raise ConfigurationError(
                f"Invalid parser: {self.parser}. Must be one of {valid_parsers}"
            )


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""

    vector_search_limit: int = 25
    keyword_search_limit: int = 50
    enable_hyde: bool = True
    enable_keyword_search: bool = True
    final_top_k: int = 20
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    score_threshold: Optional[float] = None
    store_chunk_text_in_qdrant: bool = (
        False  # NEW: Store chunk text in Qdrant (default: store in external DB)
    )

    def validate(self) -> None:
        """Validate retrieval configuration."""
        if self.vector_search_limit <= 0:
            raise ConfigurationError("vector_search_limit must be positive")
        if self.keyword_search_limit <= 0:
            raise ConfigurationError("keyword_search_limit must be positive")
        if self.final_top_k <= 0:
            raise ConfigurationError("final_top_k must be positive")


@dataclass
class GraphRAGConfig:
    """Graph RAG configuration for knowledge graph operations.

    This configuration is separate from RAGConfig and used only when
    initializing GraphRAGClient for knowledge graph-based retrieval.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "insta_rag_graph"
    llm_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-large"
    enabled: bool = True
    group_id: str = "insta_rag"

    def validate(self) -> None:
        """Validate Graph RAG configuration."""
        if not self.neo4j_uri:
            raise ConfigurationError("Neo4j URI is required for Graph RAG")
        if not self.neo4j_user:
            raise ConfigurationError("Neo4j user is required for Graph RAG")
        if not self.neo4j_password:
            raise ConfigurationError("Neo4j password is required for Graph RAG")

    @classmethod
    def from_env(cls, **kwargs) -> "GraphRAGConfig":
        """Create Graph RAG configuration from environment variables.

        Environment variables:
            NEO4J_URI: Neo4j Bolt URI
            NEO4J_USER: Neo4j username
            NEO4J_PASSWORD: Neo4j password
            NEO4J_DATABASE: Neo4j database name
            GRAPHITI_LLM_MODEL: LLM model for entity extraction
            GRAPHITI_EMBEDDING_MODEL: Embedding model

        Args:
            **kwargs: Override specific configuration values

        Returns:
            GraphRAGConfig instance
        """
        config = cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            neo4j_database=os.getenv("NEO4J_DATABASE", "insta_rag_graph"),
            llm_model=os.getenv("GRAPHITI_LLM_MODEL", "gpt-4"),
            embedding_model=os.getenv(
                "GRAPHITI_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            group_id=os.getenv("GRAPHITI_GROUP_ID", "insta_rag"),
        )

        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)."""
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_database": self.neo4j_database,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "enabled": self.enabled,
            "group_id": self.group_id,
        }


@dataclass
class RAGConfig:
    """Main RAG system configuration."""

    vectordb: VectorDBConfig
    embedding: EmbeddingConfig
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    def validate(self) -> None:
        """Validate all configuration sections."""
        self.vectordb.validate()
        self.embedding.validate()
        self.reranking.validate()
        self.llm.validate()
        self.chunking.validate()
        self.pdf.validate()
        self.retrieval.validate()

    @classmethod
    def from_env(cls, **kwargs) -> "RAGConfig":
        """Create configuration from environment variables.

        Environment variables:
            QDRANT_URL: Qdrant database URL
            QDRANT_API_KEY: Qdrant API key
            OPENAI_API_KEY or AZURE_OPENAI_API_KEY: OpenAI/Azure API key
            AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint
            AZURE_EMBEDDING_DEPLOYMENT: Azure deployment name
            COHERE_API_KEY: Cohere API key for reranking

        Args:
            **kwargs: Override specific configuration values

        Returns:
            RAGConfig instance
        """
        # Vector DB Config
        vectordb_config = VectorDBConfig(
            url=os.getenv("QDRANT_URL", ""),
            api_key=os.getenv("QDRANT_API_KEY", ""),
        )

        # Determine if using Azure or OpenAI
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if azure_endpoint and azure_key:
            # Azure OpenAI configuration
            embedding_config = EmbeddingConfig(
                provider="azure_openai",
                model="text-embedding-3-large",
                dimensions=3072,
                api_key=azure_key,
                api_base=azure_endpoint,
                api_version="2024-02-01",
                deployment_name=os.getenv(
                    "AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
                ),
            )

            llm_config = LLMConfig(
                provider="azure_openai",
                model="gpt-4",
                api_key=azure_key,
                api_base=azure_endpoint,
                api_version="2024-02-01",
                deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4"),
            )
        else:
            # Standard OpenAI configuration
            embedding_config = EmbeddingConfig(
                provider="openai",
                model="text-embedding-3-large",
                dimensions=3072,
                api_key=openai_key,
            )

            llm_config = LLMConfig(
                provider="openai",
                model="gpt-4",
                api_key=openai_key,
            )

        # Reranking config - prioritize BGE reranker
        bge_api_key = os.getenv("BGE_RERANKER_API_KEY")
        cohere_api_key = os.getenv("COHERE_API_KEY")

        # LLM fallback settings
        gpt_oss_endpoint = os.getenv("GPT_OSS_ENDPOINT")
        gpt_oss_api_key = os.getenv("GPT_OSS_API_KEY")
        gpt_oss_model = os.getenv("GPT_OSS_MODEL", "gpt-oss-120b")
        gpt_oss_fallback_enabled = (
            os.getenv("GPT_OSS_FALLBACK_ENABLED", "false").lower() == "true"
        )

        if bge_api_key:
            # Use BGE reranker if API key is available
            reranking_config = RerankingConfig(
                provider="bge",
                model="BAAI/bge-reranker-v2-m3",
                api_key=bge_api_key,
                api_url=os.getenv(
                    "BGE_RERANKER_URL", "https://api.novita.ai/openai/v1/rerank"
                ),
                enabled=True,
                normalize=False,
                fallback_enabled=gpt_oss_fallback_enabled,
                fallback_endpoint=gpt_oss_endpoint,
                fallback_api_key=gpt_oss_api_key,
                fallback_model=gpt_oss_model,
            )
        elif cohere_api_key:
            # Fallback to Cohere if available
            reranking_config = RerankingConfig(
                provider="cohere",
                model="rerank-v3.5",
                api_key=cohere_api_key,
                enabled=True,
                fallback_enabled=gpt_oss_fallback_enabled,
                fallback_endpoint=gpt_oss_endpoint,
                fallback_api_key=gpt_oss_api_key,
                fallback_model=gpt_oss_model,
            )
        else:
            # No reranking available
            reranking_config = RerankingConfig(
                provider="bge",
                model="BAAI/bge-reranker-v2-m3",
                enabled=False,
                fallback_enabled=gpt_oss_fallback_enabled,
                fallback_endpoint=gpt_oss_endpoint,
                fallback_api_key=gpt_oss_api_key,
                fallback_model=gpt_oss_model,
            )

        config = cls(
            vectordb=vectordb_config,
            embedding=embedding_config,
            reranking=reranking_config,
            llm=llm_config,
        )

        # Apply any overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key) and isinstance(value, dict):
                config_obj = getattr(config, key)
                for k, v in value.items():
                    setattr(config_obj, k, v)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (without sensitive data)."""
        config_dict = {
            "vectordb": {"provider": self.vectordb.provider, "url": self.vectordb.url},
            "embedding": {
                "provider": self.embedding.provider,
                "model": self.embedding.model,
                "dimensions": self.embedding.dimensions,
            },
            "reranking": {
                "provider": self.reranking.provider,
                "model": self.reranking.model,
                "enabled": self.reranking.enabled,
            },
            "llm": {"provider": self.llm.provider, "model": self.llm.model},
            "chunking": {
                "method": self.chunking.method,
                "max_chunk_size": self.chunking.max_chunk_size,
                "overlap_percentage": self.chunking.overlap_percentage,
            },
            "pdf": {"parser": self.pdf.parser},
            "retrieval": {
                "vector_search_limit": self.retrieval.vector_search_limit,
                "keyword_search_limit": self.retrieval.keyword_search_limit,
                "enable_hyde": self.retrieval.enable_hyde,
                "enable_keyword_search": self.retrieval.enable_keyword_search,
                "final_top_k": self.retrieval.final_top_k,
                "store_chunk_text_in_qdrant": self.retrieval.store_chunk_text_in_qdrant,
            },
        }

        return config_dict
