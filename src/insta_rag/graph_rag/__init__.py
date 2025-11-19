"""Graph RAG module - Knowledge graph based RAG operations.

This module provides Graph RAG capabilities as an optional feature
alongside the existing vector-based RAG. It enables construction and
querying of knowledge graphs using Neo4j and Graphiti.

Example:
    ```python
    import asyncio
    from insta_rag.graph_rag import GraphRAGClient, GraphAddResult
    from insta_rag import DocumentInput


    async def main():
        # Initialize Graph RAG client (separate from regular RAGClient)
        client = GraphRAGClient()
        await client.initialize()

        # Add documents to knowledge graph
        docs = [
            DocumentInput.from_text("Alice works as an engineer at TechCorp."),
            DocumentInput.from_text("TechCorp builds AI products."),
        ]
        results = await client.add_documents(docs, collection_name="company_info")

        # Retrieve from knowledge graph
        result = await client.retrieve(
            query="What does Alice do?", collection_name="company_info", k=5
        )

        print(f"Found {len(result.edges)} facts")
        for edge in result.edges:
            print(f"  - {edge.fact}")

        await client.close()


    asyncio.run(main())
    ```
"""

from .client import GraphRAGClient
from .models import GraphNode, GraphEdge, GraphRetrievalResult, GraphAddResult
from .graph_builder import GraphBuilder
from .graph_retriever import GraphRetriever

__all__ = [
    "GraphRAGClient",
    "GraphNode",
    "GraphEdge",
    "GraphRetrievalResult",
    "GraphAddResult",
    "GraphBuilder",
    "GraphRetriever",
]
