"""Example usage of Graph RAG functionality.

This example demonstrates how to use Graph RAG alongside the existing
vector-based RAG. Both systems can operate independently or together.

Prerequisites:
- Neo4j running locally on bolt://localhost:7687
- GraphRAG config in .env or environment variables
"""

import asyncio
from insta_rag import DocumentInput, RAGClient, RAGConfig
from insta_rag.graph_rag import GraphRAGClient


async def example_graph_rag_only():
    """Example: Using Graph RAG independently (without vector RAG)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Graph RAG Standalone")
    print("=" * 60)

    # Initialize Graph RAG client
    graph_client = GraphRAGClient()
    await graph_client.initialize()

    try:
        # Add documents to knowledge graph
        docs = [
            DocumentInput.from_text(
                "Alice Johnson is a Senior Software Engineer at TechCorp. "
                "She specializes in machine learning and Python."
            ),
            DocumentInput.from_text(
                "TechCorp builds AI-powered products for enterprise customers. "
                "The company was founded in 2015."
            ),
            DocumentInput.from_text(
                "The AI team at TechCorp includes Alice, Bob, and Carol. "
                "They work on NLP models."
            ),
        ]

        print("\n📚 Adding documents to knowledge graph...")
        results = await graph_client.add_documents(docs, collection_name="company_info")

        for i, result in enumerate(results, 1):
            print(f"\nDocument {i}:")
            print(f"  - Episode UUID: {result.episode_uuid}")
            print(f"  - Entities extracted: {result.nodes_created}")
            print(f"  - Relationships: {result.edges_created}")
            print(f"  - Processing time: {result.processing_time_ms:.1f}ms")

        # Retrieve from knowledge graph
        print("\n🔍 Querying knowledge graph...")
        retrieval_result = await graph_client.retrieve(
            query="Who works at TechCorp and specializes in machine learning?",
            collection_name="company_info",
            k=10,
        )

        print(f"\nFound {len(retrieval_result.edges)} relevant facts:")
        for i, edge in enumerate(retrieval_result.edges, 1):
            print(f"\n{i}. {edge.fact}")
            print(f"   Score: {edge.score:.3f}")

        # Get context around an entity
        print("\n🌐 Getting entity context...")
        context_result = await graph_client.get_entity_context(
            entity_name="Alice",
            collection_name="company_info",
            depth=2,
        )

        print("\nContext for 'Alice':")
        for edge in context_result.edges[:5]:
            print(f"  - {edge.fact}")

    finally:
        await graph_client.close()
        print("\n✅ Graph RAG example completed")


async def example_vector_rag_only():
    """Example: Using vector RAG (existing functionality)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Vector RAG Standalone (Traditional RAG)")
    print("=" * 60)

    try:
        # Initialize traditional RAG client
        rag_config = RAGConfig.from_env()
        rag_client = RAGClient(rag_config)

        # Add documents to vector database
        docs = [
            DocumentInput.from_text(
                "Alice Johnson is a Senior Software Engineer at TechCorp. "
                "She specializes in machine learning and Python."
            ),
            DocumentInput.from_text(
                "TechCorp builds AI-powered products for enterprise customers."
            ),
        ]

        print("\n📚 Adding documents to vector database...")
        response = rag_client.add_documents(docs, collection_name="company_info")
        print(f"Added {len(response.successful_ids)} documents")

        # Retrieve from vector database
        print("\n🔍 Querying vector database...")
        retrieval_result = rag_client.retrieve(
            query="Who works at TechCorp?",
            collection_name="company_info",
            k=5,
        )

        print(f"\nFound {len(retrieval_result.chunks)} chunks:")
        for chunk in retrieval_result.chunks:
            print(f"\n  Score: {chunk.score:.3f}")
            print(f"  Content: {chunk.content[:100]}...")

    except Exception as e:
        print(f"Vector RAG example error: {e}")


async def example_combined_usage():
    """Example: Using both Graph RAG and Vector RAG together.

    This demonstrates how both systems can coexist and complement each other.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Combined Graph RAG + Vector RAG")
    print("=" * 60)

    # Initialize both clients
    graph_client = GraphRAGClient()
    await graph_client.initialize()

    try:
        rag_config = RAGConfig.from_env()
        rag_client = RAGClient(rag_config)

        # Same documents go into both systems
        docs = [
            DocumentInput.from_text(
                "Alice Johnson is a Senior Software Engineer at TechCorp. "
                "She specializes in machine learning and Python. "
                "She leads the AI research team."
            ),
            DocumentInput.from_text(
                "TechCorp builds AI-powered products. Founded in 2015. "
                "Headquarters in San Francisco."
            ),
        ]

        print("\n📚 Adding documents to both systems...")

        # Add to graph
        graph_results = await graph_client.add_documents(
            docs, collection_name="company"
        )
        print(f"Graph: Added {sum(r.nodes_created for r in graph_results)} entities")

        # Add to vector
        vector_response = rag_client.add_documents(docs, collection_name="company")
        print(f"Vector: Added {len(vector_response.successful_ids)} documents")

        # Query both systems
        query = "What does Alice do at TechCorp?"

        print(f"\n🔍 Querying both systems for: '{query}'")

        # Graph retrieval
        print("\n  Graph RAG Results:")
        graph_result = await graph_client.retrieve(
            query=query, collection_name="company", k=5
        )
        for edge in graph_result.edges[:3]:
            print(f"    - {edge.fact} (score: {edge.score:.3f})")

        # Vector retrieval
        print("\n  Vector RAG Results:")
        vector_result = rag_client.retrieve(query=query, collection_name="company", k=5)
        for chunk in vector_result.chunks[:3]:
            print(f"    - {chunk.content[:80]}... (score: {chunk.score:.3f})")

        print("\n💡 You can now use both results for better context!")

    finally:
        await graph_client.close()
        print("\n✅ Combined example completed")


async def example_graph_rag_advanced():
    """Example: Advanced Graph RAG features."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Advanced Graph RAG Features")
    print("=" * 60)

    graph_client = GraphRAGClient()
    await graph_client.initialize()

    try:
        # Add diverse documents
        docs = [
            DocumentInput.from_text(
                "Project Alpha: Machine Learning infrastructure project. "
                "Led by Alice. Budget: $2M. Timeline: 12 months."
            ),
            DocumentInput.from_text(
                "Alice leads Project Alpha. She reports to Bob, the VP of AI. "
                "The team has 5 engineers."
            ),
            DocumentInput.from_text(
                "Bob joined TechCorp in 2020. He manages multiple teams. "
                "He has 15 years of industry experience."
            ),
        ]

        print("\n📚 Building rich knowledge graph...")
        await graph_client.add_documents(docs, collection_name="org_structure")

        # Basic retrieval
        print("\n🔍 Basic search:")
        result = await graph_client.retrieve(
            "Who leads the ML infrastructure project?",
            collection_name="org_structure",
            k=5,
        )
        for edge in result.edges:
            print(f"  - {edge.fact}")

        # Reranked retrieval
        print("\n🎯 Reranked search (graph distance based):")
        reranked_result = await graph_client.retrieve_with_reranking(
            "Project details",
            collection_name="org_structure",
            k=10,
        )
        for i, edge in enumerate(reranked_result.edges[:3], 1):
            print(f"  {i}. {edge.fact} (score: {edge.score:.3f})")

        # Entity context
        print("\n🌐 Entity context discovery:")
        alice_context = await graph_client.get_entity_context(
            "Alice", collection_name="org_structure", depth=2
        )
        print(f"Found {len(alice_context.edges)} facts about Alice:")
        for edge in alice_context.edges[:5]:
            print(f"  - {edge.fact}")

    finally:
        await graph_client.close()
        print("\n✅ Advanced example completed")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Graph RAG Usage Examples")
    print("=" * 60)
    print(
        "\nThese examples show how to use Graph RAG independently "
        "or alongside Vector RAG."
    )

    # Run examples
    await example_graph_rag_only()

    # Uncomment to try other examples:
    # await example_vector_rag_only()
    # await example_combined_usage()
    # await example_graph_rag_advanced()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
