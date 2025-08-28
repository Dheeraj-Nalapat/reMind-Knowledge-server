"""
Example usage of the Hierarchical RAG Service.
Demonstrates ingestion, editing, and retrieval operations.
"""

from typing import Dict, Any
from src.service.hierarchical_rag import HierarchicalRAGService
from src.service.selective_editor import EditScope
from src.service.retrieval_pipeline import RetrievalStrategy, RetrievalLevel


def example_basic_usage():
    """Basic usage example of the hierarchical RAG service."""

    # Initialize the service
    rag_service = HierarchicalRAGService()

    # Example document content
    document_content = """
    Machine Learning Fundamentals
    
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised Learning
    Supervised learning involves training a model on labeled data. The model learns to map input features to known output labels. Common algorithms include linear regression, logistic regression, decision trees, and neural networks.
    
    Unsupervised Learning
    Unsupervised learning works with unlabeled data. The goal is to discover hidden patterns or structures in the data. Common techniques include clustering, dimensionality reduction, and association rule learning.
    
    Reinforcement Learning
    Reinforcement learning involves an agent learning to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions and learns to maximize cumulative rewards.
    
    Applications of Machine Learning
    
    Machine learning has applications in various fields including healthcare, finance, marketing, and autonomous vehicles. It's used for image recognition, natural language processing, recommendation systems, and predictive analytics.
    """

    # Metadata for the document
    metadata = {
        "title": "Machine Learning Fundamentals",
        "author": "AI Expert",
        "category": "technology",
        "tags": ["machine learning", "AI", "algorithms"],
        "source": "educational",
        "created_at": "2024-01-15",
    }

    print("=== Document Ingestion ===")

    # Ingest the document
    ingestion_result = rag_service.ingest_document(
        content=document_content,
        metadata=metadata,
        auto_chunk=True,
        chunk_size=500,
        chunk_overlap=100,
        generate_embeddings=True,
    )

    print(f"Document ingested with ID: {ingestion_result.document_id}")
    print(f"Created {len(ingestion_result.section_ids)} sections")
    print(f"Created {len(ingestion_result.chunk_ids)} chunks")
    print(f"Generated {ingestion_result.embeddings_generated} embeddings")

    print("\n=== Search Operations ===")

    # Semantic search
    print("Semantic search for 'neural networks':")
    semantic_results = rag_service.semantic_search(
        query="neural networks", top_k=3, similarity_threshold=0.6
    )

    for i, result in enumerate(semantic_results.results):
        print(f"{i+1}. {result.content[:100]}... (Score: {result.relevance_score:.3f})")

    # Keyword search
    print("\nKeyword search for 'supervised learning':")
    keyword_results = rag_service.keyword_search(query="supervised learning", top_k=3)

    for i, result in enumerate(keyword_results.results):
        print(f"{i+1}. {result.content[:100]}... (Score: {result.relevance_score:.3f})")

    # Hierarchical search
    print("\nHierarchical search for 'machine learning types':")
    hierarchical_results = rag_service.hierarchical_search(
        query="machine learning types", top_k=3, include_context=True
    )

    for i, result in enumerate(hierarchical_results.results):
        print(f"{i+1}. {result.content[:100]}...")
        if result.context:
            print(f"   Context: {result.context[:150]}...")

    print("\n=== Editing Operations ===")

    # Edit content
    print("Editing document content:")
    edit_result = rag_service.edit_content(
        node_id=ingestion_result.document_id,
        new_content="Updated Machine Learning Fundamentals with new information...",
        scope=EditScope.NODE_ONLY,
        update_embeddings=True,
    )

    print(f"Edit successful: {edit_result.success}")
    print(f"Affected nodes: {len(edit_result.affected_nodes)}")
    print(f"Updated embeddings: {len(edit_result.updated_embeddings)}")

    # Edit metadata
    print("\nUpdating document metadata:")
    metadata_update = {"last_updated": "2024-01-16", "version": "2.0", "reviewed": True}

    metadata_edit_result = rag_service.edit_metadata(
        node_id=ingestion_result.document_id,
        metadata_updates=metadata_update,
        scope=EditScope.NODE_ONLY,
    )

    print(f"Metadata update successful: {metadata_edit_result.success}")

    print("\n=== Document Structure ===")

    # Get document structure
    structure = rag_service.get_document_structure(ingestion_result.document_id)
    print(f"Document structure: {len(structure.get('children', []))} sections")

    print("\n=== Edit History ===")

    # Get edit history
    history = rag_service.get_edit_history(
        node_id=ingestion_result.document_id, limit=5
    )

    print(f"Edit history: {len(history)} operations")
    for op in history:
        print(f"- {op.edit_type.value}: {op.timestamp}")


def example_structured_document():
    """Example of ingesting a structured document."""

    rag_service = HierarchicalRAGService()

    # Structured document with predefined sections and chunks
    sections = [
        {
            "content": "Introduction to Python Programming",
            "metadata": {"section_type": "introduction", "difficulty": "beginner"},
            "chunks": [
                "Python is a high-level, interpreted programming language known for its simplicity and readability.",
                "It was created by Guido van Rossum and first released in 1991.",
                "Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
            ],
        },
        {
            "content": "Python Syntax and Data Types",
            "metadata": {"section_type": "tutorial", "difficulty": "beginner"},
            "chunks": [
                "Python uses indentation to define code blocks instead of curly braces.",
                "Basic data types include integers, floats, strings, booleans, lists, tuples, and dictionaries.",
                "Variables in Python are dynamically typed and don't require explicit type declarations.",
            ],
        },
        {
            "content": "Advanced Python Features",
            "metadata": {"section_type": "advanced", "difficulty": "intermediate"},
            "chunks": [
                "List comprehensions provide a concise way to create lists based on existing sequences.",
                "Decorators are functions that modify the behavior of other functions.",
                "Context managers (with statements) ensure proper resource management.",
            ],
        },
    ]

    metadata = {
        "title": "Complete Python Guide",
        "author": "Python Expert",
        "category": "programming",
        "tags": ["python", "programming", "tutorial"],
        "difficulty_levels": ["beginner", "intermediate"],
    }

    print("=== Structured Document Ingestion ===")

    result = rag_service.ingest_structured_document(
        sections=sections, metadata=metadata, generate_embeddings=True
    )

    print(f"Structured document ingested with ID: {result.document_id}")
    print(f"Created {len(result.section_ids)} sections")
    print(f"Created {len(result.chunk_ids)} chunks")

    # Search within the structured document
    print("\n=== Search in Structured Document ===")

    search_results = rag_service.search(
        query="list comprehensions",
        strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
        level=RetrievalLevel.CHUNK_ONLY,
        top_k=2,
    )

    for i, result in enumerate(search_results.results):
        print(f"{i+1}. {result.content}")
        print(f"   Type: {result.node_type.value}")
        print(f"   Score: {result.relevance_score:.3f}")


def example_bulk_operations():
    """Example of bulk operations."""

    rag_service = HierarchicalRAGService()

    # Multiple documents
    documents = [
        {
            "content": "Data Science involves extracting insights from data using statistical methods and machine learning algorithms.",
            "metadata": {"title": "Data Science Basics", "category": "data_science"},
        },
        {
            "content": "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns.",
            "metadata": {
                "title": "Deep Learning Introduction",
                "category": "machine_learning",
            },
        },
        {
            "content": "Natural Language Processing enables computers to understand, interpret, and generate human language.",
            "metadata": {"title": "NLP Fundamentals", "category": "nlp"},
        },
    ]

    print("=== Bulk Document Ingestion ===")

    ingestion_results = []
    for doc in documents:
        result = rag_service.ingest_document(
            content=doc["content"],
            metadata=doc["metadata"],
            auto_chunk=True,
            generate_embeddings=True,
        )
        ingestion_results.append(result)
        print(f"Ingested: {doc['metadata']['title']} (ID: {result.document_id})")

    print(f"\nTotal documents ingested: {len(ingestion_results)}")

    # Bulk search across all documents
    print("\n=== Cross-Document Search ===")

    cross_doc_results = rag_service.search(
        query="machine learning neural networks",
        strategy=RetrievalStrategy.HYBRID,
        level=RetrievalLevel.MULTI_LEVEL,
        top_k=5,
    )

    print(f"Found {cross_doc_results.total_found} relevant results")
    for i, result in enumerate(cross_doc_results.results):
        print(f"{i+1}. {result.content[:100]}...")
        print(f"   Document: {result.metadata.get('title', 'Unknown')}")
        print(f"   Score: {result.relevance_score:.3f}")


if __name__ == "__main__":
    print("Hierarchical RAG Service Examples")
    print("=" * 50)

    # Run examples
    example_basic_usage()
    print("\n" + "=" * 50)
    example_structured_document()
    print("\n" + "=" * 50)
    example_bulk_operations()
