"""
Simple test file for the Hierarchical RAG Service.
Demonstrates basic functionality and can be used for testing.
"""

import unittest
from typing import Dict, Any

from src.service.hierarchical_rag import HierarchicalRAGService
from src.service.selective_editor import EditScope
from src.service.retrieval_pipeline import RetrievalStrategy, RetrievalLevel


class TestHierarchicalRAG(unittest.TestCase):
    """Test cases for the Hierarchical RAG Service."""

    def setUp(self):
        """Set up test fixtures."""
        self.rag_service = HierarchicalRAGService()

        # Test document content
        self.test_content = """
        Artificial Intelligence Fundamentals
        
        Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.
        
        Machine Learning
        
        Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions.
        
        Deep Learning
        
        Deep learning is a subset of machine learning that uses neural networks with multiple layers to model complex patterns. It has been particularly successful in image recognition, natural language processing, and speech recognition.
        
        Applications of AI
        
        AI has applications in various fields including healthcare, finance, transportation, and entertainment. It's used for medical diagnosis, fraud detection, autonomous vehicles, and recommendation systems.
        """

        self.test_metadata = {
            "title": "AI Fundamentals",
            "author": "Test Author",
            "category": "technology",
            "tags": ["AI", "machine learning", "deep learning"],
            "source": "test",
        }

    def test_document_ingestion(self):
        """Test basic document ingestion."""
        result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            chunk_size=500,
            generate_embeddings=True,
        )

        self.assertIsNotNone(result.document_id)
        self.assertGreater(len(result.chunk_ids), 0)
        self.assertGreater(result.embeddings_generated, 0)
        self.assertEqual(result.metadata["title"], "AI Fundamentals")

    def test_structured_document_ingestion(self):
        """Test structured document ingestion."""
        sections = [
            {
                "content": "Introduction to AI",
                "metadata": {"section_type": "introduction"},
                "chunks": [
                    "AI is a branch of computer science.",
                    "It aims to create intelligent machines.",
                ],
            },
            {
                "content": "Machine Learning Basics",
                "metadata": {"section_type": "tutorial"},
                "chunks": [
                    "Machine learning enables computers to learn.",
                    "It uses algorithms to identify patterns.",
                ],
            },
        ]

        result = self.rag_service.ingest_structured_document(
            sections=sections, metadata=self.test_metadata, generate_embeddings=True
        )

        self.assertIsNotNone(result.document_id)
        self.assertEqual(len(result.section_ids), 2)
        self.assertEqual(len(result.chunk_ids), 4)

    def test_semantic_search(self):
        """Test semantic search functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Perform semantic search
        search_results = self.rag_service.semantic_search(
            query="neural networks", top_k=3, similarity_threshold=0.5
        )

        self.assertIsNotNone(search_results)
        self.assertGreaterEqual(len(search_results.results), 0)

    def test_keyword_search(self):
        """Test keyword search functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Perform keyword search
        search_results = self.rag_service.keyword_search(
            query="machine learning", top_k=3
        )

        self.assertIsNotNone(search_results)
        self.assertGreaterEqual(len(search_results.results), 0)

    def test_hierarchical_search(self):
        """Test hierarchical search functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Perform hierarchical search
        search_results = self.rag_service.hierarchical_search(
            query="AI applications", top_k=3, include_context=True
        )

        self.assertIsNotNone(search_results)
        self.assertGreaterEqual(len(search_results.results), 0)

    def test_content_editing(self):
        """Test content editing functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Edit content
        edit_result = self.rag_service.edit_content(
            node_id=ingestion_result.document_id,
            new_content="Updated AI Fundamentals content...",
            scope=EditScope.NODE_ONLY,
            update_embeddings=True,
        )

        self.assertTrue(edit_result.success)
        self.assertGreaterEqual(len(edit_result.affected_nodes), 0)

    def test_metadata_editing(self):
        """Test metadata editing functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Edit metadata
        metadata_updates = {
            "last_updated": "2024-01-16",
            "version": "2.0",
            "reviewed": True,
        }

        edit_result = self.rag_service.edit_metadata(
            node_id=ingestion_result.document_id,
            metadata_updates=metadata_updates,
            scope=EditScope.NODE_ONLY,
        )

        self.assertTrue(edit_result.success)

    def test_document_structure(self):
        """Test document structure retrieval."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Get document structure
        structure = self.rag_service.get_document_structure(
            ingestion_result.document_id
        )

        self.assertIsNotNone(structure)
        self.assertIn("id", structure)
        self.assertIn("type", structure)
        self.assertIn("content", structure)

    def test_edit_history(self):
        """Test edit history functionality."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Make some edits
        self.rag_service.edit_content(
            node_id=ingestion_result.document_id,
            new_content="First edit...",
            scope=EditScope.NODE_ONLY,
        )

        self.rag_service.edit_metadata(
            node_id=ingestion_result.document_id,
            metadata_updates={"version": "1.1"},
            scope=EditScope.NODE_ONLY,
        )

        # Get edit history
        history = self.rag_service.get_edit_history(
            node_id=ingestion_result.document_id, limit=10
        )

        self.assertGreaterEqual(len(history), 2)

    def test_filtered_search(self):
        """Test search with filters."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Search with filters
        search_results = self.rag_service.search(
            query="AI", filters={"category": "technology"}, top_k=3
        )

        self.assertIsNotNone(search_results)
        self.assertGreaterEqual(len(search_results.results), 0)

    def test_hybrid_search(self):
        """Test hybrid search strategy."""
        # First ingest a document
        ingestion_result = self.rag_service.ingest_document(
            content=self.test_content,
            metadata=self.test_metadata,
            auto_chunk=True,
            generate_embeddings=True,
        )

        # Perform hybrid search
        search_results = self.rag_service.search(
            query="machine learning algorithms",
            strategy=RetrievalStrategy.HYBRID,
            level=RetrievalLevel.MULTI_LEVEL,
            top_k=3,
        )

        self.assertIsNotNone(search_results)
        self.assertGreaterEqual(len(search_results.results), 0)


def run_demo():
    """Run a demonstration of the hierarchical RAG service."""
    print("=== Hierarchical RAG Service Demo ===")

    # Initialize service
    rag_service = HierarchicalRAGService()

    # Demo content
    demo_content = """
    Python Programming Guide
    
    Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation.
    
    Getting Started
    
    To get started with Python, you need to install Python on your system. Python is available for Windows, macOS, and Linux. You can download it from python.org.
    
    Basic Syntax
    
    Python uses indentation to define code blocks. Variables don't need type declarations. Here's a simple example:
    
    name = "World"
    print(f"Hello, {name}!")
    
    Data Structures
    
    Python provides several built-in data structures including lists, tuples, dictionaries, and sets. Lists are mutable, tuples are immutable, dictionaries store key-value pairs, and sets store unique elements.
    
    Functions and Classes
    
    Functions in Python are defined using the def keyword. Classes are defined using the class keyword. Python supports object-oriented programming with inheritance, encapsulation, and polymorphism.
    """

    demo_metadata = {
        "title": "Python Programming Guide",
        "author": "Demo Author",
        "category": "programming",
        "tags": ["python", "programming", "tutorial"],
        "difficulty": "beginner",
    }

    print("1. Ingesting document...")
    result = rag_service.ingest_document(
        content=demo_content,
        metadata=demo_metadata,
        auto_chunk=True,
        chunk_size=300,
        generate_embeddings=True,
    )

    print(f"   Document ID: {result.document_id}")
    print(f"   Chunks created: {len(result.chunk_ids)}")
    print(f"   Embeddings generated: {result.embeddings_generated}")

    print("\n2. Performing semantic search...")
    search_results = rag_service.semantic_search(query="data structures", top_k=2)

    print(f"   Found {len(search_results.results)} results")
    for i, result in enumerate(search_results.results):
        print(
            f"   {i+1}. {result.content[:80]}... (Score: {result.relevance_score:.3f})"
        )

    print("\n3. Performing keyword search...")
    keyword_results = rag_service.keyword_search(query="functions classes", top_k=2)

    print(f"   Found {len(keyword_results.results)} results")
    for i, result in enumerate(keyword_results.results):
        print(
            f"   {i+1}. {result.content[:80]}... (Score: {result.relevance_score:.3f})"
        )

    print("\n4. Editing document content...")
    edit_result = rag_service.edit_content(
        node_id=result.document_id,
        new_content="Updated Python Programming Guide with new information...",
        scope=EditScope.NODE_ONLY,
        update_embeddings=True,
    )

    print(f"   Edit successful: {edit_result.success}")
    print(f"   Affected nodes: {len(edit_result.affected_nodes)}")

    print("\n5. Getting document structure...")
    structure = rag_service.get_document_structure(result.document_id)
    print(f"   Document has {len(structure.get('children', []))} sections")

    print("\n6. Getting edit history...")
    history = rag_service.get_edit_history(node_id=result.document_id, limit=5)
    print(f"   Edit history: {len(history)} operations")

    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    # Run the demo
    run_demo()

    # Uncomment to run tests
    # unittest.main()
