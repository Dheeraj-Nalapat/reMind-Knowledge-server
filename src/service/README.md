# Hierarchical RAG Service

A comprehensive hierarchical Retrieval-Augmented Generation (RAG) service that supports multi-level document organization, intelligent embeddings, selective editing, and context-aware retrieval.

## 🏗️ Architecture Overview

The hierarchical RAG service is built around a tree structure that organizes documents into three levels:

```
Document
├── Section 1
│   ├── Chunk 1.1
│   ├── Chunk 1.2
│   └── Chunk 1.3
├── Section 2
│   ├── Chunk 2.1
│   └── Chunk 2.2
└── Section 3
    └── Chunk 3.1
```

### Key Components

1. **Document Hierarchy Manager** - Manages the tree structure and node relationships
2. **Hierarchical Embedding Service** - Generates context-aware embeddings at multiple levels
3. **Selective Editor** - Enables targeted updates with propagation to child nodes
4. **Retrieval Pipeline** - Provides intelligent search with multiple strategies
5. **Main Service** - Orchestrates all components with a unified interface

## 🚀 Quick Start

```python
from src.service.hierarchical_rag import HierarchicalRAGService

# Initialize the service
rag_service = HierarchicalRAGService()

# Ingest a document
result = rag_service.ingest_document(
    content="Your document content here...",
    metadata={"title": "My Document", "category": "technology"},
    auto_chunk=True,
    generate_embeddings=True
)

# Search for relevant content
search_results = rag_service.search(
    query="machine learning algorithms",
    strategy=RetrievalStrategy.HYBRID,
    top_k=5
)

# Edit content with propagation
edit_result = rag_service.edit_content(
    node_id=result.document_id,
    new_content="Updated content...",
    scope=EditScope.FULL_HIERARCHY
)
```

## 📚 Core Features

### 1. Document Ingestion

#### Basic Ingestion

```python
# Simple document ingestion with auto-chunking
result = rag_service.ingest_document(
    content="Long document content...",
    metadata={"title": "Document Title"},
    auto_chunk=True,
    chunk_size=1000,
    chunk_overlap=200
)
```

#### Structured Ingestion

```python
# Pre-defined structure with sections and chunks
sections = [
    {
        "content": "Section 1 content",
        "metadata": {"section_type": "introduction"},
        "chunks": ["Chunk 1.1", "Chunk 1.2"]
    },
    {
        "content": "Section 2 content",
        "metadata": {"section_type": "main"},
        "chunks": ["Chunk 2.1", "Chunk 2.2"]
    }
]

result = rag_service.ingest_structured_document(
    sections=sections,
    metadata={"title": "Structured Document"}
)
```

### 2. Multi-Level Search

#### Semantic Search

```python
# Search using embeddings
results = rag_service.semantic_search(
    query="neural networks",
    level=RetrievalLevel.MULTI_LEVEL,
    top_k=5,
    similarity_threshold=0.7
)
```

#### Keyword Search

```python
# Traditional keyword matching
results = rag_service.keyword_search(
    query="machine learning",
    level=RetrievalLevel.CHUNK_ONLY,
    top_k=3
)
```

#### Hierarchical Search

```python
# Search with context aggregation
results = rag_service.hierarchical_search(
    query="deep learning applications",
    top_k=5,
    include_context=True
)
```

#### Hybrid Search

```python
# Combine semantic and keyword approaches
results = rag_service.search(
    query="AI algorithms",
    strategy=RetrievalStrategy.HYBRID,
    level=RetrievalLevel.MULTI_LEVEL
)
```

### 3. Selective Editing

#### Content Editing

```python
# Edit with different scopes
rag_service.edit_content(
    node_id="document_id",
    new_content="Updated content",
    scope=EditScope.NODE_ONLY  # or CHILDREN_ONLY, DESCENDANTS_ONLY, FULL_HIERARCHY
)
```

#### Metadata Editing

```python
# Update metadata
rag_service.edit_metadata(
    node_id="document_id",
    metadata_updates={
        "last_updated": "2024-01-16",
        "version": "2.0",
        "reviewed": True
    }
)
```

#### Bulk Editing

```python
# Multiple edits in sequence
edits = [
    {"type": "content", "node_id": "doc1", "content": "New content 1"},
    {"type": "metadata", "node_id": "doc2", "metadata": {"status": "updated"}},
    {"type": "structure", "node_id": "section1", "new_parent_id": "doc3"}
]

results = rag_service.selective_editor.bulk_edit(edits)
```

### 4. Advanced Features

#### Filtered Search

```python
# Search with metadata filters
results = rag_service.search(
    query="python programming",
    filters={
        "category": "programming",
        "difficulty": "beginner"
    }
)
```

#### Edit History

```python
# Track changes
history = rag_service.get_edit_history(
    node_id="document_id",
    limit=10
)
```

#### Embedding Updates

```python
# Update embeddings after content changes
rag_service.update_embeddings("node_id")
```

## 🔧 Configuration

### Embedding Models

The service uses OpenAI embeddings by default, but you can customize:

```python
from src.common.embedding.embedder import OpenAITextVectorizer

# Custom embedder
embedder = OpenAITextVectorizer(model="text-embedding-3-large")
embedding_service = HierarchicalEmbeddingService(embedder=embedder)
```

### Chunking Parameters

```python
# Custom chunking
result = rag_service.ingest_document(
    content="...",
    auto_chunk=True,
    chunk_size=800,      # Characters per chunk
    chunk_overlap=150    # Overlap between chunks
)
```

## 📊 Search Strategies

### RetrievalStrategy

- `SEMANTIC_SIMILARITY` - Uses embedding similarity
- `KEYWORD_MATCH` - Traditional keyword matching
- `HYBRID` - Combines semantic and keyword approaches
- `HIERARCHICAL` - Context-aware search with aggregation

### RetrievalLevel

- `CHUNK_ONLY` - Search only at chunk level
- `SECTION_ONLY` - Search only at section level
- `DOCUMENT_ONLY` - Search only at document level
- `MULTI_LEVEL` - Search across all levels

### EditScope

- `NODE_ONLY` - Edit only the target node
- `CHILDREN_ONLY` - Edit only direct children
- `DESCENDANTS_ONLY` - Edit all descendants
- `FULL_HIERARCHY` - Edit node and all descendants

## 🎯 Use Cases

### 1. Knowledge Base Management

```python
# Ingest technical documentation
docs = [
    {"content": "API documentation...", "metadata": {"type": "api_doc"}},
    {"content": "User guide...", "metadata": {"type": "user_guide"}},
    {"content": "Troubleshooting...", "metadata": {"type": "troubleshooting"}}
]

for doc in docs:
    rag_service.ingest_document(**doc)

# Search across all documentation
results = rag_service.search("authentication error")
```

### 2. Content Management System

```python
# Structured content with sections
article = {
    "sections": [
        {"content": "Introduction", "chunks": ["intro1", "intro2"]},
        {"content": "Main Content", "chunks": ["main1", "main2", "main3"]},
        {"content": "Conclusion", "chunks": ["conclusion"]}
    ],
    "metadata": {"author": "John Doe", "category": "technology"}
}

result = rag_service.ingest_structured_document(**article)
```

### 3. Research Paper Analysis

```python
# Ingest research papers with structured sections
paper = {
    "sections": [
        {"content": "Abstract", "metadata": {"section": "abstract"}},
        {"content": "Introduction", "metadata": {"section": "introduction"}},
        {"content": "Methodology", "metadata": {"section": "methodology"}},
        {"content": "Results", "metadata": {"section": "results"}},
        {"content": "Conclusion", "metadata": {"section": "conclusion"}}
    ]
}

# Search for specific methodologies
results = rag_service.search(
    query="neural network architecture",
    filters={"section": "methodology"}
)
```

## 🔍 Performance Considerations

### Embedding Generation

- Embeddings are generated lazily by default
- Use `generate_embeddings=True` for immediate generation
- Consider batch processing for large documents

### Search Optimization

- Use appropriate `RetrievalLevel` to limit search scope
- Set `similarity_threshold` to filter low-quality results
- Use filters to narrow down search space

### Memory Management

- Large document collections may require external storage
- Consider implementing persistence for the hierarchy manager
- Embedding cache can be cleared when needed

## 🛠️ Extending the Service

### Custom Embedding Models

```python
class CustomEmbedder:
    def embed_text(self, text: str) -> List[float]:
        # Your embedding logic here
        pass

embedding_service = HierarchicalEmbeddingService(embedder=CustomEmbedder())
```

### Custom Retrieval Strategies

```python
class CustomRetrievalStrategy(RetrievalStrategy):
    CUSTOM = "custom"

# Extend RetrievalPipeline to handle custom strategy
```

### Persistence Layer

```python
# Implement database persistence for hierarchy manager
class PersistentHierarchyManager(DocumentHierarchyManager):
    def save_to_database(self):
        # Database save logic
        pass

    def load_from_database(self):
        # Database load logic
        pass
```

## 📝 Examples

See `example_usage.py` for comprehensive examples including:

- Basic document ingestion and search
- Structured document handling
- Bulk operations
- Editing workflows
- Advanced search scenarios

## 🤝 Contributing

When extending the hierarchical RAG service:

1. Follow the existing architecture patterns
2. Add comprehensive type hints
3. Include logging for debugging
4. Write tests for new functionality
5. Update documentation

## 📄 License

This module is part of the ReMind Knowledge Processor project.
