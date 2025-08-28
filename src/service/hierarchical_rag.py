"""
Main Hierarchical RAG Service that orchestrates all components.
Provides a unified interface for ingestion, editing, and retrieval operations.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

from src.common.logger.logger import get_logger
from .document_hierarchy import DocumentHierarchyManager, HierarchyNode, NodeType
from .hierarchical_embedding import HierarchicalEmbeddingService, EmbeddingLevel
from .selective_editor import SelectiveEditor, EditScope, EditResult
from .retrieval_pipeline import (
    RetrievalPipeline,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    RetrievalLevel,
)

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: str
    section_ids: List[str]
    chunk_ids: List[str]
    embeddings_generated: int
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Result of search operation."""

    query: str
    results: List[RetrievalResult]
    total_found: int
    search_time_ms: float


class HierarchicalRAGService:
    """
    Main service for hierarchical RAG operations.
    Orchestrates document hierarchy, embeddings, editing, and retrieval.
    """

    def __init__(self):
        self.hierarchy_manager = DocumentHierarchyManager()
        self.embedding_service = HierarchicalEmbeddingService()
        self.selective_editor = SelectiveEditor(
            self.hierarchy_manager, self.embedding_service
        )
        self.retrieval_pipeline = RetrievalPipeline(
            self.hierarchy_manager, self.embedding_service
        )

    def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        auto_chunk: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        generate_embeddings: bool = True,
    ) -> IngestionResult:
        """
        Ingest a document into the hierarchical RAG system.

        Args:
            content: Document content
            metadata: Document metadata
            auto_chunk: Whether to automatically chunk the document
            chunk_size: Size of chunks in characters
            chunk_overlap: Overlap between chunks
            generate_embeddings: Whether to generate embeddings

        Returns:
            Ingestion result
        """
        start_time = datetime.utcnow()

        # Create document node
        document_id = self.hierarchy_manager.create_document(content, metadata)
        document_node = self.hierarchy_manager.get_node(document_id)

        section_ids = []
        chunk_ids = []
        embeddings_generated = 0

        if auto_chunk:
            # Auto-chunk the document
            chunks = self._chunk_content(content, chunk_size, chunk_overlap)

            # Create sections and chunks
            for i, chunk in enumerate(chunks):
                # Create section for each chunk (or group chunks into sections)
                section_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "auto_generated": True,
                }

                section_id = self.hierarchy_manager.add_section(
                    document_id, f"Section {i+1}", section_metadata
                )
                section_ids.append(section_id)

                # Create chunk
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "auto_generated": True,
                }

                chunk_id = self.hierarchy_manager.add_chunk(
                    section_id, chunk, chunk_metadata
                )
                chunk_ids.append(chunk_id)

        # Generate embeddings if requested
        if generate_embeddings:
            embeddings_generated = self._generate_document_embeddings(document_id)

        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()

        result = IngestionResult(
            document_id=document_id,
            section_ids=section_ids,
            chunk_ids=chunk_ids,
            embeddings_generated=embeddings_generated,
            metadata={
                **metadata,
                "processing_time_seconds": processing_time,
                "auto_chunked": auto_chunk,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )

        logger.info(f"Ingested document {document_id} with {len(chunk_ids)} chunks")
        return result

    def ingest_structured_document(
        self,
        sections: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        generate_embeddings: bool = True,
    ) -> IngestionResult:
        """
        Ingest a document with predefined structure.

        Args:
            sections: List of sections with content and metadata
            metadata: Document metadata
            generate_embeddings: Whether to generate embeddings

        Returns:
            Ingestion result
        """
        # Create document with first section as main content
        main_content = sections[0].get("content", "") if sections else ""
        document_id = self.hierarchy_manager.create_document(main_content, metadata)

        section_ids = []
        chunk_ids = []
        embeddings_generated = 0

        # Create sections and chunks
        for i, section_data in enumerate(sections):
            section_content = section_data.get("content", "")
            section_metadata = {**metadata, **section_data.get("metadata", {})}

            section_id = self.hierarchy_manager.add_section(
                document_id, section_content, section_metadata
            )
            section_ids.append(section_id)

            # Create chunks for this section
            if "chunks" in section_data:
                for j, chunk_content in enumerate(section_data["chunks"]):
                    chunk_metadata = {
                        **section_metadata,
                        "chunk_index": j,
                        "section_index": i,
                    }

                    chunk_id = self.hierarchy_manager.add_chunk(
                        section_id, chunk_content, chunk_metadata
                    )
                    chunk_ids.append(chunk_id)

        # Generate embeddings if requested
        if generate_embeddings:
            embeddings_generated = self._generate_document_embeddings(document_id)

        result = IngestionResult(
            document_id=document_id,
            section_ids=section_ids,
            chunk_ids=chunk_ids,
            embeddings_generated=embeddings_generated,
            metadata=metadata,
        )

        logger.info(f"Ingested structured document {document_id}")
        return result

    def edit_content(
        self,
        node_id: str,
        new_content: str,
        scope: EditScope = EditScope.NODE_ONLY,
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> EditResult:
        """
        Edit content of a node with optional propagation.

        Args:
            node_id: Node to edit
            new_content: New content
            scope: Scope of the edit
            update_embeddings: Whether to update embeddings
            user_id: User performing the edit

        Returns:
            Edit result
        """
        return self.selective_editor.edit_node_content(
            node_id, new_content, scope, update_embeddings, user_id
        )

    def edit_metadata(
        self,
        node_id: str,
        metadata_updates: Dict[str, Any],
        scope: EditScope = EditScope.NODE_ONLY,
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> EditResult:
        """
        Edit metadata of a node with optional propagation.

        Args:
            node_id: Node to edit
            metadata_updates: Metadata changes
            scope: Scope of the edit
            update_embeddings: Whether to update embeddings
            user_id: User performing the edit

        Returns:
            Edit result
        """
        return self.selective_editor.edit_node_metadata(
            node_id, metadata_updates, scope, update_embeddings, user_id
        )

    def search(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        level: RetrievalLevel = RetrievalLevel.MULTI_LEVEL,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_context: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """
        Search for relevant documents.

        Args:
            query: Search query
            strategy: Retrieval strategy
            level: Retrieval level
            top_k: Number of results
            similarity_threshold: Minimum similarity
            include_context: Whether to include context
            filters: Optional filters

        Returns:
            Search result
        """
        start_time = datetime.utcnow()

        retrieval_query = RetrievalQuery(
            text=query,
            strategy=strategy,
            level=level,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_context=include_context,
            filters=filters,
        )

        results = self.retrieval_pipeline.retrieve(retrieval_query)

        end_time = datetime.utcnow()
        search_time_ms = (end_time - start_time).total_seconds() * 1000

        return SearchResult(
            query=query,
            results=results,
            total_found=len(results),
            search_time_ms=search_time_ms,
        )

    def semantic_search(
        self,
        query: str,
        level: RetrievalLevel = RetrievalLevel.MULTI_LEVEL,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Semantic search using embeddings."""
        return self.search(
            query=query,
            strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
            level=level,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

    def keyword_search(
        self,
        query: str,
        level: RetrievalLevel = RetrievalLevel.MULTI_LEVEL,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Keyword-based search."""
        return self.search(
            query=query,
            strategy=RetrievalStrategy.KEYWORD_MATCH,
            level=level,
            top_k=top_k,
            filters=filters,
        )

    def hierarchical_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_context: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResult:
        """Hierarchical search with context aggregation."""
        return self.search(
            query=query,
            strategy=RetrievalStrategy.HIERARCHICAL,
            level=RetrievalLevel.MULTI_LEVEL,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_context=include_context,
            filters=filters,
        )

    def get_document_structure(self, document_id: str) -> Dict[str, Any]:
        """Get the complete structure of a document."""
        return self.hierarchy_manager.get_document_structure(document_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document structures."""
        return self.hierarchy_manager.get_all_documents()

    def delete_document(self, document_id: str, recursive: bool = True) -> bool:
        """Delete a document and optionally its descendants."""
        return self.hierarchy_manager.delete_node(document_id, recursive)

    def get_edit_history(
        self,
        node_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Any]:
        """Get edit history with optional filtering."""
        return self.selective_editor.get_edit_history(node_id, user_id, limit)

    def update_embeddings(self, node_id: str) -> bool:
        """Update embeddings for a specific node."""
        try:
            node = self.hierarchy_manager.get_node(node_id)
            if not node:
                return False

            if node.node_type == NodeType.DOCUMENT:
                self.embedding_service.generate_document_embedding(node)
            elif node.node_type == NodeType.SECTION:
                self.embedding_service.generate_section_embedding(node)
            elif node.node_type == NodeType.CHUNK:
                self.embedding_service.generate_chunk_embedding(node)

            return True
        except Exception as e:
            logger.error(f"Failed to update embeddings for {node_id}: {e}")
            return False

    def _chunk_content(
        self, content: str, chunk_size: int, chunk_overlap: int
    ) -> List[str]:
        """Chunk content into smaller pieces."""
        chunks = []
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)

            if end >= len(content):
                break

            start = end - chunk_overlap

        return chunks

    def _generate_document_embeddings(self, document_id: str) -> int:
        """Generate embeddings for all nodes in a document."""
        document_node = self.hierarchy_manager.get_node(document_id)
        if not document_node:
            return 0

        embeddings_generated = 0

        # Generate document embedding
        try:
            self.embedding_service.generate_document_embedding(document_node)
            embeddings_generated += 1
        except Exception as e:
            logger.error(f"Failed to generate document embedding: {e}")

        # Generate section embeddings
        for section_id in document_node.children_ids:
            section_node = self.hierarchy_manager.get_node(section_id)
            if section_node:
                try:
                    self.embedding_service.generate_section_embedding(section_node)
                    embeddings_generated += 1
                except Exception as e:
                    logger.error(f"Failed to generate section embedding: {e}")

                # Generate chunk embeddings
                for chunk_id in section_node.children_ids:
                    chunk_node = self.hierarchy_manager.get_node(chunk_id)
                    if chunk_node:
                        try:
                            self.embedding_service.generate_chunk_embedding(chunk_node)
                            embeddings_generated += 1
                        except Exception as e:
                            logger.error(f"Failed to generate chunk embedding: {e}")

        return embeddings_generated
