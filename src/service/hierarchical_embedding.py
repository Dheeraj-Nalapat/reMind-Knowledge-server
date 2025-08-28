"""
Hierarchical Embedding Service for generating and managing multi-level embeddings.
Supports document-level, section-level, and chunk-level embeddings with context awareness.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.common.embedding.embedder import OpenAITextVectorizer
from src.common.logger.logger import get_logger
from .document_hierarchy import HierarchyNode, NodeType

logger = get_logger(__name__)


class EmbeddingLevel(Enum):
    """Levels of embedding in the hierarchy."""

    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    node_id: str
    level: EmbeddingLevel
    embedding: List[float]
    context: str
    metadata: Dict[str, Any]


class HierarchicalEmbeddingService:
    """
    Service for generating hierarchical embeddings with context awareness.
    """

    def __init__(self, embedder: Optional[OpenAITextVectorizer] = None):
        self.embedder = embedder or OpenAITextVectorizer()
        self.embedding_cache: Dict[str, EmbeddingResult] = {}

    def generate_document_embedding(
        self, node: HierarchyNode, include_sections: bool = True
    ) -> EmbeddingResult:
        """
        Generate document-level embedding with optional section context.

        Args:
            node: Document node
            include_sections: Whether to include section summaries in context

        Returns:
            Embedding result
        """
        if node.node_type != NodeType.DOCUMENT:
            raise ValueError(f"Node {node.id} is not a document")

        # Build document context
        context_parts = [node.content]

        if include_sections:
            section_summaries = self._get_section_summaries(node)
            if section_summaries:
                context_parts.append(f"Sections: {'; '.join(section_summaries)}")

        full_context = "\n\n".join(context_parts)

        # Generate embedding
        embedding = self.embedder.embed_text(full_context)

        result = EmbeddingResult(
            node_id=node.id,
            level=EmbeddingLevel.DOCUMENT,
            embedding=embedding,
            context=full_context,
            metadata={
                "include_sections": include_sections,
                "section_count": len(node.children_ids),
                **node.metadata,
            },
        )

        self.embedding_cache[node.id] = result
        logger.info(f"Generated document embedding for {node.id}")

        return result

    def generate_section_embedding(
        self,
        node: HierarchyNode,
        include_parent_context: bool = True,
        include_chunks: bool = True,
    ) -> EmbeddingResult:
        """
        Generate section-level embedding with parent and chunk context.

        Args:
            node: Section node
            include_parent_context: Whether to include parent document context
            include_chunks: Whether to include chunk summaries

        Returns:
            Embedding result
        """
        if node.node_type != NodeType.SECTION:
            raise ValueError(f"Node {node.id} is not a section")

        # Build section context
        context_parts = [node.content]

        if include_parent_context and node.parent_id:
            parent_context = self._get_parent_context(node.parent_id)
            if parent_context:
                context_parts.insert(0, f"Document: {parent_context}")

        if include_chunks:
            chunk_summaries = self._get_chunk_summaries(node)
            if chunk_summaries:
                context_parts.append(f"Chunks: {'; '.join(chunk_summaries)}")

        full_context = "\n\n".join(context_parts)

        # Generate embedding
        embedding = self.embedder.embed_text(full_context)

        result = EmbeddingResult(
            node_id=node.id,
            level=EmbeddingLevel.SECTION,
            embedding=embedding,
            context=full_context,
            metadata={
                "include_parent_context": include_parent_context,
                "include_chunks": include_chunks,
                "chunk_count": len(node.children_ids),
                **node.metadata,
            },
        )

        self.embedding_cache[node.id] = result
        logger.info(f"Generated section embedding for {node.id}")

        return result

    def generate_chunk_embedding(
        self, node: HierarchyNode, include_hierarchy_context: bool = True
    ) -> EmbeddingResult:
        """
        Generate chunk-level embedding with hierarchy context.

        Args:
            node: Chunk node
            include_hierarchy_context: Whether to include parent context

        Returns:
            Embedding result
        """
        if node.node_type != NodeType.CHUNK:
            raise ValueError(f"Node {node.id} is not a chunk")

        # Build chunk context
        context_parts = [node.content]

        if include_hierarchy_context:
            hierarchy_context = self._get_hierarchy_context(node)
            if hierarchy_context:
                context_parts.insert(0, f"Context: {hierarchy_context}")

        full_context = "\n\n".join(context_parts)

        # Generate embedding
        embedding = self.embedder.embed_text(full_context)

        result = EmbeddingResult(
            node_id=node.id,
            level=EmbeddingLevel.CHUNK,
            embedding=embedding,
            context=full_context,
            metadata={
                "include_hierarchy_context": include_hierarchy_context,
                **node.metadata,
            },
        )

        self.embedding_cache[node.id] = result
        logger.info(f"Generated chunk embedding for {node.id}")

        return result

    def generate_hierarchical_embeddings(
        self,
        document_node: HierarchyNode,
        levels: Optional[List[EmbeddingLevel]] = None,
    ) -> Dict[str, EmbeddingResult]:
        """
        Generate embeddings for all levels of a document hierarchy.

        Args:
            document_node: Root document node
            levels: Which levels to generate embeddings for

        Returns:
            Dictionary of node_id -> embedding result
        """
        if levels is None:
            levels = [
                EmbeddingLevel.DOCUMENT,
                EmbeddingLevel.SECTION,
                EmbeddingLevel.CHUNK,
            ]

        results = {}

        # Generate document embedding
        if EmbeddingLevel.DOCUMENT in levels:
            doc_result = self.generate_document_embedding(document_node)
            results[document_node.id] = doc_result

        # Generate section embeddings
        if EmbeddingLevel.SECTION in levels:
            for section_id in document_node.children_ids:
                # Note: This assumes we have access to the hierarchy manager
                # In practice, you'd pass the section nodes here
                pass

        # Generate chunk embeddings
        if EmbeddingLevel.CHUNK in levels:
            # Similar to sections, would iterate through all chunks
            pass

        return results

    def update_embedding(
        self, node: HierarchyNode, level: EmbeddingLevel
    ) -> EmbeddingResult:
        """
        Update embedding for a specific node and level.

        Args:
            node: Node to update
            level: Embedding level

        Returns:
            Updated embedding result
        """
        if level == EmbeddingLevel.DOCUMENT:
            return self.generate_document_embedding(node)
        elif level == EmbeddingLevel.SECTION:
            return self.generate_section_embedding(node)
        elif level == EmbeddingLevel.CHUNK:
            return self.generate_chunk_embedding(node)
        else:
            raise ValueError(f"Unknown embedding level: {level}")

    def get_embedding(self, node_id: str) -> Optional[EmbeddingResult]:
        """Get cached embedding for a node."""
        return self.embedding_cache.get(node_id)

    def clear_cache(self, node_id: Optional[str] = None):
        """Clear embedding cache."""
        if node_id:
            self.embedding_cache.pop(node_id, None)
        else:
            self.embedding_cache.clear()

    def _get_section_summaries(self, document_node: HierarchyNode) -> List[str]:
        """Get summaries of document sections."""
        summaries = []
        # This would need access to section nodes
        # For now, return empty list
        return summaries

    def _get_chunk_summaries(self, section_node: HierarchyNode) -> List[str]:
        """Get summaries of section chunks."""
        summaries = []
        # This would need access to chunk nodes
        # For now, return empty list
        return summaries

    def _get_parent_context(self, parent_id: str) -> Optional[str]:
        """Get context from parent node."""
        # This would need access to the parent node
        # For now, return None
        return None

    def _get_hierarchy_context(self, chunk_node: HierarchyNode) -> Optional[str]:
        """Get hierarchy context for a chunk."""
        # This would need access to parent nodes
        # For now, return None
        return None

    def batch_generate_embeddings(
        self, nodes: List[Tuple[HierarchyNode, EmbeddingLevel]]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple nodes in batch.

        Args:
            nodes: List of (node, level) tuples

        Returns:
            List of embedding results
        """
        results = []

        for node, level in nodes:
            try:
                if level == EmbeddingLevel.DOCUMENT:
                    result = self.generate_document_embedding(node)
                elif level == EmbeddingLevel.SECTION:
                    result = self.generate_section_embedding(node)
                elif level == EmbeddingLevel.CHUNK:
                    result = self.generate_chunk_embedding(node)
                else:
                    logger.warning(f"Unknown embedding level: {level}")
                    continue

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to generate embedding for node {node.id}: {e}")
                continue

        return results

    def get_similar_nodes(
        self,
        query_embedding: List[float],
        level: EmbeddingLevel,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> List[Tuple[EmbeddingResult, float]]:
        """
        Find similar nodes based on embedding similarity.

        Args:
            query_embedding: Query embedding
            level: Embedding level to search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of (embedding_result, similarity_score) tuples
        """
        similarities = []

        for result in self.embedding_cache.values():
            if result.level != level:
                continue

            similarity = self._cosine_similarity(query_embedding, result.embedding)
            if similarity >= similarity_threshold:
                similarities.append((result, similarity))

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
