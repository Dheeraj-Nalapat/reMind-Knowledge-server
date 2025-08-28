"""
Retrieval Pipeline for hierarchical RAG with context-aware search.
Supports multi-level retrieval with relevance scoring and context aggregation.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from src.common.logger.logger import get_logger
from .document_hierarchy import HierarchyNode, NodeType, DocumentHierarchyManager
from .hierarchical_embedding import HierarchicalEmbeddingService, EmbeddingLevel

logger = get_logger(__name__)


class RetrievalStrategy(Enum):
    """Retrieval strategies for different use cases."""

    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCH = "keyword_match"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class RetrievalLevel(Enum):
    """Levels for retrieval operations."""

    CHUNK_ONLY = "chunk_only"
    SECTION_ONLY = "section_only"
    DOCUMENT_ONLY = "document_only"
    MULTI_LEVEL = "multi_level"


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    node_id: str
    node_type: NodeType
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    context: str
    hierarchy_path: List[str]
    embedding_level: EmbeddingLevel


@dataclass
class RetrievalQuery:
    """Query for retrieval operations."""

    text: str
    strategy: RetrievalStrategy
    level: RetrievalLevel
    top_k: int = 5
    similarity_threshold: float = 0.7
    include_context: bool = True
    filters: Optional[Dict[str, Any]] = None


class RetrievalPipeline:
    """
    Intelligent retrieval pipeline for hierarchical RAG.
    """

    def __init__(
        self,
        hierarchy_manager: DocumentHierarchyManager,
        embedding_service: HierarchicalEmbeddingService,
    ):
        self.hierarchy_manager = hierarchy_manager
        self.embedding_service = embedding_service

    def retrieve(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """
        Main retrieval method that routes to appropriate strategy.

        Args:
            query: Retrieval query specification

        Returns:
            List of retrieval results
        """
        if query.strategy == RetrievalStrategy.SEMANTIC_SIMILARITY:
            return self._semantic_retrieval(query)
        elif query.strategy == RetrievalStrategy.KEYWORD_MATCH:
            return self._keyword_retrieval(query)
        elif query.strategy == RetrievalStrategy.HYBRID:
            return self._hybrid_retrieval(query)
        elif query.strategy == RetrievalStrategy.HIERARCHICAL:
            return self._hierarchical_retrieval(query)
        else:
            raise ValueError(f"Unknown retrieval strategy: {query.strategy}")

    def semantic_search(
        self,
        query_text: str,
        level: RetrievalLevel = RetrievalLevel.MULTI_LEVEL,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Semantic search using embeddings.

        Args:
            query_text: Search query
            level: Retrieval level
            top_k: Number of results
            similarity_threshold: Minimum similarity
            filters: Optional filters

        Returns:
            List of retrieval results
        """
        query = RetrievalQuery(
            text=query_text,
            strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
            level=level,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filters=filters,
        )

        return self.retrieve(query)

    def keyword_search(
        self,
        query_text: str,
        level: RetrievalLevel = RetrievalLevel.MULTI_LEVEL,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Keyword-based search.

        Args:
            query_text: Search query
            level: Retrieval level
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of retrieval results
        """
        query = RetrievalQuery(
            text=query_text,
            strategy=RetrievalStrategy.KEYWORD_MATCH,
            level=level,
            top_k=top_k,
            filters=filters,
        )

        return self.retrieve(query)

    def hierarchical_search(
        self,
        query_text: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_context: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Hierarchical search with context aggregation.

        Args:
            query_text: Search query
            top_k: Number of results
            similarity_threshold: Minimum similarity
            include_context: Whether to include context
            filters: Optional filters

        Returns:
            List of retrieval results
        """
        query = RetrievalQuery(
            text=query_text,
            strategy=RetrievalStrategy.HIERARCHICAL,
            level=RetrievalLevel.MULTI_LEVEL,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            include_context=include_context,
            filters=filters,
        )

        return self.retrieve(query)

    def _semantic_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Semantic similarity-based retrieval."""
        # Generate query embedding
        query_embedding = self.embedding_service.embedder.embed_text(query.text)

        results = []

        # Get target nodes based on level
        target_nodes = self._get_target_nodes_by_level(query.level)

        # Filter nodes if filters provided
        if query.filters:
            target_nodes = self._apply_filters(target_nodes, query.filters)

        # Calculate similarities
        similarities = []
        for node in target_nodes:
            if node.embedding:
                similarity = self._cosine_similarity(query_embedding, node.embedding)
                if similarity >= query.similarity_threshold:
                    similarities.append((node, similarity))

        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)

        for node, similarity in similarities[: query.top_k]:
            result = self._create_retrieval_result(
                node, similarity, query.include_context
            )
            results.append(result)

        return results

    def _keyword_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Keyword-based retrieval."""
        results = []
        query_terms = query.text.lower().split()

        # Get target nodes based on level
        target_nodes = self._get_target_nodes_by_level(query.level)

        # Filter nodes if filters provided
        if query.filters:
            target_nodes = self._apply_filters(target_nodes, query.filters)

        # Calculate keyword relevance
        keyword_scores = []
        for node in target_nodes:
            content_lower = node.content.lower()
            score = sum(1 for term in query_terms if term in content_lower)
            if score > 0:
                # Normalize score by content length
                normalized_score = score / len(content_lower.split())
                keyword_scores.append((node, normalized_score))

        # Sort by score and take top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        for node, score in keyword_scores[: query.top_k]:
            result = self._create_retrieval_result(node, score, query.include_context)
            results.append(result)

        return results

    def _hybrid_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Hybrid retrieval combining semantic and keyword approaches."""
        # Get semantic results
        semantic_results = self._semantic_retrieval(query)

        # Get keyword results
        keyword_results = self._keyword_retrieval(query)

        # Combine and re-rank results
        combined_results = self._combine_results(semantic_results, keyword_results)

        return combined_results[: query.top_k]

    def _hierarchical_retrieval(self, query: RetrievalQuery) -> List[RetrievalResult]:
        """Hierarchical retrieval with context aggregation."""
        # Start with chunk-level search
        chunk_query = RetrievalQuery(
            text=query.text,
            strategy=RetrievalStrategy.SEMANTIC_SIMILARITY,
            level=RetrievalLevel.CHUNK_ONLY,
            top_k=query.top_k * 2,  # Get more chunks for context
            similarity_threshold=query.similarity_threshold,
            include_context=query.include_context,
            filters=query.filters,
        )

        chunk_results = self._semantic_retrieval(chunk_query)

        # Aggregate context from parent nodes
        aggregated_results = []
        for chunk_result in chunk_results:
            # Get parent context
            parent_context = self._get_parent_context(chunk_result.node_id)

            # Create aggregated result
            aggregated_result = RetrievalResult(
                node_id=chunk_result.node_id,
                node_type=chunk_result.node_type,
                content=chunk_result.content,
                metadata=chunk_result.metadata,
                relevance_score=chunk_result.relevance_score,
                context=(
                    f"{parent_context}\n\n{chunk_result.context}"
                    if parent_context
                    else chunk_result.context
                ),
                hierarchy_path=chunk_result.hierarchy_path,
                embedding_level=chunk_result.embedding_level,
            )

            aggregated_results.append(aggregated_result)

        return aggregated_results[: query.top_k]

    def _get_target_nodes_by_level(self, level: RetrievalLevel) -> List[HierarchyNode]:
        """Get target nodes based on retrieval level."""
        all_nodes = list(self.hierarchy_manager.nodes.values())

        if level == RetrievalLevel.CHUNK_ONLY:
            return [node for node in all_nodes if node.node_type == NodeType.CHUNK]
        elif level == RetrievalLevel.SECTION_ONLY:
            return [node for node in all_nodes if node.node_type == NodeType.SECTION]
        elif level == RetrievalLevel.DOCUMENT_ONLY:
            return [node for node in all_nodes if node.node_type == NodeType.DOCUMENT]
        else:  # MULTI_LEVEL
            return all_nodes

    def _apply_filters(
        self, nodes: List[HierarchyNode], filters: Dict[str, Any]
    ) -> List[HierarchyNode]:
        """Apply filters to nodes."""
        filtered_nodes = []

        for node in nodes:
            matches_filters = True

            for key, value in filters.items():
                if key in node.metadata:
                    if node.metadata[key] != value:
                        matches_filters = False
                        break
                else:
                    matches_filters = False
                    break

            if matches_filters:
                filtered_nodes.append(node)

        return filtered_nodes

    def _create_retrieval_result(
        self, node: HierarchyNode, relevance_score: float, include_context: bool
    ) -> RetrievalResult:
        """Create a retrieval result from a node."""
        context = ""
        if include_context:
            context = self._get_node_context(node)

        hierarchy_path = self._get_hierarchy_path(node)

        # Determine embedding level
        if node.node_type == NodeType.DOCUMENT:
            embedding_level = EmbeddingLevel.DOCUMENT
        elif node.node_type == NodeType.SECTION:
            embedding_level = EmbeddingLevel.SECTION
        else:
            embedding_level = EmbeddingLevel.CHUNK

        return RetrievalResult(
            node_id=node.id,
            node_type=node.node_type,
            content=node.content,
            metadata=node.metadata,
            relevance_score=relevance_score,
            context=context,
            hierarchy_path=hierarchy_path,
            embedding_level=embedding_level,
        )

    def _get_node_context(self, node: HierarchyNode) -> str:
        """Get context for a node."""
        context_parts = []

        # Add parent context
        if node.parent_id:
            parent = self.hierarchy_manager.get_node(node.parent_id)
            if parent:
                context_parts.append(f"Parent: {parent.content[:200]}...")

        # Add sibling context for chunks
        if node.node_type == NodeType.CHUNK and node.parent_id:
            siblings = self.hierarchy_manager.get_children(node.parent_id)
            sibling_contexts = []
            for sibling in siblings[:3]:  # Limit to 3 siblings
                if sibling.id != node.id:
                    sibling_contexts.append(sibling.content[:100])
            if sibling_contexts:
                context_parts.append(f"Siblings: {'; '.join(sibling_contexts)}")

        return "\n".join(context_parts)

    def _get_hierarchy_path(self, node: HierarchyNode) -> List[str]:
        """Get the hierarchy path to a node."""
        path = [node.id]
        current_id = node.parent_id

        while current_id:
            path.insert(0, current_id)
            parent = self.hierarchy_manager.get_node(current_id)
            if parent:
                current_id = parent.parent_id
            else:
                break

        return path

    def _get_parent_context(self, node_id: str) -> Optional[str]:
        """Get context from parent nodes."""
        node = self.hierarchy_manager.get_node(node_id)
        if not node or not node.parent_id:
            return None

        parent = self.hierarchy_manager.get_node(node.parent_id)
        if parent:
            return (
                parent.content[:300] + "..."
                if len(parent.content) > 300
                else parent.content
            )

        return None

    def _combine_results(
        self,
        semantic_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Combine and re-rank semantic and keyword results."""
        # Create a mapping of node_id to results
        result_map = {}

        # Add semantic results
        for result in semantic_results:
            result_map[result.node_id] = {
                "result": result,
                "semantic_score": result.relevance_score,
                "keyword_score": 0.0,
            }

        # Add keyword results
        for result in keyword_results:
            if result.node_id in result_map:
                result_map[result.node_id]["keyword_score"] = result.relevance_score
            else:
                result_map[result.node_id] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.relevance_score,
                }

        # Calculate combined scores (simple average)
        combined_results = []
        for node_id, scores in result_map.items():
            combined_score = (scores["semantic_score"] + scores["keyword_score"]) / 2
            result = scores["result"]
            result.relevance_score = combined_score
            combined_results.append(result)

        # Sort by combined score
        combined_results.sort(key=lambda x: x.relevance_score, reverse=True)

        return combined_results

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
