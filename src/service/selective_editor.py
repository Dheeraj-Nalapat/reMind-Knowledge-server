"""
Selective Editor for targeted updates to hierarchical RAG data.
Supports selective editing with propagation to child nodes and embedding updates.
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from src.common.logger.logger import get_logger
from .document_hierarchy import HierarchyNode, NodeType, DocumentHierarchyManager
from .hierarchical_embedding import HierarchicalEmbeddingService, EmbeddingLevel

logger = get_logger(__name__)


class EditScope(Enum):
    """Scope of edit operations."""

    NODE_ONLY = "node_only"
    CHILDREN_ONLY = "children_only"
    DESCENDANTS_ONLY = "descendants_only"
    FULL_HIERARCHY = "full_hierarchy"


class EditType(Enum):
    """Types of edit operations."""

    CONTENT_UPDATE = "content_update"
    METADATA_UPDATE = "metadata_update"
    STRUCTURE_CHANGE = "structure_change"
    EMBEDDING_UPDATE = "embedding_update"


@dataclass
class EditOperation:
    """Represents an edit operation."""

    node_id: str
    edit_type: EditType
    scope: EditScope
    changes: Dict[str, Any]
    timestamp: str
    user_id: Optional[str] = None


@dataclass
class EditResult:
    """Result of an edit operation."""

    success: bool
    affected_nodes: List[str]
    updated_embeddings: List[str]
    errors: List[str]
    operation: EditOperation


class SelectiveEditor:
    """
    Selective editor for hierarchical RAG data with propagation support.
    """

    def __init__(
        self,
        hierarchy_manager: DocumentHierarchyManager,
        embedding_service: HierarchicalEmbeddingService,
    ):
        self.hierarchy_manager = hierarchy_manager
        self.embedding_service = embedding_service
        self.edit_history: List[EditOperation] = []

    def edit_node_content(
        self,
        node_id: str,
        new_content: str,
        scope: EditScope = EditScope.NODE_ONLY,
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> EditResult:
        """
        Edit node content with optional propagation.

        Args:
            node_id: Node to edit
            new_content: New content
            scope: Scope of the edit
            update_embeddings: Whether to update embeddings
            user_id: User performing the edit

        Returns:
            Edit result
        """
        operation = EditOperation(
            node_id=node_id,
            edit_type=EditType.CONTENT_UPDATE,
            scope=scope,
            changes={"content": new_content},
            timestamp=self._get_timestamp(),
            user_id=user_id,
        )

        affected_nodes = []
        errors = []
        updated_embeddings = []

        try:
            # Get target nodes based on scope
            target_nodes = self._get_target_nodes(node_id, scope)

            # Update content for all target nodes
            for node in target_nodes:
                if self.hierarchy_manager.update_node_content(node.id, new_content):
                    affected_nodes.append(node.id)

                    # Update embeddings if requested
                    if update_embeddings:
                        try:
                            self._update_node_embedding(node)
                            updated_embeddings.append(node.id)
                        except Exception as e:
                            errors.append(
                                f"Failed to update embedding for {node.id}: {e}"
                            )
                else:
                    errors.append(f"Failed to update content for {node.id}")

            success = len(errors) == 0
            self.edit_history.append(operation)

            logger.info(f"Content edit completed for {len(affected_nodes)} nodes")

        except Exception as e:
            errors.append(f"Edit operation failed: {e}")
            success = False
            logger.error(f"Content edit failed: {e}")

        return EditResult(
            success=success,
            affected_nodes=affected_nodes,
            updated_embeddings=updated_embeddings,
            errors=errors,
            operation=operation,
        )

    def edit_node_metadata(
        self,
        node_id: str,
        metadata_updates: Dict[str, Any],
        scope: EditScope = EditScope.NODE_ONLY,
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> EditResult:
        """
        Edit node metadata with optional propagation.

        Args:
            node_id: Node to edit
            metadata_updates: Metadata changes
            scope: Scope of the edit
            update_embeddings: Whether to update embeddings
            user_id: User performing the edit

        Returns:
            Edit result
        """
        operation = EditOperation(
            node_id=node_id,
            edit_type=EditType.METADATA_UPDATE,
            scope=scope,
            changes={"metadata": metadata_updates},
            timestamp=self._get_timestamp(),
            user_id=user_id,
        )

        affected_nodes = []
        errors = []
        updated_embeddings = []

        try:
            # Get target nodes based on scope
            target_nodes = self._get_target_nodes(node_id, scope)

            # Update metadata for all target nodes
            for node in target_nodes:
                if self.hierarchy_manager.update_node_content(
                    node.id, node.content, metadata_updates
                ):
                    affected_nodes.append(node.id)

                    # Update embeddings if requested
                    if update_embeddings:
                        try:
                            self._update_node_embedding(node)
                            updated_embeddings.append(node.id)
                        except Exception as e:
                            errors.append(
                                f"Failed to update embedding for {node.id}: {e}"
                            )
                else:
                    errors.append(f"Failed to update metadata for {node.id}")

            success = len(errors) == 0
            self.edit_history.append(operation)

            logger.info(f"Metadata edit completed for {len(affected_nodes)} nodes")

        except Exception as e:
            errors.append(f"Edit operation failed: {e}")
            success = False
            logger.error(f"Metadata edit failed: {e}")

        return EditResult(
            success=success,
            affected_nodes=affected_nodes,
            updated_embeddings=updated_embeddings,
            errors=errors,
            operation=operation,
        )

    def restructure_hierarchy(
        self,
        node_id: str,
        new_parent_id: Optional[str],
        scope: EditScope = EditScope.NODE_ONLY,
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> EditResult:
        """
        Restructure hierarchy by moving nodes.

        Args:
            node_id: Node to move
            new_parent_id: New parent ID (None for root)
            scope: Scope of the edit
            update_embeddings: Whether to update embeddings
            user_id: User performing the edit

        Returns:
            Edit result
        """
        operation = EditOperation(
            node_id=node_id,
            edit_type=EditType.STRUCTURE_CHANGE,
            scope=scope,
            changes={"new_parent_id": new_parent_id},
            timestamp=self._get_timestamp(),
            user_id=user_id,
        )

        affected_nodes = []
        errors = []
        updated_embeddings = []

        try:
            # Get target nodes based on scope
            target_nodes = self._get_target_nodes(node_id, scope)

            # Move nodes to new parent
            for node in target_nodes:
                if self._move_node(node.id, new_parent_id):
                    affected_nodes.append(node.id)

                    # Update embeddings if requested
                    if update_embeddings:
                        try:
                            self._update_node_embedding(node)
                            updated_embeddings.append(node.id)
                        except Exception as e:
                            errors.append(
                                f"Failed to update embedding for {node.id}: {e}"
                            )
                else:
                    errors.append(f"Failed to move node {node.id}")

            success = len(errors) == 0
            self.edit_history.append(operation)

            logger.info(f"Structure change completed for {len(affected_nodes)} nodes")

        except Exception as e:
            errors.append(f"Structure change failed: {e}")
            success = False
            logger.error(f"Structure change failed: {e}")

        return EditResult(
            success=success,
            affected_nodes=affected_nodes,
            updated_embeddings=updated_embeddings,
            errors=errors,
            operation=operation,
        )

    def bulk_edit(
        self,
        edits: List[Dict[str, Any]],
        update_embeddings: bool = True,
        user_id: Optional[str] = None,
    ) -> List[EditResult]:
        """
        Perform multiple edits in sequence.

        Args:
            edits: List of edit specifications
            update_embeddings: Whether to update embeddings
            user_id: User performing the edits

        Returns:
            List of edit results
        """
        results = []

        for edit_spec in edits:
            edit_type = edit_spec.get("type")
            node_id = edit_spec.get("node_id")
            scope = EditScope(edit_spec.get("scope", "node_only"))

            if edit_type == "content":
                result = self.edit_node_content(
                    node_id=node_id,
                    new_content=edit_spec["content"],
                    scope=scope,
                    update_embeddings=update_embeddings,
                    user_id=user_id,
                )
            elif edit_type == "metadata":
                result = self.edit_node_metadata(
                    node_id=node_id,
                    metadata_updates=edit_spec["metadata"],
                    scope=scope,
                    update_embeddings=update_embeddings,
                    user_id=user_id,
                )
            elif edit_type == "structure":
                result = self.restructure_hierarchy(
                    node_id=node_id,
                    new_parent_id=edit_spec.get("new_parent_id"),
                    scope=scope,
                    update_embeddings=update_embeddings,
                    user_id=user_id,
                )
            else:
                result = EditResult(
                    success=False,
                    affected_nodes=[],
                    updated_embeddings=[],
                    errors=[f"Unknown edit type: {edit_type}"],
                    operation=EditOperation(
                        node_id=node_id or "",
                        edit_type=EditType.CONTENT_UPDATE,
                        scope=scope,
                        changes={},
                        timestamp=self._get_timestamp(),
                        user_id=user_id,
                    ),
                )

            results.append(result)

        return results

    def get_edit_history(
        self,
        node_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[EditOperation]:
        """
        Get edit history with optional filtering.

        Args:
            node_id: Filter by node ID
            user_id: Filter by user ID
            limit: Maximum number of results

        Returns:
            List of edit operations
        """
        history = self.edit_history

        if node_id:
            history = [op for op in history if op.node_id == node_id]

        if user_id:
            history = [op for op in history if op.user_id == user_id]

        return history[-limit:]

    def revert_edit(
        self, operation_index: int, update_embeddings: bool = True
    ) -> EditResult:
        """
        Revert a specific edit operation.

        Args:
            operation_index: Index of operation to revert
            update_embeddings: Whether to update embeddings

        Returns:
            Edit result
        """
        if operation_index >= len(self.edit_history):
            return EditResult(
                success=False,
                affected_nodes=[],
                updated_embeddings=[],
                errors=["Invalid operation index"],
                operation=EditOperation(
                    node_id="",
                    edit_type=EditType.CONTENT_UPDATE,
                    scope=EditScope.NODE_ONLY,
                    changes={},
                    timestamp=self._get_timestamp(),
                ),
            )

        # This would implement revert logic based on the original operation
        # For now, return a placeholder result
        return EditResult(
            success=True,
            affected_nodes=[],
            updated_embeddings=[],
            errors=[],
            operation=self.edit_history[operation_index],
        )

    def _get_target_nodes(self, node_id: str, scope: EditScope) -> List[HierarchyNode]:
        """Get target nodes based on edit scope."""
        node = self.hierarchy_manager.get_node(node_id)
        if not node:
            return []

        target_nodes = [node]

        if scope == EditScope.CHILDREN_ONLY:
            target_nodes = self.hierarchy_manager.get_children(node_id)
        elif scope == EditScope.DESCENDANTS_ONLY:
            target_nodes = self.hierarchy_manager.get_descendants(node_id)
        elif scope == EditScope.FULL_HIERARCHY:
            # Include node and all descendants
            target_nodes = [node] + self.hierarchy_manager.get_descendants(node_id)

        return target_nodes

    def _update_node_embedding(self, node: HierarchyNode):
        """Update embedding for a node based on its type."""
        if node.node_type == NodeType.DOCUMENT:
            self.embedding_service.generate_document_embedding(node)
        elif node.node_type == NodeType.SECTION:
            self.embedding_service.generate_section_embedding(node)
        elif node.node_type == NodeType.CHUNK:
            self.embedding_service.generate_chunk_embedding(node)

    def _move_node(self, node_id: str, new_parent_id: Optional[str]) -> bool:
        """Move a node to a new parent."""
        # This would implement the actual move logic
        # For now, return True as placeholder
        return True

    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime

        return datetime.utcnow().isoformat()
