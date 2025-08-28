"""
Document Hierarchy Manager for organizing documents in a tree structure.
Supports document → sections → chunks hierarchy with metadata tracking.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import uuid
from dataclasses import dataclass, field
from enum import Enum

from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """Types of nodes in the document hierarchy."""

    DOCUMENT = "document"
    SECTION = "section"
    CHUNK = "chunk"


@dataclass
class HierarchyNode:
    """Represents a node in the document hierarchy."""

    id: str
    node_type: NodeType
    content: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None
    is_deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for storage."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding": self.embedding,
            "is_deleted": self.is_deleted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchyNode":
        """Create node from dictionary."""
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            content=data["content"],
            metadata=data["metadata"],
            parent_id=data.get("parent_id"),
            children_ids=set(data.get("children_ids", [])),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            embedding=data.get("embedding"),
            is_deleted=data.get("is_deleted", False),
        )


class DocumentHierarchyManager:
    """
    Manages the hierarchical structure of documents.
    Supports document → sections → chunks organization.
    """

    def __init__(self):
        self.nodes: Dict[str, HierarchyNode] = {}
        self.document_roots: Set[str] = set()

    def create_document(
        self, content: str, metadata: Dict[str, Any], document_id: Optional[str] = None
    ) -> str:
        """
        Create a new document node.

        Args:
            content: Document content
            metadata: Document metadata
            document_id: Optional custom ID

        Returns:
            Document ID
        """
        doc_id = document_id or str(uuid.uuid4())

        node = HierarchyNode(
            id=doc_id, node_type=NodeType.DOCUMENT, content=content, metadata=metadata
        )

        self.nodes[doc_id] = node
        self.document_roots.add(doc_id)
        logger.info(f"Created document: {doc_id}")

        return doc_id

    def add_section(
        self,
        document_id: str,
        content: str,
        metadata: Dict[str, Any],
        section_id: Optional[str] = None,
    ) -> str:
        """
        Add a section to a document.

        Args:
            document_id: Parent document ID
            content: Section content
            metadata: Section metadata
            section_id: Optional custom ID

        Returns:
            Section ID
        """
        if document_id not in self.nodes:
            raise ValueError(f"Document {document_id} not found")

        if self.nodes[document_id].node_type != NodeType.DOCUMENT:
            raise ValueError(f"Node {document_id} is not a document")

        section_id = section_id or str(uuid.uuid4())

        section_node = HierarchyNode(
            id=section_id,
            node_type=NodeType.SECTION,
            content=content,
            metadata=metadata,
            parent_id=document_id,
        )

        # Update parent's children
        self.nodes[document_id].children_ids.add(section_id)
        self.nodes[document_id].updated_at = datetime.utcnow()

        self.nodes[section_id] = section_node
        logger.info(f"Added section {section_id} to document {document_id}")

        return section_id

    def add_chunk(
        self,
        parent_id: str,
        content: str,
        metadata: Dict[str, Any],
        chunk_id: Optional[str] = None,
    ) -> str:
        """
        Add a chunk to a section or document.

        Args:
            parent_id: Parent section or document ID
            content: Chunk content
            metadata: Chunk metadata
            chunk_id: Optional custom ID

        Returns:
            Chunk ID
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")

        parent_type = self.nodes[parent_id].node_type
        if parent_type not in [NodeType.DOCUMENT, NodeType.SECTION]:
            raise ValueError(f"Parent node {parent_id} is not a document or section")

        chunk_id = chunk_id or str(uuid.uuid4())

        chunk_node = HierarchyNode(
            id=chunk_id,
            node_type=NodeType.CHUNK,
            content=content,
            metadata=metadata,
            parent_id=parent_id,
        )

        # Update parent's children
        self.nodes[parent_id].children_ids.add(chunk_id)
        self.nodes[parent_id].updated_at = datetime.utcnow()

        self.nodes[chunk_id] = chunk_node
        logger.info(f"Added chunk {chunk_id} to parent {parent_id}")

        return chunk_id

    def get_node(self, node_id: str) -> Optional[HierarchyNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[HierarchyNode]:
        """Get all children of a node."""
        if node_id not in self.nodes:
            return []

        children = []
        for child_id in self.nodes[node_id].children_ids:
            if child_id in self.nodes:
                children.append(self.nodes[child_id])

        return children

    def get_descendants(self, node_id: str) -> List[HierarchyNode]:
        """Get all descendants of a node (recursive)."""
        descendants = []

        def collect_descendants(current_id: str):
            children = self.get_children(current_id)
            for child in children:
                descendants.append(child)
                collect_descendants(child.id)

        collect_descendants(node_id)
        return descendants

    def get_ancestors(self, node_id: str) -> List[HierarchyNode]:
        """Get all ancestors of a node."""
        ancestors = []
        current_id = node_id

        while current_id in self.nodes:
            parent_id = self.nodes[current_id].parent_id
            if parent_id and parent_id in self.nodes:
                ancestors.append(self.nodes[parent_id])
                current_id = parent_id
            else:
                break

        return ancestors

    def update_node_content(
        self,
        node_id: str,
        new_content: str,
        update_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update node content and optionally metadata.

        Args:
            node_id: Node ID to update
            new_content: New content
            update_metadata: Optional metadata updates

        Returns:
            Success status
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return False

        node = self.nodes[node_id]
        node.content = new_content
        node.updated_at = datetime.utcnow()

        if update_metadata:
            node.metadata.update(update_metadata)

        logger.info(f"Updated node {node_id}")
        return True

    def delete_node(self, node_id: str, recursive: bool = True) -> bool:
        """
        Delete a node and optionally its descendants.

        Args:
            node_id: Node ID to delete
            recursive: Whether to delete descendants

        Returns:
            Success status
        """
        if node_id not in self.nodes:
            logger.error(f"Node {node_id} not found")
            return False

        if recursive:
            # Delete all descendants first
            descendants = self.get_descendants(node_id)
            for descendant in descendants:
                self.nodes[descendant.id].is_deleted = True
                self.nodes[descendant.id].updated_at = datetime.utcnow()

        # Mark node as deleted
        self.nodes[node_id].is_deleted = True
        self.nodes[node_id].updated_at = datetime.utcnow()

        # Remove from parent's children
        parent_id = self.nodes[node_id].parent_id
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children_ids.discard(node_id)
            self.nodes[parent_id].updated_at = datetime.utcnow()

        # Remove from document roots if it's a document
        if node_id in self.document_roots:
            self.document_roots.discard(node_id)

        logger.info(f"Deleted node {node_id}")
        return True

    def get_document_structure(self, document_id: str) -> Dict[str, Any]:
        """
        Get the complete structure of a document.

        Args:
            document_id: Document ID

        Returns:
            Document structure as nested dictionary
        """
        if document_id not in self.nodes:
            return {}

        def build_structure(node_id: str) -> Dict[str, Any]:
            node = self.nodes[node_id]
            if node.is_deleted:
                return {}

            structure = {
                "id": node.id,
                "type": node.node_type.value,
                "content": node.content,
                "metadata": node.metadata,
                "created_at": node.created_at.isoformat(),
                "updated_at": node.updated_at.isoformat(),
            }

            children = []
            for child_id in node.children_ids:
                if child_id in self.nodes and not self.nodes[child_id].is_deleted:
                    child_structure = build_structure(child_id)
                    if child_structure:
                        children.append(child_structure)

            if children:
                structure["children"] = children

            return structure

        return build_structure(document_id)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document structures."""
        documents = []
        for doc_id in self.document_roots:
            if doc_id in self.nodes and not self.nodes[doc_id].is_deleted:
                doc_structure = self.get_document_structure(doc_id)
                if doc_structure:
                    documents.append(doc_structure)

        return documents

    def search_nodes(
        self,
        query: str,
        node_types: Optional[List[NodeType]] = None,
        include_deleted: bool = False,
    ) -> List[HierarchyNode]:
        """
        Search nodes by content.

        Args:
            query: Search query
            node_types: Filter by node types
            include_deleted: Whether to include deleted nodes

        Returns:
            List of matching nodes
        """
        results = []
        query_lower = query.lower()

        for node in self.nodes.values():
            if not include_deleted and node.is_deleted:
                continue

            if node_types and node.node_type not in node_types:
                continue

            if query_lower in node.content.lower():
                results.append(node)

        return results
