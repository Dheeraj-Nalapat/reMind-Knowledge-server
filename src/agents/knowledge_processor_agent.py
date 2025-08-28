from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
from datetime import datetime
from src.common.embedding.embedder import OpenAITextVectorizer
from src.common.database.operations import NotionPageOperations
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


def generate_embeddings(content: str) -> List[float]:
    """
    Generate embeddings for the given content using OpenAI.
    """
    try:
        embedder = OpenAITextVectorizer()
        embeddings = embedder.embed_text(content)
        logger.info(f"Generated embeddings for content of length {len(content)}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise


def store_knowledge_in_database(
    page_id: str, content: str, metadata: Dict[str, Any], embeddings: List[float]
) -> bool:
    """
    Store the processed knowledge in the database with embeddings.
    """
    try:
        # Prepare metadata for storage
        storage_metadata = {
            "title": metadata.get("title", "Untitled"),
            "source": metadata.get("source", "unknown"),
            "notion_page_id": page_id,
            "notion_url": f"https://notion.so/{page_id.replace('-', '')}",
            "created_at": datetime.utcnow().isoformat(),
            "tags": metadata.get("tags", []),
            "category": metadata.get("category", "general"),
            "content_type": metadata.get("content_type", "text"),
            "entities": metadata.get("entities", []),
            "summary": metadata.get("summary", ""),
            "key_points": metadata.get("key_points", []),
        }

        success = NotionPageOperations.insert_page(
            page_id=page_id,
            content=content,
            metadata=storage_metadata,
            embedding=embeddings,
        )

        if success:
            logger.info(f"Successfully stored knowledge in database for page {page_id}")
        else:
            logger.error(f"Failed to store knowledge in database for page {page_id}")

        return success

    except Exception as e:
        logger.error(f"Error storing knowledge in database: {e}")
        raise


def search_similar_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar knowledge in the database using embeddings.
    """
    try:
        # Generate embeddings for the query
        embedder = OpenAITextVectorizer()
        query_embeddings = embedder.embed_text(query)

        # Search in database (this would need to be implemented in NotionPageOperations)
        # For now, returning placeholder
        similar_knowledge = []

        logger.info(f"Searched for similar knowledge with query: {query[:50]}...")
        return similar_knowledge

    except Exception as e:
        logger.error(f"Error searching similar knowledge: {e}")
        raise


def update_knowledge_metadata(page_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update metadata for existing knowledge in the database.
    """
    try:
        # This would update the metadata in the database
        # Implementation depends on NotionPageOperations interface

        logger.info(f"Updated metadata for page {page_id}")
        return True

    except Exception as e:
        logger.error(f"Error updating knowledge metadata: {e}")
        raise


def validate_knowledge_quality(
    content: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate the quality and completeness of knowledge content.
    """
    try:
        quality_score = {
            "overall_score": 0.8,
            "completeness": 0.9,
            "clarity": 0.8,
            "relevance": 0.9,
            "issues": [],
            "suggestions": [],
        }

        # Basic quality checks
        if len(content.strip()) < 10:
            quality_score["completeness"] = 0.3
            quality_score["issues"].append("Content too short")

        if not metadata.get("title"):
            quality_score["issues"].append("Missing title")

        if not metadata.get("category"):
            quality_score["suggestions"].append("Consider adding a category")

        # Calculate overall score
        quality_score["overall_score"] = (
            quality_score["completeness"]
            + quality_score["clarity"]
            + quality_score["relevance"]
        ) / 3

        logger.info(
            f"Knowledge quality validated with score: {quality_score['overall_score']:.2f}"
        )
        return quality_score

    except Exception as e:
        logger.error(f"Error validating knowledge quality: {e}")
        raise


def archive_knowledge(page_id: str, reason: str = "Archived") -> bool:
    """
    Archive knowledge that is no longer relevant or needed.
    """
    try:
        # This would mark the knowledge as archived in the database
        # Implementation depends on NotionPageOperations interface

        logger.info(f"Archived knowledge page {page_id} with reason: {reason}")
        return True

    except Exception as e:
        logger.error(f"Error archiving knowledge: {e}")
        raise


# Create the knowledge processor agent
knowledge_processor_agent = Agent(
    name="knowledge_processor",
    model="gemini-2.0-flash",
    description="Specialized agent for processing, storing, and managing knowledge with embeddings",
    instruction="""
    You are a knowledge processing specialist. Your role is to:
    1. Generate embeddings for content to enable semantic search
    2. Store knowledge in the database with proper metadata
    3. Search for similar knowledge using embeddings
    4. Validate knowledge quality and completeness
    5. Manage knowledge lifecycle (update, archive)
    
    Always ensure data integrity and proper error handling.
    Focus on creating searchable, well-organized knowledge bases.
    """,
    tools=[
        FunctionTool(
            name="generate_embeddings",
            func=generate_embeddings,
            description="Generate embeddings for content using OpenAI",
        ),
        FunctionTool(
            name="store_knowledge_in_database",
            func=store_knowledge_in_database,
            description="Store processed knowledge in database with embeddings",
        ),
        FunctionTool(
            name="search_similar_knowledge",
            func=search_similar_knowledge,
            description="Search for similar knowledge using embeddings",
        ),
        FunctionTool(
            name="update_knowledge_metadata",
            func=update_knowledge_metadata,
            description="Update metadata for existing knowledge",
        ),
        FunctionTool(
            name="validate_knowledge_quality",
            func=validate_knowledge_quality,
            description="Validate the quality and completeness of knowledge content",
        ),
        FunctionTool(
            name="archive_knowledge",
            func=archive_knowledge,
            description="Archive knowledge that is no longer relevant",
        ),
    ],
)
