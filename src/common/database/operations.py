import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from psycopg2.extras import Json

from src.common.database.connection import (get_db_connection,
                                            return_db_connection)
from src.common.logger.logger import get_logger

logger = get_logger(__name__)


class NotionPageOperations:

    @staticmethod
    def insert_page(
        page_id: str, content: str, metadata: Dict[str, Any], embedding: List[float]
    ) -> bool:
        """
        Insert a new notion page with embedding into the database

        Args:
            page_id: The Notion page ID
            content: The page content text
            metadata: JSON metadata about the page
            embedding: Vector embedding of the content (1536 dimensions)

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Convert string page_id to UUID if needed, then back to string for psycopg2
            if isinstance(page_id, str):
                page_uuid = uuid.UUID(page_id)
            else:
                page_uuid = page_id

            # Convert UUID to string for psycopg2
            page_uuid_str = str(page_uuid)

            query = """
                INSERT INTO notion_pages (page_id, content, metadata, embedding, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (page_id) 
                DO UPDATE SET 
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = EXCLUDED.updated_at
            """

            current_time = datetime.utcnow()
            cursor.execute(
                query,
                (
                    page_uuid_str,
                    content,
                    Json(metadata),
                    embedding,
                    current_time,
                    current_time,
                ),
            )

            conn.commit()
            logger.info(f"Successfully inserted/updated page: {page_id}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to insert page {page_id}: {e}")
            return False
        finally:
            if conn:
                return_db_connection(conn)

    @staticmethod
    def get_page_by_id(page_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a page by its ID

        Args:
            page_id: The Notion page ID

        Returns:
            Dict containing page data or None if not found
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if isinstance(page_id, str):
                page_uuid = uuid.UUID(page_id)
            else:
                page_uuid = page_id

            # Convert UUID to string for psycopg2
            page_uuid_str = str(page_uuid)

            query = (
                "SELECT * FROM notion_pages WHERE page_id = %s AND deleted_at IS NULL"
            )
            cursor.execute(query, (page_uuid_str,))

            result = cursor.fetchone()
            if result:
                # Convert RealDictRow to regular dict
                return dict(result)
            return None

        except Exception as e:
            logger.error(f"Failed to get page {page_id}: {e}")
            return None
        finally:
            if conn:
                return_db_connection(conn)

    @staticmethod
    def search_similar_pages(
        embedding: List[float], limit: int = 5, similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for pages similar to the given embedding using cosine similarity

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of similar pages with similarity scores
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT 
                    page_id,
                    content,
                    metadata,
                    embedding,
                    created_at,
                    updated_at,
                    1 - (embedding <=> %s) as similarity
                FROM notion_pages 
                WHERE deleted_at IS NULL
                AND 1 - (embedding <=> %s) > %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            cursor.execute(
                query, (embedding, embedding, similarity_threshold, embedding, limit)
            )
            results = cursor.fetchall()

            # Convert RealDictRow objects to regular dicts
            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to search similar pages: {e}")
            return []
        finally:
            if conn:
                return_db_connection(conn)

    @staticmethod
    def update_page(
        page_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Update an existing page

        Args:
            page_id: The Notion page ID
            content: New content (optional)
            metadata: New metadata (optional)
            embedding: New embedding (optional)

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if isinstance(page_id, str):
                page_uuid = uuid.UUID(page_id)
            else:
                page_uuid = page_id

            # Convert UUID to string for psycopg2
            page_uuid_str = str(page_uuid)

            # Build dynamic update query
            update_parts = []
            params = []

            if content is not None:
                update_parts.append("content = %s")
                params.append(content)

            if metadata is not None:
                update_parts.append("metadata = %s")
                params.append(Json(metadata))

            if embedding is not None:
                update_parts.append("embedding = %s")
                params.append(embedding)

            if not update_parts:
                logger.warning(f"No fields to update for page {page_id}")
                return False

            update_parts.append("updated_at = %s")
            params.append(datetime.utcnow())
            params.append(page_uuid_str)

            query = f"""
                UPDATE notion_pages 
                SET {', '.join(update_parts)}
                WHERE page_id = %s AND deleted_at IS NULL
            """

            cursor.execute(query, params)

            if cursor.rowcount == 0:
                logger.warning(f"Page {page_id} not found or already deleted")
                return False

            conn.commit()
            logger.info(f"Successfully updated page: {page_id}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to update page {page_id}: {e}")
            return False
        finally:
            if conn:
                return_db_connection(conn)

    @staticmethod
    def soft_delete_page(page_id: str) -> bool:
        """
        Soft delete a page by setting deleted_at timestamp

        Args:
            page_id: The Notion page ID

        Returns:
            bool: True if successful, False otherwise
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            if isinstance(page_id, str):
                page_uuid = uuid.UUID(page_id)
            else:
                page_uuid = page_id

            # Convert UUID to string for psycopg2
            page_uuid_str = str(page_uuid)

            query = """
                UPDATE notion_pages 
                SET deleted_at = %s
                WHERE page_id = %s AND deleted_at IS NULL
            """

            cursor.execute(query, (datetime.utcnow(), page_uuid_str))

            if cursor.rowcount == 0:
                logger.warning(f"Page {page_id} not found or already deleted")
                return False

            conn.commit()
            logger.info(f"Successfully soft deleted page: {page_id}")
            return True

        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to soft delete page {page_id}: {e}")
            return False
        finally:
            if conn:
                return_db_connection(conn)

    @staticmethod
    def get_all_pages(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all non-deleted pages with pagination

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of pages
        """
        conn = None
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT * FROM notion_pages 
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """

            cursor.execute(query, (limit, offset))
            results = cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get all pages: {e}")
            return []
        finally:
            if conn:
                return_db_connection(conn)
