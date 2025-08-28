from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from typing import Dict, List, Any, Optional
import requests
from datetime import datetime
from src.config import Config
from src.common.logger.logger import get_logger

logger = get_logger(__name__)

NOTION_VERSION = "2022-06-28"


def create_notion_page(
    title: str, content: str, parent_id: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new Notion page with structured content and metadata.
    """
    try:
        url = "https://api.notion.com/v1/pages"
        headers = {
            "Authorization": f"Bearer {Config.NOTION_API_KEY}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }

        # Prepare page properties
        properties = {
            "title": {"title": [{"type": "text", "text": {"content": title}}]}
        }

        # Add custom properties based on metadata
        if metadata.get("category"):
            properties["Category"] = {"select": {"name": metadata["category"]}}

        if metadata.get("tags"):
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in metadata["tags"]]
            }

        if metadata.get("priority"):
            properties["Priority"] = {"select": {"name": metadata["priority"]}}

        body = {
            "parent": {"page_id": parent_id},
            "icon": {"emoji": metadata.get("emoji", "📝")},
            "properties": properties,
        }

        response = requests.post(url, json=body, headers=headers)

        if response.status_code == 200:
            page_data = response.json()
            page_id = page_data["id"]

            # Add content blocks
            add_content_blocks(page_id, content, metadata)

            logger.info(f"Notion page created successfully: {page_id}")
            return {
                "page_id": page_id,
                "url": f"https://notion.so/{page_id.replace('-', '')}",
                "status": "created",
            }
        else:
            raise Exception(
                f"Failed to create page. Status: {response.status_code}\n{response.text}"
            )

    except Exception as e:
        logger.error(f"Error creating Notion page: {e}")
        raise


def add_content_blocks(page_id: str, content: str, metadata: Dict[str, Any]) -> bool:
    """
    Add structured content blocks to the Notion page.
    """
    try:
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        headers = {
            "Authorization": f"Bearer {Config.NOTION_API_KEY}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }

        blocks = []

        # Add summary if available
        if metadata.get("summary"):
            blocks.append(
                {
                    "object": "block",
                    "type": "callout",
                    "callout": {
                        "rich_text": [
                            {"type": "text", "text": {"content": metadata["summary"]}}
                        ],
                        "icon": {"emoji": "💡"},
                    },
                }
            )

        # Add key points if available
        if metadata.get("key_points"):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [
                            {"type": "text", "text": {"content": "Key Points"}}
                        ]
                    },
                }
            )

            for point in metadata["key_points"]:
                blocks.append(
                    {
                        "object": "block",
                        "type": "bulleted_list_item",
                        "bulleted_list_item": {
                            "rich_text": [{"type": "text", "text": {"content": point}}]
                        },
                    }
                )

        # Add main content
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            }
        )

        # Add metadata section
        if metadata.get("entities") or metadata.get("related_topics"):
            blocks.append(
                {
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [
                            {"type": "text", "text": {"content": "Related Information"}}
                        ]
                    },
                }
            )

            if metadata.get("entities"):
                blocks.append(
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {
                                    "type": "text",
                                    "text": {
                                        "content": f"Entities: {', '.join(metadata['entities'])}"
                                    },
                                }
                            ]
                        },
                    }
                )

        body = {"children": blocks}
        response = requests.patch(url, json=body, headers=headers)

        if response.status_code == 200:
            logger.info(f"Content blocks added to page {page_id}")
            return True
        else:
            raise Exception(
                f"Failed to add content blocks. Status: {response.status_code}"
            )

    except Exception as e:
        logger.error(f"Error adding content blocks: {e}")
        raise


def update_notion_page(page_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update an existing Notion page with new information.
    """
    try:
        url = f"https://api.notion.com/v1/pages/{page_id}"
        headers = {
            "Authorization": f"Bearer {Config.NOTION_API_KEY}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }

        body = {"properties": {}}

        # Update properties based on updates dict
        if "title" in updates:
            body["properties"]["title"] = {
                "title": [{"type": "text", "text": {"content": updates["title"]}}]
            }

        if "tags" in updates:
            body["properties"]["Tags"] = {
                "multi_select": [{"name": tag} for tag in updates["tags"]]
            }

        response = requests.patch(url, json=body, headers=headers)

        if response.status_code == 200:
            logger.info(f"Page {page_id} updated successfully")
            return True
        else:
            raise Exception(f"Failed to update page. Status: {response.status_code}")

    except Exception as e:
        logger.error(f"Error updating Notion page: {e}")
        raise


def create_notion_database(
    parent_id: str, title: str, properties: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a new Notion database for organizing related pages.
    """
    try:
        url = "https://api.notion.com/v1/databases"
        headers = {
            "Authorization": f"Bearer {Config.NOTION_API_KEY}",
            "Content-Type": "application/json",
            "Notion-Version": NOTION_VERSION,
        }

        body = {
            "parent": {"page_id": parent_id},
            "title": [{"type": "text", "text": {"content": title}}],
            "properties": properties,
        }

        response = requests.post(url, json=body, headers=headers)

        if response.status_code == 200:
            database_data = response.json()
            logger.info(f"Notion database created: {database_data['id']}")
            return {
                "database_id": database_data["id"],
                "url": f"https://notion.so/{database_data['id'].replace('-', '')}",
                "status": "created",
            }
        else:
            raise Exception(
                f"Failed to create database. Status: {response.status_code}"
            )

    except Exception as e:
        logger.error(f"Error creating Notion database: {e}")
        raise


# Create the Notion builder agent
notion_builder_agent = Agent(
    name="notion_builder",
    model="gemini-2.0-flash",
    description="Specialized agent for creating and managing Notion pages and databases",
    instruction="""
    You are a Notion page building specialist. Your role is to:
    1. Create well-structured Notion pages with appropriate formatting
    2. Organize content using Notion's block system
    3. Add metadata and properties to pages
    4. Create databases for organizing related content
    5. Update existing pages with new information
    
    Always ensure pages are well-organized and follow Notion best practices.
    Use appropriate icons, callouts, and formatting to enhance readability.
    """,
    tools=[
        FunctionTool(
            name="create_notion_page",
            func=create_notion_page,
            description="Create a new Notion page with structured content and metadata",
        ),
        FunctionTool(
            name="add_content_blocks",
            func=add_content_blocks,
            description="Add structured content blocks to an existing Notion page",
        ),
        FunctionTool(
            name="update_notion_page",
            func=update_notion_page,
            description="Update an existing Notion page with new information",
        ),
        FunctionTool(
            name="create_notion_database",
            func=create_notion_database,
            description="Create a new Notion database for organizing related pages",
        ),
    ],
)
