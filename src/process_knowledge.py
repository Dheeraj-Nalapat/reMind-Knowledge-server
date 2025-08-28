from datetime import datetime

import requests

from src.common.database.operations import NotionPageOperations
from src.common.embedding.embedder import OpenAITextVectorizer
from src.common.logger.logger import get_logger
from src.config import Config

logger = get_logger(__name__)

NOTION_VERSION = "2022-06-28"


def create_empty_notion_page(
    api_key: str, parent_page_id: str, emoji: str = "🧠"
) -> str:
    now = datetime.now()
    formatted_title = now.strftime("%Y-%m-%d %H:%M:%S")

    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    body = {
        "parent": {"page_id": parent_page_id},
        "icon": {"emoji": emoji},
        "properties": {
            "title": {"title": [{"type": "text", "text": {"content": formatted_title}}]}
        },
    }

    response = requests.post(url, json=body, headers=headers)

    if response.status_code == 200:
        new_page_id = response.json()["id"]
        print(f"New page created: {new_page_id}")
        return new_page_id
    else:
        raise Exception(
            f"Failed to create page. Status: {response.status_code}\n{response.text}"
        )


def create_notion_block_body(content: str) -> dict:
    return {
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": content}}]
                },
            }
        ]
    }


def send_notion_patch_request(block_id: str, api_key: str, block_body: dict):
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION,
    }

    response = requests.patch(url, json=block_body, headers=headers)

    if response.status_code == 200:
        print("Block(s) added successfully!")
    else:
        print(f"Failed to update block. Status code: {response.status_code}")
        print("Response:", response.text)


def process_content_entry(message):
    try:
        # Create a new Notion page
        new_page_id = create_empty_notion_page(
            Config.NOTION_API_KEY, Config.NOTION_DEFAULT_PARENT_ID
        )
        block_body = create_notion_block_body(message["content"])
        send_notion_patch_request(new_page_id, Config.NOTION_API_KEY, block_body)

        # Generate embeddings for the content
        embedded_text = OpenAITextVectorizer().embed_text(message["content"])

        # Prepare metadata for the page
        metadata = {
            "title": message.get("title", "Untitled"),
            "source": message.get("source", "unknown"),
            "notion_page_id": new_page_id,
            "notion_url": f"https://notion.so/{new_page_id.replace('-', '')}",
            "created_at": datetime.utcnow().isoformat(),
            "tags": message.get("tags", []),
            "category": message.get("category", "general"),
        }

        # Store the page data and embedding in the database
        success = NotionPageOperations.insert_page(
            page_id=new_page_id,
            content=message["content"],
            metadata=metadata,
            embedding=embedded_text,
        )

        if success:
            logger.info(
                f"Successfully stored page {new_page_id} with embeddings in database"
            )
        else:
            logger.error(f"Failed to store page {new_page_id} in database")

        return {
            "page_id": new_page_id,
            "embedding": embedded_text,
            "stored_in_db": success,
        }

    except Exception as e:
        logger.error(f"Error processing content entry: {e}")
        raise
