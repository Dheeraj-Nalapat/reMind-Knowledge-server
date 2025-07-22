import requests
from datetime import datetime
from src.config import Config
from src.common.embedding.embedder import OpenAITextVectorizer

NOTION_VERSION = "2022-06-28"


def create_empty_notion_page(api_key: str, parent_page_id: str, emoji: str = "🧠") -> str:
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
            "title": {
                "title": [
                    {"type": "text", "text": {"content": formatted_title}}
                ]
            }
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
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content
                            }
                        }
                    ]
                }
            }
        ]
    }


def send_notion_patch_request(block_id: str, api_key: str, block_body: dict):
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_VERSION
    }

    response = requests.patch(url, json=block_body, headers=headers)

    if response.status_code == 200:
        print("Block(s) added successfully!")
    else:
        print(f"Failed to update block. Status code: {response.status_code}")
        print("Response:", response.text)

def process_content_entry(message):
    new_page_id = create_empty_notion_page(Config.NOTION_API_KEY, Config.PARENT_PAGE_ID)
    block_body = create_notion_block_body(message["content"])
    send_notion_patch_request(new_page_id, Config.NOTION_API_KEY, block_body)

    embedded_text = OpenAITextVectorizer().embed_text(message["content"])

    
