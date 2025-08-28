import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")
    RABBITMQ_URL = os.getenv("RABBITMQ_URL")
    PG_URL = os.getenv("PG_URL")
    NOTION_DEFAULT_PARENT_ID = os.getenv("NOTION_DEFAULT_PARENT_ID")
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY")
