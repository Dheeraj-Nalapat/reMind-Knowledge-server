import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://localhost:5672")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NOTION_API_KEY = os.getenv("NOTION_API_KEY")