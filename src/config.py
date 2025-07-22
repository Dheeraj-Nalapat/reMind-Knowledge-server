import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    RABBITMQ_URL = os.getenv("RABBITMQ_URL")