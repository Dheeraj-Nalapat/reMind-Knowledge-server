from typing import List

from langchain_openai import OpenAIEmbeddings

from src.config import Config


class OpenAITextVectorizer:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embeddings client.
        If api_key is not provided, it will be picked up from OPENAI_API_KEY env var.
        """
        self.client = OpenAIEmbeddings(model=model, api_key=Config.OPENAI_API_KEY)

    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single piece of text.
        Returns a dense vector (list of floats).
        """
        return self.client.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text strings.
        Returns a list of vectors.
        """
        return self.client.embed_documents(texts)
