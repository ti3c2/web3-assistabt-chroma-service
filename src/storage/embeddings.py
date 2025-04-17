from abc import ABC, abstractmethod
from typing import List

import openai

from ..config.settings import settings


class BaseEmbeddings(ABC):
    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        self.client = openai.AsyncClient(api_key=settings.openai_api_key)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]

    async def embed_query(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding
