import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import chromadb
from chromadb.api import AsyncClientAPI
from chromadb.api.models.AsyncCollection import AsyncCollection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from pydantic import BaseModel, ConfigDict

from ..config.settings import settings
from ..io.models import TelegramMessage
from .chunking import MessageChunker
from .embeddings import BaseEmbeddings, OpenAIEmbeddings

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):  # NOTE: Tightly bound with document from chunking.py
    """Wrapper for a single search result from ChromaDB."""

    document: str
    distance: float
    datetime: str
    # token_mentions: str # Deprecated. Now this is implemented by full text search
    username: str
    message_id: str
    chunk_id: str
    content: str

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_chromadb(cls, doc: str, dist: float, meta: Dict) -> "SearchResult":
        """Creates SearchResult from raw ChromaDB output."""
        return cls(
            document=doc,
            distance=dist,
            **meta,
        )

    def to_string(self) -> str:
        out = ""
        for key, value in self.model_dump().items():
            key = key.capitalize()
            if key == "Document":
                wrapped_text = textwrap.fill(value, width=80)
                out += f"{key}:\n{wrapped_text}\n\n"
                out += "Metadata:\n"
            else:
                out += " " * 4 + f"{key}: {value}\n"
        return out


class SearchResults(BaseModel):
    """Container for multiple search results from ChromaDB."""

    query: Optional[str] = None
    results: List[SearchResult]

    @classmethod
    def from_chromadb(
        cls, chroma_results: Dict[str, Any], query: Optional[str] = None
    ) -> "SearchResults":
        results = []
        for dist, doc, meta in zip(
            chroma_results["distances"][0],
            chroma_results["documents"][0],
            chroma_results["metadatas"][0],
        ):
            result = SearchResult.from_chromadb(doc=doc, dist=dist, meta=meta)
            results.append(result)
        return cls(results=results, query=query)

    def to_string(self) -> str:
        out = f"Query: {self.query}\n\n"
        for result in self.results:
            out += result.to_string()
            out += "\n======\n"
        return out

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> SearchResult:
        return self.results[idx]

    def __iter__(self):
        return iter(self.results)


@dataclass
class ChromaDbWrapper:
    collection_name: str = "telegram_messages"
    embedding_function: BaseEmbeddings = OpenAIEmbeddings()
    host: str = settings.chromadb_host
    port: int = settings.chromadb_port
    chroma_metadata: Dict[str, Any] = field(
        default_factory=lambda: {
            "hnsw:space": "cosine",
            "hnsw:search_ef": 100,
        }
    )
    client: AsyncClientAPI = None

    chunker: MessageChunker = field(default_factory=MessageChunker)

    async def init_client(self) -> None:
        self.client = await chromadb.AsyncHttpClient(
            host=self.host,
            port=self.port,
            settings=Settings(anonymized_telemetry=False),
        )

    async def get_client(self) -> AsyncClientAPI:
        if self.client is None:
            await self.init_client()
        return self.client

    async def get_collection(self, collection_name: Optional[str]) -> AsyncCollection:
        collection_name = collection_name or self.collection_name
        client = await self.get_client()
        collection = await client.get_or_create_collection(
            name=collection_name,
            # embedding_function=self._embedding_function,
            metadata=self.chroma_metadata,
        )
        return collection

    async def add_messages(self, messages: List[TelegramMessage]) -> None:
        collection = await self.get_collection(self.collection_name)
        collection_entries = await collection.get(include=[])

        # FIXME: think of how to skip chunking for messages that are already in the collection
        all_documents = self.chunker.split_messages(messages)
        documents = []
        logger.info(
            f"Adding {len(all_documents)} new documents out of {len(messages)} messages to vector store..."
        )
        for doc in self.chunker.split_messages(messages):
            if doc.metadata["chunk_id"] in collection_entries["ids"]:
                logger.warning(f"Skipping chunk {doc.metadata['chunk_id']}")
                continue
            documents.append(doc)
        if not documents:
            logger.warning("No new documents to add")
            return

        ids = [doc.metadata["chunk_id"] for doc in documents]
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = await self.embedding_function.embed_documents(texts)
        await collection.add(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )
        logger.info(
            f"Added {len(documents)} new documents out of required {len(all_documents)} to vector store"
        )

    async def search(
        self,
        query: Optional[str],
        n_results: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
        where_document_filter: Optional[Dict[str, Any]] = None,
        full_text_items: Optional[List[str]] = None,
    ) -> SearchResults:
        """
        Search the vector store for messages based on parameters.
        If query is not set, search for all messages by filters
        If tokens are specified, initiate full-text search for them
        """
        if not query:
            logger.info("No query provided, searching for all messages")
            results = await self._search_all(
                n_results=n_results,
                where_filter=where_filter,
                where_document_filter=where_document_filter,
                full_text_items=full_text_items,
            )
        logger.info("Query provided, searching for relevant messages")
        results = await self._search_semantic(
            query, n_results, where_filter, where_document_filter, full_text_items
        )
        return SearchResults.from_chromadb(results, query=query)

    async def _search_semantic(
        self,
        query: str,
        n_results: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
        where_document_filter: Optional[Dict[str, Any]] = None,
        full_text_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search the vector store for messages relevant to the query"""
        collection = await self.get_collection(self.collection_name)
        query_embedding = await self.embedding_function.embed_query(query)
        if full_text_items:
            where_document_filter = self.create_full_text_items_filter(
                full_text_items, where_document_filter
            )
        results = await collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            where_document=where_document_filter,  # pyright: ignore
        )
        return results

    async def _search_all(
        self,
        n_results: int = 20,
        where_filter: Optional[Dict[str, Any]] = None,
        where_document_filter: Optional[Dict[str, Any]] = None,
        full_text_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        collection = await self.get_collection(self.collection_name)
        if full_text_items:
            where_document_filter = self.create_full_text_items_filter(
                full_text_items, where_document_filter
            )
        results = await collection.get(
            limit=n_results,
            where=where_filter,
            where_document=where_document_filter,  # pyright: ignore
        )
        logger.debug("Search results: {}".format(results))
        out = dict(
            ids=[results["ids"]],
            documents=[results["documents"]],
            metadatas=[results["metadatas"]],
            distances=[[0.0] * len(results["ids"])],
        )
        return out

    def create_full_text_items_filter(
        self,
        full_text_items: List[str],
        where_document_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not full_text_items and where_document_filter is None:
            return None
        contains_items = [{"$contains": item} for item in full_text_items]
        if not full_text_items:
            return where_document_filter
        items_filter = (
            {"$or": contains_items} if len(contains_items) > 1 else contains_items[0]
        )
        if not where_document_filter:
            return items_filter
        logger.warning(
            "Both full text items and where_document_filter provided: {} {}".format(
                items_filter, where_document_filter
            )
        )
        if where_doc_items := where_document_filter.get("$or"):
            return {"$or": contains_items + where_doc_items}
        out_filter = {"$or": [where_document_filter] + contains_items}
        return out_filter

    async def delete_messages(self, message_ids: List[str]) -> None:
        collection = await self.get_collection(self.collection_name)
        await collection.delete(ids=message_ids)


async def main():
    import argparse

    from ..io.parse_tg_html import parse_tg_files

    parser = argparse.ArgumentParser(description="Vector Store CLI")
    parser.add_argument("-f", "--files", type=str, nargs="+", default=[])
    parser.add_argument(
        "-c", "--collection_name", type=str, default="telegram_messages"
    )
    args = parser.parse_args()

    files = args.files or list(settings.path_data_html.glob("*.html"))
    messages = parse_tg_files(files)
    vector_store = ChromaDbWrapper(
        collection_name=args.collection_name,
    )
    await vector_store.add_messages(messages)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
