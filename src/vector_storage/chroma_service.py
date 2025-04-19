import datetime as dt
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config.settings import settings
from ..io.models import TelegramMessage
from .vector_storage import ChromaDbWrapper, SearchResults

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vector Storage Service",
    description="API for vector storage operations",
    version="1.0.0",
)

vector_store = ChromaDbWrapper()


class Message(BaseModel):
    message_id: int
    text: str
    date: dt.datetime
    username: str


class SearchQuery(BaseModel):
    query: Optional[str] = None
    n_results: int = 15
    tokens: Optional[List[str]] = None


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Chroma service...")
    await vector_store.init_client()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await vector_store.init_client()
#     yield


@app.post("/chroma/messages")
async def add_messages(messages: List[Message]) -> Dict[str, str]:
    """Add messages to the vector store."""
    try:
        telegram_messages = [
            TelegramMessage(
                message_id=str(msg.message_id),
                content=msg.text,
                datetime=msg.date,
                username=msg.username,
            )
            for msg in messages
        ]
        await vector_store.add_messages(telegram_messages)
        return {"status": "success", "message": f"Added {len(messages)} messages"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chroma/search")
async def search_messages(query: SearchQuery) -> SearchResults:
    """
    Search messages in the vector store.
    - If query is specified, do semantic search. Otherwise, search for all messages using filters.
    - If tokens are specified, include them in full-text search.
    - If only tokens are specified, do full-text search for them.
    - If neither query nor tokens are specified, return all messages.
    """
    try:
        return await vector_store.search(
            query.query, query.n_results, full_text_items=query.tokens
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error on /chroma/search: {}".format(e)
        )


@app.delete("/chroma/messages")
async def delete_messages(message_ids: List[str]):
    """Delete messages from the vector store by ids"""
    try:
        await vector_store.delete_messages(message_ids)
        return {"status": "success", "message": f"Deleted {len(message_ids)} messages"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chroma/fetch")
async def fetch_messages(
    usernames: Optional[List[str]] = None, limit: int = 500, offset: int = 0
):
    """Fetch messages from the tg parser and add them to the vector store"""
    try:
        endpoint = settings.tg_parser_posts_endpoint
        params = {"usernames": usernames, "limit": limit, "offset": offset}
        logger.info(f"Fetching messages from {endpoint}")
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as response:
                if response.status == 200:
                    messages = await response.json()
                    messages = [Message(**m) for m in messages]
                    await add_messages(messages)
                    return {
                        "status": "success",
                        "message": f"Added {len(messages)} messages",
                    }
                else:
                    raise HTTPException(
                        status_code=response.status, detail="Failed to fetch messages"
                    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(
        "src.storage.chroma_service:app",
        host="0.0.0.0",
        port=settings.chromadb_api_port,
        reload=True,
    )
