import datetime as dt
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

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
    text: str = ""
    date: str
    username: str

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, text: Optional[str] = None) -> str:
        if text is None:
            logger.warning("No text provided for message")
            return ""
        return text



class SearchQuery(BaseModel):
    query: Optional[str] = None
    n_results: int = 5
    tokens: Optional[List[str]] = None
    return_unique: bool = Field(
        default=True,
        description="Return unique messages. Do not change this if you are unsure.",
    )


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
async def search_messages(
    query: SearchQuery,
) -> SearchResults:
    """
    Search messages in the vector store.
    - If query is specified, do semantic search. Otherwise, search for all messages using filters.
    - If tokens are specified, include them in full-text search.
    - If only tokens are specified, do full-text search for them.
    - If neither query nor tokens are specified, return all messages.
    """
    try:
        return await vector_store.search(
            query.query,
            query.n_results,
            full_text_items=query.tokens,
            return_unique=query.return_unique,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error on /chroma/search: {}".format(e)
        )


@app.delete("/chroma/messages")
async def delete_messages(
    message_ids: Optional[List[str]] = None, usernames: Optional[List[str]] = None
):
    """
    Delete messages from the vector store by ids.
    - If message_ids are specified, delete messages with those ids.
    - If usernames are specified, delete ALL messages from those users.
    """
    try:
        await vector_store.delete_messages(message_ids, usernames)
        return {
            "status": "success",
            "message": f"Deleted messages for ids {message_ids} and usernames {usernames}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chroma/fetch")
async def fetch_messages(
    usernames: Optional[List[str]] = None, limit: int = 500, offset: int = 0
):
    """Fetch messages from the tg parser and add them to the vector store"""
    try:
        endpoint = settings.tg_parser_posts_endpoint
        params = {"limit": limit, "offset": offset}
        if usernames is not None:
            params["usernames"] = usernames
        logger.info(f"Fetching messages from {endpoint}")
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint, params=params) as response:
                if response.status == 200:
                    messages = await response.json()
                    logger.info("Fetched messages: %s", messages[:5])
                    messages_parsed = []
                    for m in messages:
                        try:
                            messages_parsed.append(Message(**m))
                        except Exception as e:
                            logger.error(f"Failed to parse message: {m}, error: {e}")
                    await add_messages(messages_parsed)
                    return {
                        "status": "success",
                        "message": f"Added {len(messages_parsed)} messages",
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
