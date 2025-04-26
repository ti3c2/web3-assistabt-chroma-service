# Web3 Assistant ChromaDB Service
Provides api for ChromaDB vector storage.
- Accepts telegram messages
- Processes them
- Vectorizes with OpenAI embeddings and stores in ChromaDB
- Provides a search API for inserted messages

To launch with other services, make sure you added `web3-assistant-network` network to all the compose files and initialized the network with `docker network create web3-assistant-network`

## Endpoints
Swagger: http://localhost:6400/docs

### Models

```python
class Message(BaseModel):
    message_id: int
    text: str
    date: datetime
    username: str

class SearchQuery(BaseModel):
    query: str
    n_results: Optional[int] = 15

class SearchResult(BaseModel):
    document: str
    distance: float
    datetime: datetime
    token_mentions: str
    channel: str
    message_id: str
    chunk_id: str

class SearchResults(BaseModel):
    results: List[SearchResult]
```

### Endpoints
#### POST `/chroma/messages`
Add messages to vector storage.
- **Input**: `List[Message]`
- **Returns**: `Dict[str, str]`
```json
{
    "status": "success",
    "message": "Added N messages"
}
```

#### POST `/chroma/search`
Semantic search across messages.
- **Input**: `SearchQuery`
- **Returns**: `SearchResults`
- **Example Response**:
```json
{
  "results": [
    {
      "document": "Test message\nThis is test message to add to database",
      "distance": 0.19344091,
      "datetime": "2025-04-15T10:57:15+00:00",
      "token_mentions": "",
      "channel": "test",
      "message_id": "0",
      "chunk_id": "test__0__chunk-0"
    }
  ]
}
```

#### DELETE `/chroma/messages`
Delete messages from storage.
- **Input**: `List[str]` (message IDs)
- **Returns**: `Dict[str, str]`
```json
{
    "status": "success",
    "message": "Deleted N messages"
}
```

#### POST `/chroma/fetch`
Fetch and store messages from telegram parser.
- **Input Parameters**:
  - `usernames: List[str]`
  - `limit: int = 50`
  - `offset: int = 0`
- **Returns**: `Dict[str, str]`
```json
{
    "status": "success",
    "message": "Added N messages"
}
```

## CLI Tools
- Test chunking: `python -m src.vector_storage.chunking --help`
- Test parsing from html files: `python -m src.io.parse_tg_html --help`
- Add messages to Vector Storage from html files `python -m src.vector_storage.vector_storage --help`

Put htmls in `./data/html/`

## Code Diagram
```mermaid
classDiagram
    direction TB
    class ProjectSettings {
	    +path_root: Path
	    +path_data: Path
	    +openai_api_key: str
	    +chromadb_host: str
	    +chromadb_port: int
	    +tg_parser_host: str
    }

    class TelegramMessage {
	    +username: str
	    +message_id: str
	    +datetime: datetime
	    +content: str
	    +parsed_content: str
	    +token_mentions: List[str]
    }

    class TextCleaner {
        +remove_emojis(text: str)
        +remove_telegram_links(text: str)
        +remove_urls(text: str)
        +remove_md_emphasis(text: str)
        +remove_md_list_bullets(text: str)
        +replace_md_urls(text: str)
        +remove_hashtags(text: str)
        +remove_cashtags(text: str)
        +remove_whitespace(text: str)
        +parse_html(text: str)
        +cleanup_text(text: str)
    }

    class TokenExtractor {
        +TICKER_PATTERN: str
        +extract_token_single(text: str): List[str]
        +extract_token_pairs(text: str): List[Tuple[str,str]]
        +extract_token_mentions(text: str): List[str]
    }

    class MessageChunker {
	    +chunk_size: int
	    +chunk_overlap: int
	    +split_message(message: TelegramMessage)
	    +split_messages(messages: List[TelegramMessage])
    }

    class BaseEmbeddings {
	    Base class to allow further experiments with embedders
	    +embed_documents(texts: List[str])
	    +embed_query(text: str)
    }

    class OpenAIEmbeddings {
	    +model: str
	    +client: OpenAI
	    +embed_documents(texts: List[str])
	    +embed_query(text: str)
    }

    class ChromaDbWrapper {
	    +collection_name: str
	    +embedding_function: BaseEmbeddings
	    +chunker: MessageChunker
	    +init_client()
	    +add_messages(messages: List[TelegramMessage])
	    +search(query: str)
	    +delete_messages(message_ids: List[str])
    }

    class SearchResult {
	    +document: str
	    +distance: float
	    +datetime: str
	    +token_mentions: str
	    +username: str
	    +message_id: str
    }

    class SearchResults {
	    +results: List[SearchResult]
	    +query: str
	    +from_chromadb()
	    +to_string()
    }

        class SearchQuery {
	    +query: Optional[str]
	    +n_results: Optional[int]
	    +tokens: Optional[List[str]]
    }

    class Message {
	    +message_id: int
	    +text: str
	    +date: datetime
	    +username: str
    }

    class VectorStorAPI {
	    +vector_store: ChromaDbWrapper
	    +POST /chroma/messages(messages: List[Message])
	    +POST /chroma/search(query: SearchQuery)
	    +DELETE /chroma/messages(message_ids: List[str])
	    +POST /chroma/fetch(usernames: List[str], limit: int, offset: int) Fetch messages from external telegram parser API]
	    +POST /chroma/fetchall()
    }

	<<abstract>> BaseEmbeddings

    BaseEmbeddings <|-- OpenAIEmbeddings
    SearchQuery --> VectorStorAPI
    Message <-- VectorStorAPI
    ChromaDbWrapper --> MessageChunker
    ChromaDbWrapper --> BaseEmbeddings
    SearchResults <-- ChromaDbWrapper
    SearchResult <-- SearchResults
    VectorStorAPI --> ChromaDbWrapper
    MessageChunker --> TelegramMessage
    TelegramMessage --> TextCleaner
    TelegramMessage --> TokenExtractor
```
