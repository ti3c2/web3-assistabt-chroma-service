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

class SearchResults:
    documents: List[str]
    metadatas: List[Dict]
    distances: List[float]
    ids: List[str]
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
```python
{
    "documents": ["message text 1", "message text 2"],
    "metadatas": [
        {"username": "user1", "datetime": "2023-01-01T00:00:00"},
        {"username": "user2", "datetime": "2023-01-02T00:00:00"}
    ],
    "distances": [0.123, 0.456],
    "ids": ["msg_id_1", "msg_id_2"]
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
