from typing import Callable, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..io.models import TelegramMessage


class MessageChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 2,
        length_function: Callable = len,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
        )

    def split_message(self, message: TelegramMessage) -> List[Document]:
        chunks = self.splitter.split_text(message.parsed_content)
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{message.channel}__{message.message_id}__chunk-{i}"
            doc = Document(
                page_content=chunk,
                metadata=dict(
                    chunk_id=chunk_id,
                    channel=message.channel,
                    message_id=message.message_id,
                    datetime=message.datetime.isoformat() if message.datetime else "",
                    token_mentions=",".join(message.token_mentions),
                ),
            )
            documents.append(doc)

        return documents

    def split_messages(self, messages: List[TelegramMessage]) -> List[Document]:
        documents = []
        for message in messages:
            documents.extend(self.split_message(message))
        return documents


def example_chunking() -> None:
    """Shows how MessageChunker works with a sample message"""
    import datetime as dt

    # Create a sample message with a long text
    message = TelegramMessage(
        channel="crypto_news",
        message_id="12345",
        datetime=dt.datetime(2023, 1, 1, 12, 0),
        content="""\
🚀 Bitcoin Analysis and Market Update 🚀

Bitcoin has shown remarkable strength in recent days, pushing past key resistance levels.
Technical indicators suggest a bullish momentum:

1. RSI levels indicate oversold conditions
2. MACD shows positive crossover
3. Volume profile remains strong

Key support levels to watch:
- $45,000
- $43,500
- $42,000

Resistance levels:
- $48,000
- $50,000
- $52,000

Notable mentions: $BTC $ETH are showing correlation in price movement.

Stay tuned for more updates! 📊""",
    )
    print(f"Original message:\n{message.content}\n====================")

    # Initialize chunker with small chunk size for demonstration
    chunker = MessageChunker(chunk_size=200, chunk_overlap=100)

    # Split the message into chunks
    documents = chunker.split_message(message)

    # Print results
    print(f"Number of chunks: {len(documents)}\n")

    # Show all chunks
    for i, doc in enumerate(documents):
        print(f"=== Chunk {i + 1} ===")
        print(f"ID: {doc.metadata['message_id']}")
        print(f"Content length: {len(doc.page_content)}")
        print("Content:")
        print(doc.page_content)
        print("\nMetadata:")
        print(f"Original message ID: {doc.metadata['message_id']}")
        print(f"Channel: {doc.metadata['channel']}")
        print(f"Token mentions: {doc.metadata['token_mentions']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    example_chunking()
