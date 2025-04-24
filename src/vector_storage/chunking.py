import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config.settings import settings
from ..io.models import TelegramMessage

logger = logging.getLogger(__name__)


def remove_short(chunks: List[str]) -> List[str]:
    return [c for c in chunks if len(c.split()) > 5]


def remove_newlines(chunks: List[str]) -> List[str]:
    return [c.replace("\n", " ") for c in chunks]


def get_chunk_transforms(
    funcs: List[Callable[[List[str]], List[str]]],
) -> Callable[[List[str]], List[str]]:
    def transform_chunks(chunks: List[str]) -> List[str]:
        for func in funcs:
            chunks = func(chunks)
        return chunks

    return transform_chunks


transform_chunks = get_chunk_transforms([remove_short, remove_newlines])


@dataclass
class MessageChunker:
    chunk_size: int = 150
    chunk_overlap: int = 0
    force_paragraph_split: bool = False
    length_function: Callable = len

    def __post_init__(self):
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            # length_function=self.length_function,
        )
        self.transform_chunks: Callable[[List[str]], List[str]] = transform_chunks

    def split_text(self, text: str) -> List[str]:
        chunks = text.split("\n\n") if self.force_paragraph_split else [text]
        chunks = [c for para in chunks for c in self.splitter.split_text(para)]
        chunks = self.transform_chunks(chunks)
        return chunks

    def split_message(
        self, message: TelegramMessage
    ) -> List[Document]:  # NOTE: Tightly bound with search results in vector_store.py
        chunks = self.split_text(message.parsed_content)
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{message.username}__{message.message_id}__chunk-{i}"
            doc = Document(
                page_content=chunk,
                metadata=dict(
                    chunk_id=chunk_id,
                    username=message.username,
                    message_id=message.message_id,
                    datetime=message.datetime,
                    # token_mentions=",".join(message.token_mentions),
                    content=message.content,
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
    import argparse
    import datetime as dt

    parser = argparse.ArgumentParser(description="Example chunking script")
    parser.add_argument("-t", "--text", type=str, help="input string")
    parser.add_argument("-f", "--file", type=str, default=None, help="input file")
    parser.add_argument("-s", "--chunk_size", type=int, help="chunk size")
    parser.add_argument("-o", "--overlap", type=int, help="overlap size")
    parser.add_argument(
        "-fp",
        "--force-paragraph",
        action="store_true",
        help="force paragraph splitting",
    )
    args = parser.parse_args()

    text = args.text
    if (fname := args.file) is not None:
        text = settings.find_file(fname).read_text()

    # Create a sample message with a long text
    message = TelegramMessage(
        username="crypto_news",
        message_id="12345",
        datetime=dt.datetime(2023, 1, 1, 12, 0),
        content=text,
    )
    print(f"Original message:\n{message.content}\n====================")

    # Initialize chunker with small chunk size for demonstration
    chunker = MessageChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        force_paragraph_split=args.force_paragraph,
    )

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
        print(f"Channel: {doc.metadata['username']}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    example_chunking()
