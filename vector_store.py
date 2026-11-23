"""Shared vector store settings.

Ingestion and search must use the same embedding model: vectors from
different models can't be compared. So both get their store from here.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "company_policies"


def get_vector_store() -> Chroma:
    """Open the vector database, creating it if it doesn't exist yet."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(CHROMA_DIR),
    )
