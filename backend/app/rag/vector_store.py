
import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy.engine import make_url
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
VECTOR_TABLE = "researchos_chunks"


def get_database_config() -> dict:
    """Read PostgreSQL connection details from the existing DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from your environment.")

    url = make_url(database_url)

    if not url.host or not url.database or not url.username:
        raise RuntimeError("DATABASE_URL must specify host, database, and username.")

    return {
        "host": url.host,
        "port": url.port or 5432,
        "database": url.database,
        "user": url.username,
        "password": url.password or "",
    }


@lru_cache(maxsize=1)
def get_embed_model() -> OpenAIEmbedding:
    """Create the embedding model once and reuse it."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing from your environment.")

    return OpenAIEmbedding(
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PGVectorStore:
    """Create and reuse the PostgreSQL vector store."""
    config = get_database_config()

    return PGVectorStore.from_params(
        **config,
        table_name=VECTOR_TABLE,
        embed_dim=EMBEDDING_DIMENSIONS,
    )


def configure_rag() -> None:
    """Configure LlamaIndex to use our embedding model."""
    Settings.embed_model = get_embed_model()
