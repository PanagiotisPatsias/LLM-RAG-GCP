# rag/store.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions


@dataclass(frozen=True)
class VectorStoreConfig:
    """
    Central configuration for the vector store.

    Keep ALL Chroma/OpenAI embedding settings here so:
    - Streamlit UI has no DB setup logic
    - ingestion/retrieval/eval all share the same collection
    """
    persist_path: str = "./chroma_db"
    collection_name: str = "rag-docs"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key_env: str = "OPENAI_API_KEY"


def _get_openai_api_key(env_name: str = "OPENAI_API_KEY") -> str:
    key = os.getenv(env_name)
    if not key:
        raise RuntimeError(
            f"Missing {env_name}. Set it in your environment or .env file."
        )
    return key


def get_chroma_client(config: VectorStoreConfig = VectorStoreConfig()) -> chromadb.PersistentClient:
    """
    Returns a persistent Chroma client.
    """
    return chromadb.PersistentClient(path=config.persist_path)


def get_embedding_function(config: VectorStoreConfig = VectorStoreConfig()):
    """
    Returns Chroma's built-in OpenAI embedding function.
    """
    api_key = _get_openai_api_key(config.openai_api_key_env)
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=config.embedding_model,
    )


def get_collection(
    config: VectorStoreConfig = VectorStoreConfig(),
    *,
    create_if_missing: bool = True,
) -> Collection:
    """
    Returns the Chroma collection used by the RAG system.

    - `create_if_missing=True` is convenient for local dev and first-time runs.
    - In CI, you might set it to False if you want to enforce a pre-built index.
    """
    client = get_chroma_client(config)
    ef = get_embedding_function(config)

    if create_if_missing:
        return client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=ef,
        )

    # If you want strict behavior (fail if missing)
    return client.get_collection(
        name=config.collection_name,
        embedding_function=ef,
    )


def collection_count(config: VectorStoreConfig = VectorStoreConfig()) -> int:
    """
    Convenience helper to check whether the store has data.
    """
    col = get_collection(config, create_if_missing=True)
    return col.count()


def reset_collection(config: VectorStoreConfig = VectorStoreConfig()) -> Collection:
    """
    Deletes and recreates the collection. Useful for deterministic rebuilds.
    """
    client = get_chroma_client(config)
    try:
        client.delete_collection(name=config.collection_name)
    except Exception:
        # Collection might not exist yet
        pass

    ef = get_embedding_function(config)
    return client.get_or_create_collection(
        name=config.collection_name,
        embedding_function=ef,
    )
