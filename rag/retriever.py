# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.store import VectorStoreConfig, get_collection


@dataclass(frozen=True)
class Chunk:
    """
    A retrieved chunk from the vector store.
    """
    id: str
    text: str
    source: Optional[str]
    chunk_index: Optional[int]
    distance: Optional[float]
    metadata: Dict[str, Any]


def retrieve(
    query: str,
    *,
    top_k: int = 4,
    config: VectorStoreConfig = VectorStoreConfig(),
) -> List[Chunk]:
    """
    Retrieve top_k chunks for a query from Chroma.

    Returns a list of Chunk objects with ids + metadata for debugging and citations.
    """
    if not query or not query.strip():
        return []

    collection = get_collection(config, create_if_missing=True)

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    ids = (results.get("ids") or [[]])[0]

    chunks: List[Chunk] = []
    for doc, meta, dist, _id in zip(docs, metas, dists, ids):
        meta = meta or {}
        chunks.append(
            Chunk(
                id=str(_id),
                text=str(doc),
                source=meta.get("source"),
                chunk_index=meta.get("chunk_index"),
                distance=float(dist) if dist is not None else None,
                metadata=dict(meta),
            )
        )

    return chunks


def format_context(chunks: List[Chunk]) -> str:
    """
    Formats retrieved chunks into a context block with stable numeric citations [1], [2], ...
    (We don't use chunk_index for citations because it's document-dependent and can be sparse.)
    """
    if not chunks:
        return "No relevant context found."

    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(f"[{i}] {ch.text}")
    return "\n\n".join(lines)
