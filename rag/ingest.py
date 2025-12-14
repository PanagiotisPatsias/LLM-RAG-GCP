# rag/ingest.py
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Union

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.store import VectorStoreConfig, get_collection, reset_collection


def extract_text_from_pdf_path(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n".join(pages)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "Article ", ".", " ", ""],
    )
    return splitter.split_text(text)


def ingest_pdf_path(
    pdf_path: Path,
    *,
    config: VectorStoreConfig = VectorStoreConfig(),
    reset: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Ingest a single PDF from disk into the vector store.
    Returns number of chunks added.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    collection = reset_collection(config) if reset else get_collection(config, create_if_missing=True)

    text = extract_text_from_pdf_path(pdf_path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        return 0

    ids = [f"{pdf_path.name}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks)

#Without this function the RAG cannot see the uploaded PDFs in Streamlit because they are provided as bytes and wants pdfs or director with pdfs.
def ingest_pdf_bytes(
    pdf_bytes: bytes,
    *,
    filename: str = "uploaded.pdf",
    config: VectorStoreConfig = VectorStoreConfig(),
    reset: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Ingest a PDF provided as bytes (Streamlit uploader).
    We write to a temp file because PdfReader expects a file path reliably.
    """
    if not pdf_bytes:
        return 0

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        return ingest_pdf_path(
            Path(tmp.name),
            config=config,
            reset=reset,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def ingest_pdf_dir(
    pdf_dir: Path,
    *,
    config: VectorStoreConfig = VectorStoreConfig(),
    reset: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """
    Ingest all PDFs in a directory. Returns total chunks added.
    If reset=True, it resets once at the start.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)

    if reset:
        reset_collection(config)

    total = 0
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        n = ingest_pdf_path(
            pdf,
            config=config,
            reset=False,  # already reset above (if needed)
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        total += n
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the Chroma vector store.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingest")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args()

    total = ingest_pdf_dir(
        Path(args.pdf_dir),
        reset=args.reset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"✅ Ingested total chunks: {total}")


if __name__ == "__main__":
    main()
