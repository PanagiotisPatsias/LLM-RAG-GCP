import argparse
from pathlib import Path
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
    return "\n".join(pages)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        raise FileNotFoundError(pdf_dir)

    chroma_client = chromadb.PersistentClient(path="chroma_db")

    if args.reset:
        try:
            chroma_client.delete_collection("rag-docs")
            print("🧹 Deleted existing collection")
        except Exception:
            pass

    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-3-small",
    )

    collection = chroma_client.get_or_create_collection(
        name="rag-docs",
        embedding_function=embedding_fn,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "Article ", ".", " ", ""],
    )

    for pdf in pdf_dir.glob("*.pdf"):
        print(f"📄 Ingesting {pdf.name}")
        text = extract_text(pdf)
        chunks = splitter.split_text(text)

        ids = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            ids.append(f"{pdf.name}-{i}")
            metadatas.append({
                "source": pdf.name,
                "chunk_index": i,
            })

        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=metadatas,
        )

        print(f"✅ Added {len(chunks)} chunks from {pdf.name}")

    print(f"\n🎉 Ingestion complete. Total chunks: {collection.count()}")

if __name__ == "__main__":
    main()
