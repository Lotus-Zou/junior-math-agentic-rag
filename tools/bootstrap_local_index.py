#!/usr/bin/env python3
"""Build an offline lexical Chroma index for local contract and readiness tests.

This is not a replacement for ``ingest.py`` in production. It creates stable
placeholder vectors so the service can exercise BM25/GraphRAG and readiness
without downloading a model. Production deployments must rebuild the index
with the configured embedding model before enabling dense retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from agentic_rag.math_retriever import CHUNK_COLLECTION_NAME
from ingest import load_documents_from_directory, split_math_document
from config import CHROMA_PATH, KNOWLEDGE_DATA_PATH


def _offline_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = [0.0] * dimensions
    tokens = text.lower().split()
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        vector[bucket] += 1.0 / (1 + index)
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="recreate the local collection")
    args = parser.parse_args()

    source_documents = load_documents_from_directory(KNOWLEDGE_DATA_PATH)
    chunks = [
        chunk
        for document in source_documents
        for chunk in split_math_document(document)
    ]
    if not chunks:
        raise SystemExit("no mathematics documents found")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if args.reset:
        try:
            client.delete_collection(CHUNK_COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(CHUNK_COLLECTION_NAME, embedding_function=None)
    existing = set(collection.get(include=[]).get("ids", []))
    ids, documents, metadatas, embeddings = [], [], [], []
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "knowledge"))
        chunk_id = hashlib.sha1(f"{source}:{index}:{chunk.page_content}".encode("utf-8")).hexdigest()
        if chunk_id in existing:
            continue
        ids.append(chunk_id)
        documents.append(chunk.page_content)
        metadatas.append({**chunk.metadata, "index_state": "offline_lexical"})
        embeddings.append(_offline_embedding(chunk.page_content))
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"offline index ready: documents={len(source_documents)} chunks={collection.count()} path={Path(CHROMA_PATH)}")


if __name__ == "__main__":
    main()
