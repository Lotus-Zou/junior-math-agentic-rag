# -*- coding: utf-8 -*-
"""View the mathematics ChromaDB collections and metadata."""

import argparse
import sys

import chromadb

from config import CHROMA_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COLLECTIONS = {"summaries": "math_summaries", "chunks": "math_chunks", "memory": "math_learning_memory"}


def main():
    parser = argparse.ArgumentParser(description="查看初中数学向量库")
    parser.add_argument("-c", "--collection", choices=COLLECTIONS, default="chunks")
    parser.add_argument("-l", "--limit", type=int, default=5)
    args = parser.parse_args()
    collection_name = COLLECTIONS[args.collection]
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        print(f"集合 {collection_name} 不存在，请先运行 python ingest.py。")
        return
    count = collection.count()
    print(f"{collection_name}: {count} 条")
    if count == 0:
        return
    results = collection.get(limit=min(args.limit, count), include=["documents", "metadatas"])
    for index, (item_id, metadata, document) in enumerate(zip(results["ids"], results["metadatas"], results["documents"]), start=1):
        print(f"\n[{index}] ID={item_id}\n元数据={metadata}\n{document[:800]}")


if __name__ == "__main__":
    main()