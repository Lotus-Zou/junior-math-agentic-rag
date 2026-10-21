# -*- coding: utf-8 -*-
"""Inspect three-stage mathematics retrieval from the command line."""

import argparse
import sys

from agentic_rag.math_retriever import math_retriever
from agentic_rag.math_taxonomy import classify_math_text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="查询初中数学知识库")
    parser.add_argument("query", nargs="?", help="数学问题；省略后进入交互模式")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()
    query = args.query or input("数学问题: ").strip()
    if not query:
        return
    classification = classify_math_text(query)
    documents, trace = math_retriever.search(query, classification.chapter, classification.knowledge_points, top_k=args.top_k)
    print(f"分类: {classification.chapter} / {'、'.join(classification.knowledge_points)}")
    print(f"检索轨迹: {trace}")
    for index, document in enumerate(documents, start=1):
        print(f"\n[{index}] {document.metadata}")
        print(document.page_content[:800])


if __name__ == "__main__":
    main()