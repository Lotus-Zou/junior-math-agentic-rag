# -*- coding: utf-8 -*-
"""Compatibility wrappers for callers of the former generic retrievers."""

from agentic_rag.math_retriever import math_retriever
from agentic_rag.math_taxonomy import classify_math_text


def hierarchical_retriever(query: str, n_docs: int = 3, n_chunks: int = 5):
    classification = classify_math_text(query)
    documents, _ = math_retriever.search(query, classification.chapter, classification.knowledge_points, top_k=n_chunks)
    return documents


def direct_chunk_retriever(query: str, n_chunks: int = 5):
    classification = classify_math_text(query)
    documents, _ = math_retriever.search(query, classification.chapter, classification.knowledge_points, top_k=n_chunks)
    return documents