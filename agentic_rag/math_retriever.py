# -*- coding: utf-8 -*-
"""Dense + BM25 + GraphRAG retrieval with reciprocal-rank fusion."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import chromadb
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from agentic_rag.knowledge_graph import math_knowledge_graph
from agentic_rag.math_taxonomy import tokenize_math
from config import CHROMA_PATH, RETRIEVAL_CANDIDATES, RETRIEVAL_TOP_K

CHUNK_COLLECTION_NAME = "math_chunks"


def fuse_rankings(*rankings: Sequence[str], k: int = 60) -> Dict[str, float]:
    """Fuse any number of ranked result lists using RRF."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class MathKnowledgeRetriever:
    """Category filtering, multi-query dense/BM25 retrieval, and GraphRAG expansion."""

    def __init__(self, persist_path: str = CHROMA_PATH):
        self.persist_path = persist_path
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            from agentic_rag.chains import get_embedding_function

            client = chromadb.PersistentClient(path=self.persist_path)
            self._collection = client.get_or_create_collection(
                CHUNK_COLLECTION_NAME,
                embedding_function=get_embedding_function(),
            )
        return self._collection

    @staticmethod
    def _where(chapter: str):
        return {"chapter": chapter} if chapter and chapter != "综合" else None

    def _category_pool(self, chapter: str) -> dict[str, tuple[str, dict]]:
        where = self._where(chapter)
        get_args = {"include": ["documents", "metadatas"]}
        if where:
            get_args["where"] = where
        pool = self.collection.get(**get_args)
        return {
            doc_id: (text, metadata or {})
            for doc_id, text, metadata in zip(
                pool.get("ids", []), pool.get("documents", []), pool.get("metadatas", [])
            )
        }

    @staticmethod
    def _channel_documents(
        ordered_ids: Sequence[str],
        document_by_id: dict[str, tuple[str, dict]],
        channel: str,
        scores: dict[str, float] | None = None,
    ) -> List[Document]:
        documents = []
        for rank, doc_id in enumerate(ordered_ids, start=1):
            text, metadata = document_by_id[doc_id]
            enriched = dict(metadata)
            enriched.update({
                "chunk_id": doc_id,
                "retrieval_channel": channel,
                "channel_rank": rank,
                "channel_score": round(float((scores or {}).get(doc_id, 0.0)), 6),
            })
            documents.append(Document(page_content=text, metadata=enriched))
        return documents

    def retrieve_dense_channel(self, queries: Sequence[str], chapter: str, candidate_k: int) -> Tuple[List[Document], List[dict]]:
        document_by_id = self._category_pool(chapter)
        if not document_by_id:
            return [], [{"stage": "dense_retrieval", "candidates": 0}]
        where = self._where(chapter)
        ordered, scores = [], {}
        for query in dict.fromkeys(item.strip() for item in queries if item.strip()):
            args = {
                "query_texts": [query],
                "n_results": min(candidate_k, len(document_by_id)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                args["where"] = where
            result = self.collection.query(**args)
            for doc_id, text, metadata, distance in zip(
                (result.get("ids") or [[]])[0],
                (result.get("documents") or [[]])[0],
                (result.get("metadatas") or [[]])[0],
                (result.get("distances") or [[]])[0],
            ):
                document_by_id.setdefault(doc_id, (text, metadata or {}))
                if doc_id not in ordered:
                    ordered.append(doc_id)
                scores[doc_id] = max(scores.get(doc_id, 0.0), 1.0 / (1.0 + float(distance)))
        ordered.sort(key=lambda item: scores.get(item, 0.0), reverse=True)
        documents = self._channel_documents(ordered[:candidate_k], document_by_id, "dense", scores)
        return documents, [{"stage": "dense_retrieval", "candidates": len(documents)}]

    def retrieve_bm25_channel(self, queries: Sequence[str], chapter: str, candidate_k: int) -> Tuple[List[Document], List[dict]]:
        document_by_id = self._category_pool(chapter)
        corpus_ids = list(document_by_id)
        if not corpus_ids:
            return [], [{"stage": "bm25_retrieval", "candidates": 0}]
        corpus = [tokenize_math(document_by_id[doc_id][0]) or ["<empty>"] for doc_id in corpus_ids]
        bm25, scores = BM25Okapi(corpus), {}
        for query in dict.fromkeys(item.strip() for item in queries if item.strip()):
            values = bm25.get_scores(tokenize_math(query))
            for index, value in enumerate(values):
                if value > 0:
                    scores[corpus_ids[index]] = max(scores.get(corpus_ids[index], 0.0), float(value))
        ordered = sorted(scores, key=scores.get, reverse=True)[:candidate_k]
        documents = self._channel_documents(ordered, document_by_id, "bm25", scores)
        return documents, [{"stage": "bm25_retrieval", "candidates": len(documents)}]

    def retrieve_graph_channel(self, knowledge_points: Sequence[str], chapter: str, candidate_k: int) -> Tuple[List[Document], List[dict]]:
        document_by_id = self._category_pool(chapter)
        expanded = math_knowledge_graph.expand(knowledge_points, depth=1)
        scores = {
            doc_id: float(sum(
                point in text or point in str(metadata.get("prerequisites", ""))
                for point in expanded
            ))
            for doc_id, (text, metadata) in document_by_id.items()
        }
        ordered = [doc_id for doc_id in sorted(scores, key=scores.get, reverse=True) if scores[doc_id] > 0][:candidate_k]
        documents = self._channel_documents(ordered, document_by_id, "graph", scores)
        return documents, [{"stage": "graph_rag", "expanded_points": expanded, "candidates": len(documents)}]

    def retrieve_candidates(
        self,
        queries: Sequence[str],
        chapter: str,
        knowledge_points: Sequence[str],
        candidate_k: int = RETRIEVAL_CANDIDATES,
        strategy: str = "hybrid_graph",
    ) -> Tuple[List[Document], List[dict]]:
        """Return RRF candidates; LLM-Rerank is deliberately a separate graph node."""
        if strategy not in {"dense", "hybrid", "hybrid_graph"}:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
        total = self.collection.count()
        if total == 0:
            return [], [{"stage": "category_recall", "chapter": chapter, "candidates": 0}]
        normalized_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not normalized_queries:
            return [], [{"stage": "query_decomposition", "queries": 0}]

        where = self._where(chapter)
        get_args = {"include": ["documents", "metadatas"]}
        if where:
            get_args["where"] = where
        category_pool = self.collection.get(**get_args)
        document_by_id = {
            doc_id: (text, metadata or {})
            for doc_id, text, metadata in zip(
                category_pool.get("ids", []),
                category_pool.get("documents", []),
                category_pool.get("metadatas", []),
            )
        }
        if not document_by_id:
            return [], [{"stage": "category_recall", "chapter": chapter, "candidates": 0}]

        dense_rankings: List[List[str]] = []
        for query in normalized_queries:
            query_args = {
                "query_texts": [query],
                "n_results": min(candidate_k, len(document_by_id)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                query_args["where"] = where
            dense = self.collection.query(**query_args)
            dense_ids = (dense.get("ids") or [[]])[0]
            dense_rankings.append(dense_ids)
            for doc_id, text, metadata in zip(
                dense_ids,
                (dense.get("documents") or [[]])[0],
                (dense.get("metadatas") or [[]])[0],
            ):
                document_by_id.setdefault(doc_id, (text, metadata or {}))

        corpus_ids = list(document_by_id)
        bm25_rankings: List[List[str]] = []
        if strategy in {"hybrid", "hybrid_graph"}:
            tokenized_corpus = [tokenize_math(document_by_id[doc_id][0]) or ["<empty>"] for doc_id in corpus_ids]
            bm25 = BM25Okapi(tokenized_corpus)
            for query in normalized_queries:
                scores = bm25.get_scores(tokenize_math(query))
                ranking = [
                    corpus_ids[index]
                    for index in sorted(range(len(corpus_ids)), key=lambda item: scores[item], reverse=True)
                    if scores[index] > 0
                ][:candidate_k]
                bm25_rankings.append(ranking)

        graph_points = math_knowledge_graph.expand(knowledge_points, depth=1)
        graph_ranking = []
        if graph_points and strategy == "hybrid_graph":
            graph_ranking = sorted(
                corpus_ids,
                key=lambda doc_id: sum(
                    point in document_by_id[doc_id][0]
                    or point in str(document_by_id[doc_id][1].get("prerequisites", ""))
                    for point in graph_points
                ),
                reverse=True,
            )
            graph_ranking = [
                doc_id for doc_id in graph_ranking
                if any(point in document_by_id[doc_id][0] or point in str(document_by_id[doc_id][1]) for point in graph_points)
            ][:candidate_k]

        rankings = [*dense_rankings, *bm25_rankings]
        if graph_ranking:
            rankings.append(graph_ranking)
        fused = fuse_rankings(*rankings)
        ordered = sorted(fused, key=fused.get, reverse=True)[:candidate_k]
        candidates = []
        for rank, doc_id in enumerate(ordered, start=1):
            text, metadata = document_by_id[doc_id]
            enriched = dict(metadata)
            enriched.update({
                "chunk_id": doc_id,
                "rrf_score": round(fused[doc_id], 6),
                "rrf_rank": rank,
                "graph_expansion": "、".join(graph_points),
            })
            candidates.append(Document(page_content=text, metadata=enriched))

        trace = [
            {"stage": "query_decomposition", "queries": normalized_queries, "strategy": strategy},
            {"stage": "category_recall", "chapter": chapter, "candidates": len(document_by_id)},
            {"stage": "dense_retrieval", "rankings": [len(items) for items in dense_rankings]},
            {"stage": "bm25_retrieval", "enabled": strategy != "dense", "rankings": [len(items) for items in bm25_rankings]},
            {"stage": "graph_rag", "enabled": strategy == "hybrid_graph", "expanded_points": graph_points, "matched": len(graph_ranking)},
            {"stage": "rrf_fusion", "candidates": len(candidates)},
        ]
        return candidates, trace

    def search(
        self,
        query: str,
        chapter: str,
        knowledge_points: Sequence[str],
        candidate_k: int = RETRIEVAL_CANDIDATES,
        top_k: int = RETRIEVAL_TOP_K,
        strategy: str = "hybrid_graph",
    ) -> Tuple[List[Document], List[dict]]:
        """Deterministic utility path; production graph adds LLM-Rerank afterward."""
        candidates, trace = self.retrieve_candidates([query], chapter, knowledge_points, candidate_k, strategy)
        documents = []
        for rank, document in enumerate(candidates[:top_k], start=1):
            document.metadata.update({"rank": rank, "retrieval_score": document.metadata.get("rrf_score", 0.0)})
            documents.append(document)
        trace.append({"stage": "deterministic_top_k", "returned": len(documents)})
        return documents, trace


math_retriever = MathKnowledgeRetriever()
