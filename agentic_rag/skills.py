# -*- coding: utf-8 -*-
"""Whitelisted business Skills shared by Tool-Calling and MCP."""

from __future__ import annotations

from langchain_core.tools import tool

from agentic_rag.chains import get_answer_validation_chain, get_question_parser_chain
from agentic_rag.guardrails import ensure_tool_allowed
from agentic_rag.math_retriever import math_retriever
from agentic_rag.math_taxonomy import classify_math_text
from agentic_rag.math_validation import deterministic_math_checks


def _serialize_documents(documents):
    return [
        {"chunk_id": doc.metadata.get("chunk_id"), "content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]


def question_parse_service(query: str, conversation_summary: str = "") -> dict:
    ensure_tool_allowed("question_parse_skill")
    return get_question_parser_chain().invoke({"query": query, "conversation_summary": conversation_summary or "无"})


def math_retrieval_service(query: str, sub_queries: list[str] | None = None, chapter: str = "", knowledge_points: list[str] | None = None, top_k: int = 6) -> dict:
    ensure_tool_allowed("math_retrieval_skill")
    classification = classify_math_text(query)
    chapter = chapter or classification.chapter
    points = knowledge_points or classification.knowledge_points
    candidates, trace = math_retriever.retrieve_candidates(sub_queries or [query], chapter, points)
    return {"documents": _serialize_documents(candidates[:top_k]), "trace": trace}


def similar_exercise_service(query: str, knowledge_points: list[str] | None = None, top_k: int = 3) -> dict:
    ensure_tool_allowed("similar_exercise_skill")
    classification = classify_math_text(query)
    documents, trace = math_retriever.search(f"典型例题 巩固练习 {query}", classification.chapter, knowledge_points or classification.knowledge_points, top_k=top_k)
    return {"exercises": _serialize_documents(documents), "trace": trace}


def answer_verify_service(query: str, answer: str, contexts: list[str], student_answer: str = "") -> dict:
    ensure_tool_allowed("answer_verify_skill")
    context_text = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(contexts, start=1))
    deterministic = deterministic_math_checks(query, answer, len(contexts))
    critic = get_answer_validation_chain().invoke({"query": query, "student_answer": student_answer, "context": context_text, "answer": answer, "deterministic_checks": deterministic})
    return {"deterministic": deterministic, "critic": critic}


@tool("question_parse_skill")
def question_parse_skill(query: str, conversation_summary: str = "") -> dict:
    """Parse a math mistake into stem, student answer, intent, and visible error clues."""
    return question_parse_service(query, conversation_summary)


@tool("math_retrieval_skill")
def math_retrieval_skill(query: str, chapter: str = "", knowledge_points: str = "") -> dict:
    """Run dense, BM25, GraphRAG, and RRF retrieval with source snippets."""
    points = [item.strip() for item in knowledge_points.split("、") if item.strip()]
    return math_retrieval_service(query, [query], chapter, points)


@tool("similar_exercise_skill")
def similar_exercise_skill(query: str, knowledge_points: str = "") -> dict:
    """Retrieve similar exercises for the current knowledge point."""
    points = [item.strip() for item in knowledge_points.split("、") if item.strip()]
    return similar_exercise_service(query, points)


@tool("answer_verify_skill")
def answer_verify_skill(query: str, answer: str, context: str) -> dict:
    """Use deterministic math checks and an isolated Critic to verify a draft."""
    return answer_verify_service(query, answer, [context])


ALL_SKILLS = [question_parse_skill, math_retrieval_skill, similar_exercise_skill, answer_verify_skill]
