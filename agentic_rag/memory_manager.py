# -*- coding: utf-8 -*-
"""Three-level memory management with automatic conversation compression."""

from __future__ import annotations

from config import MAX_HISTORY_MESSAGES


def history_text(history: list[dict], limit: int | None = None) -> str:
    items = history[-limit:] if limit else history
    return "\n".join(f"{item.get('role', 'unknown')}: {item.get('content', '')}" for item in items) or "无"


def compress_history(history: list[dict], existing_summary: str = "") -> tuple[str, list[dict]]:
    if len(history) <= MAX_HISTORY_MESSAGES:
        return existing_summary, history
    older = history[:-MAX_HISTORY_MESSAGES]
    retained = history[-MAX_HISTORY_MESSAGES:]
    try:
        from agentic_rag.chains import get_summarizer_chain, message_text
        source = f"已有摘要:\n{existing_summary or '无'}\n\n待压缩历史:\n{history_text(older)}"
        summary = message_text(get_summarizer_chain().invoke({"conversation_history": source}))
    except Exception:
        summary = (existing_summary + "\n" + history_text(older, limit=6)).strip()
    return summary, retained


def working_memory(state: dict) -> dict:
    return {
        "query": state.get("updated_query") or state.get("query"),
        "sub_queries": state.get("sub_queries", []),
        "knowledge_points": state.get("knowledge_points", []),
        "retrieved_chunk_ids": [document.metadata.get("chunk_id") for document in state.get("documents", [])],
        "validation_issues": state.get("validation_issues", []),
    }
