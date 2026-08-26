# -*- coding: utf-8 -*-
"""SQLite + vector memory for reusable student learning signals."""

import datetime
import math
import re
import sqlite3

import chromadb

from agentic_rag.chains import get_embedding_function
from config import CHROMA_PATH


DB_PATH = "long_term_memory.sqlite"
MEMORY_COLLECTION_NAME = "math_learning_memory"
ALLOWED_MEMORY_TYPES = {"mistake_pattern", "knowledge_gap", "preference"}


def sanitize_memory_text(text: str) -> str:
    value = (text or "").strip()[:2000]
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_API_KEY]", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", value)
    value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[REDACTED_PHONE]", value)
    return value


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def _collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        MEMORY_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def initialize_memory_db():
    """Initialize durable metadata without blocking startup on embedding downloads."""
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'knowledge_gap',
                importance INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_memory(text: str, type: str = "knowledge_gap", importance: int = 5):
    text = sanitize_memory_text(text)
    if not text:
        return None
    type = type if type in ALLOWED_MEMORY_TYPES else "knowledge_gap"
    now = datetime.datetime.now()
    with get_db_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO memories (text, type, importance, created_at, last_accessed_at) VALUES (?, ?, ?, ?, ?)",
            (text, type, max(1, min(10, int(importance))), now, now),
        )
        memory_id = cursor.lastrowid
    _collection().upsert(
        ids=[str(memory_id)],
        documents=[text],
        metadatas=[{"type": type, "importance": int(importance), "sqlite_id": memory_id}],
    )
    return memory_id


def retrieve_memories(query_text: str, top_k: int = 3) -> list[dict]:
    collection = _collection()
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_texts=[query_text],
        n_results=min(count, max(top_k, top_k * 3)),
    )
    ranked = []
    now = datetime.datetime.now()
    with get_db_connection() as connection:
        for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (int(memory_id),)).fetchone()
            if not row:
                continue
            semantic_score = 1.0 / (1.0 + distance)
            last_accessed = datetime.datetime.fromisoformat(row["last_accessed_at"])
            hours = max(0.0, (now - last_accessed).total_seconds() / 3600)
            recency_score = 1.0 / (1.0 + math.log1p(hours))
            score = semantic_score * (1 + 0.1 * row["importance"]) * (1 + 0.5 * recency_score)
            ranked.append({"id": row["id"], "text": row["text"], "type": row["type"], "score": score})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[:top_k]
        for item in selected:
            connection.execute("UPDATE memories SET last_accessed_at = ? WHERE id = ?", (now, item["id"]))
    return selected


def delete_memory(memory_id: int):
    with get_db_connection() as connection:
        connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    _collection().delete(ids=[str(memory_id)])


def view_memories(limit: int = 10):
    with get_db_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
