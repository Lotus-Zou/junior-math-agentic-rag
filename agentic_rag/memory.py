# -*- coding: utf-8 -*-
"""User-scoped long-term memory with MySQL production and SQLite fallback."""

from __future__ import annotations

import datetime
import math
import re
import sqlite3
from threading import Lock
from urllib.parse import unquote, urlparse

try:
    import chromadb
except ImportError:  # Local metadata tests can run without the optional vector extra.
    chromadb = None

from config import CHROMA_PATH, MEMORY_TTL_DAYS, MYSQL_POOL_SIZE, MYSQL_URL

DB_PATH = "long_term_memory.sqlite"
MEMORY_COLLECTION_NAME = "math_learning_memory"
ALLOWED_MEMORY_TYPES = {"mistake_pattern", "knowledge_gap", "preference"}
_mysql_pool = None
_mysql_lock = Lock()


def sanitize_memory_text(text: str) -> str:
    value = (text or "").strip()[:2000]
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED_API_KEY]", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", value)
    return re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[REDACTED_PHONE]", value)


def _mysql_config() -> dict[str, object]:
    parsed = urlparse(MYSQL_URL)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"} or not parsed.hostname:
        raise ValueError("MYSQL_URL must use mysql://")
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/"),
        "pool_name": "math_agent_memory_pool",
        "pool_size": MYSQL_POOL_SIZE,
        "connection_timeout": 2,
        "read_timeout": 3,
        "write_timeout": 3,
    }


def _mysql_connection():
    global _mysql_pool
    if not MYSQL_URL:
        return None
    with _mysql_lock:
        if _mysql_pool is None:
            import mysql.connector.pooling

            _mysql_pool = mysql.connector.pooling.MySQLConnectionPool(**_mysql_config())
        return _mysql_pool.get_connection()


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def memory_ready() -> bool:
    try:
        if MYSQL_URL:
            connection = _mysql_connection()
            try:
                connection.ping(reconnect=True, attempts=1, delay=0)
            finally:
                connection.close()
        else:
            initialize_memory_db()
        return True
    except Exception:
        return False


def _collection():
    if chromadb is None:
        raise RuntimeError("chromadb is not installed")
    from agentic_rag.chains import get_embedding_function

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        MEMORY_COLLECTION_NAME,
        embedding_function=get_embedding_function(),
    )


def initialize_memory_db():
    if MYSQL_URL:
        connection = _mysql_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id BIGINT PRIMARY KEY AUTO_INCREMENT,
                    user_id VARCHAR(128) NOT NULL,
                    session_id VARCHAR(128) NOT NULL,
                    text TEXT NOT NULL,
                    type VARCHAR(64) NOT NULL DEFAULT 'knowledge_gap',
                    importance TINYINT NOT NULL DEFAULT 5,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    INDEX idx_memory_owner (user_id, session_id, created_at)
                )"""
            )
            cursor.execute("CREATE TABLE IF NOT EXISTS memory_task_effects (effect_key VARCHAR(64) PRIMARY KEY, memory_id BIGINT NULL)")
            connection.commit()
        finally:
            connection.close()
        return
    with get_db_connection() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'anonymous',
                session_id TEXT NOT NULL DEFAULT 'anonymous',
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'knowledge_gap',
                importance INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )"""
        )
        connection.execute("CREATE TABLE IF NOT EXISTS memory_task_effects (effect_key TEXT PRIMARY KEY, memory_id INTEGER)")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(memories)")}
        if "user_id" not in columns:
            connection.execute("ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'anonymous'")
        if "session_id" not in columns:
            connection.execute("ALTER TABLE memories ADD COLUMN session_id TEXT NOT NULL DEFAULT 'anonymous'")
        if "expires_at" not in columns:
            connection.execute("ALTER TABLE memories ADD COLUMN expires_at TIMESTAMP")
        connection.execute(
            "UPDATE memories SET expires_at = datetime(created_at, ?) WHERE expires_at IS NULL",
            (f"+{MEMORY_TTL_DAYS} days",),
        )


def _owner(user_id: str, session_id: str) -> tuple[str, str]:
    return (user_id or "anonymous")[:128], (session_id or "anonymous")[:128]


def add_memory(text: str, type: str = "knowledge_gap", importance: int = 5, *, user_id: str = "anonymous", session_id: str = "anonymous", idempotency_key: str = ""):
    text = sanitize_memory_text(text)
    if not text:
        return None
    type = type if type in ALLOWED_MEMORY_TYPES else "knowledge_gap"
    user_id, session_id = _owner(user_id, session_id)
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    expires = now + datetime.timedelta(days=MEMORY_TTL_DAYS)
    connection = _mysql_connection()
    if connection:
        try:
            cursor = connection.cursor()
            memory_id = None
            if idempotency_key:
                cursor.execute("INSERT IGNORE INTO memory_task_effects(effect_key) VALUES (%s)", (idempotency_key,))
                cursor.execute("SELECT memory_id FROM memory_task_effects WHERE effect_key=%s FOR UPDATE", (idempotency_key,))
                memory_id = cursor.fetchone()[0]
            if memory_id is None:
                cursor.execute(
                    "INSERT INTO memories (user_id, session_id, text, type, importance, created_at, last_accessed_at, expires_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (user_id, session_id, text, type, max(1, min(10, int(importance))), now, now, expires),
                )
                memory_id = cursor.lastrowid
                if idempotency_key:
                    cursor.execute("UPDATE memory_task_effects SET memory_id=%s WHERE effect_key=%s", (memory_id, idempotency_key))
            connection.commit()
        finally:
            connection.close()
    else:
        initialize_memory_db()
        with get_db_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute("SELECT memory_id FROM memory_task_effects WHERE effect_key=?", (idempotency_key,)).fetchone() if idempotency_key else None
            if existing:
                memory_id = existing[0]
            else:
                cursor = connection.execute(
                    "INSERT INTO memories (user_id, session_id, text, type, importance, created_at, last_accessed_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, session_id, text, type, max(1, min(10, int(importance))), now, now, expires),
                )
                memory_id = cursor.lastrowid
                if idempotency_key:
                    connection.execute("INSERT INTO memory_task_effects VALUES (?,?)", (idempotency_key, memory_id))
    _collection().upsert(ids=[str(memory_id)], documents=[text], metadatas=[{"type": type, "importance": int(importance), "sqlite_id": memory_id, "user_id": user_id, "session_id": session_id}])
    return memory_id


def retrieve_memories(query_text: str, top_k: int = 3, *, user_id: str = "anonymous", session_id: str = "anonymous") -> list[dict]:
    user_id, session_id = _owner(user_id, session_id)
    collection = _collection()
    if collection.count() == 0:
        return []
    results = collection.query(
        query_texts=[query_text],
        n_results=min(collection.count(), max(top_k, top_k * 3)),
        where={"$and": [{"user_id": user_id}, {"session_id": session_id}]},
    )
    ranked = []
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    connection = _mysql_connection()
    try:
        for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
            if connection:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM memories WHERE id = %s AND user_id = %s AND session_id = %s AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())", (int(memory_id), user_id, session_id))
                row = cursor.fetchone()
            else:
                with get_db_connection() as sqlite:
                    sqlite.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
                    row = sqlite.execute("SELECT * FROM memories WHERE id = ? AND user_id = ? AND session_id = ? AND (expires_at IS NULL OR expires_at > ?)", (int(memory_id), user_id, session_id, now)).fetchone()
            if not row:
                continue
            last_accessed = row["last_accessed_at"]
            if isinstance(last_accessed, str):
                last_accessed = datetime.datetime.fromisoformat(last_accessed)
            hours = max(0.0, (now - last_accessed.replace(tzinfo=None)).total_seconds() / 3600)
            score = (1.0 / (1.0 + distance)) * (1 + 0.1 * row["importance"]) * (1 + 0.5 / (1.0 + math.log1p(hours)))
            ranked.append({"id": row["id"], "text": row["text"], "type": row["type"], "score": score, "user_id": row["user_id"], "session_id": row["session_id"]})
        ranked.sort(key=lambda item: item["score"], reverse=True)
        selected = ranked[:top_k]
        for item in selected:
            if connection:
                cursor = connection.cursor()
                cursor.execute("UPDATE memories SET last_accessed_at = %s WHERE id = %s AND user_id = %s AND session_id = %s", (now, item["id"], user_id, session_id))
            else:
                with get_db_connection() as sqlite:
                    sqlite.execute("UPDATE memories SET last_accessed_at = ? WHERE id = ? AND user_id = ? AND session_id = ?", (now, item["id"], user_id, session_id))
        if connection:
            connection.commit()
        return selected
    finally:
        if connection:
            connection.close()


def delete_memory(memory_id: int, *, user_id: str = "anonymous", session_id: str = "anonymous"):
    user_id, session_id = _owner(user_id, session_id)
    connection = _mysql_connection()
    deleted = False
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM memories WHERE id = %s AND user_id = %s AND session_id = %s", (memory_id, user_id, session_id))
            deleted = cursor.rowcount > 0
            connection.commit()
        finally:
            connection.close()
    else:
        with get_db_connection() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ? AND user_id = ? AND session_id = ?", (memory_id, user_id, session_id))
            deleted = cursor.rowcount > 0
    if deleted:
        _collection().delete(ids=[str(memory_id)])
    return deleted


def view_memories(limit: int = 10, *, user_id: str = "anonymous", session_id: str = "anonymous"):
    user_id, session_id = _owner(user_id, session_id)
    connection = _mysql_connection()
    try:
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM memories WHERE user_id = %s AND session_id = %s AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP()) ORDER BY created_at DESC LIMIT %s", (user_id, session_id, limit))
            return cursor.fetchall()
        with get_db_connection() as sqlite:
            rows = sqlite.execute("SELECT * FROM memories WHERE user_id = ? AND session_id = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP) ORDER BY created_at DESC LIMIT ?", (user_id, session_id, limit)).fetchall()
            return [dict(row) for row in rows]
    finally:
        if connection:
            connection.close()
