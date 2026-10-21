"""Transactional MySQL domain store with a SQLite development fallback."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any
from threading import Lock

from agentic_rag import memory


def _mysql():
    return memory._mysql_connection()


_schema_lock = Lock()
_schema_ready = False


def initialize_domain_schema() -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        _initialize_domain_schema()
        _schema_ready = True


def _initialize_domain_schema() -> None:
    connection = _mysql()
    if connection:
        statements = (
            "CREATE TABLE IF NOT EXISTS users (id BIGINT PRIMARY KEY AUTO_INCREMENT, external_id VARCHAR(128) NOT NULL UNIQUE, grade TINYINT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
            "CREATE TABLE IF NOT EXISTS chat_sessions (id BIGINT PRIMARY KEY AUTO_INCREMENT, user_id BIGINT NOT NULL, session_key VARCHAR(128) NOT NULL, summary TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, UNIQUE KEY uq_chat_session_owner(user_id, session_key), INDEX idx_session_user(user_id))",
            "CREATE TABLE IF NOT EXISTS chat_messages (id BIGINT PRIMARY KEY AUTO_INCREMENT, user_id BIGINT NOT NULL, session_key VARCHAR(128) NOT NULL, role VARCHAR(32) NOT NULL, content TEXT NOT NULL, trace_id VARCHAR(64), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, INDEX idx_message_session(user_id, session_key, created_at))",
            "CREATE TABLE IF NOT EXISTS mistake_questions (id BIGINT PRIMARY KEY AUTO_INCREMENT, user_id BIGINT NOT NULL, session_key VARCHAR(128) NOT NULL, question_text TEXT NOT NULL, answer_text TEXT, knowledge_points JSON, error_type VARCHAR(64), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, INDEX idx_mistake_user_time(user_id, created_at))",
            "CREATE TABLE IF NOT EXISTS user_feedback (id BIGINT PRIMARY KEY AUTO_INCREMENT, user_id BIGINT, trace_id VARCHAR(64) NOT NULL, correct BOOLEAN NOT NULL, comment VARCHAR(2000), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, INDEX idx_feedback_trace(trace_id))",
        )
        try:
            cursor = connection.cursor()
            for statement in statements:
                cursor.execute(statement)
            cursor.execute("SHOW INDEX FROM chat_sessions")
            indexes = cursor.fetchall()
            if any(row[2] == "session_key" and int(row[1]) == 0 for row in indexes):
                cursor.execute("ALTER TABLE chat_sessions DROP INDEX session_key")
            if not any(row[2] == "uq_chat_session_owner" and int(row[1]) == 0 for row in indexes):
                cursor.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_session_owner(user_id, session_key)")
            try:
                cursor.execute("CREATE INDEX idx_mistake_user_session ON mistake_questions(user_id, session_key, created_at)")
            except Exception as exc:
                if "Duplicate key name" not in str(exc):
                    raise
            connection.commit()
        finally:
            connection.close()
        return
    with memory.get_db_connection() as connection:
        existing = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name='chat_sessions'"
        ).fetchone()
        if existing and "UNIQUE" in str(existing[1]).upper() and "SESSION_KEY" in str(existing[1]).upper():
            # Migrate the pre-isolation schema whose global session_key UNIQUE
            # constraint made one user's session overwrite another user's data.
            connection.executescript(
                """
                CREATE TABLE chat_sessions_migrated (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_key TEXT NOT NULL,
                    summary TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO chat_sessions_migrated(id, user_id, session_key, summary, updated_at)
                    SELECT id, user_id, session_key, summary, updated_at FROM chat_sessions;
                DROP TABLE chat_sessions;
                ALTER TABLE chat_sessions_migrated RENAME TO chat_sessions;
                """
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                external_id TEXT NOT NULL UNIQUE,
                grade INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_key TEXT NOT NULL,
                summary TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_key TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                trace_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS mistake_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_key TEXT NOT NULL,
                question_text TEXT NOT NULL,
                answer_text TEXT,
                knowledge_points TEXT,
                error_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                trace_id TEXT NOT NULL,
                correct INTEGER NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_chat_session_owner
                ON chat_sessions(user_id, session_key);
            CREATE INDEX IF NOT EXISTS idx_message_session
                ON chat_messages(user_id, session_key, created_at);
            CREATE INDEX IF NOT EXISTS idx_mistake_user_session
                ON mistake_questions(user_id, session_key, created_at);
            """
        )


def ensure_user(external_id: str, grade: int | None = None) -> int:
    initialize_domain_schema()
    external_id = (external_id or "anonymous")[:128]
    connection = _mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO users (external_id, grade) VALUES (%s, %s) ON DUPLICATE KEY UPDATE id=LAST_INSERT_ID(id), grade=COALESCE(VALUES(grade), grade)",
                (external_id, grade),
            )
            connection.commit()
            if cursor.lastrowid:
                return int(cursor.lastrowid)
            cursor.execute("SELECT id FROM users WHERE external_id = %s", (external_id,))
            return int(cursor.fetchone()[0])
        finally:
            connection.close()
    with memory.get_db_connection() as connection:
        connection.execute(
            "INSERT INTO users (external_id, grade) VALUES (?, ?) ON CONFLICT(external_id) DO UPDATE SET grade=COALESCE(excluded.grade, users.grade)",
            (external_id, grade),
        )
        row = connection.execute("SELECT id FROM users WHERE external_id = ?", (external_id,)).fetchone()
        return int(row[0])


def upsert_session(external_id: str, session_key: str, summary: str = "") -> int:
    user_id = ensure_user(external_id)
    session_key = (session_key or "anonymous")[:128]
    connection = _mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (user_id, session_key, summary) VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE user_id=VALUES(user_id), summary=VALUES(summary), updated_at=CURRENT_TIMESTAMP",
                (user_id, session_key, summary[:8000]),
            )
            connection.commit()
            cursor.execute("SELECT id FROM chat_sessions WHERE user_id = %s AND session_key = %s", (user_id, session_key))
            return int(cursor.fetchone()[0])
        finally:
            connection.close()
    with memory.get_db_connection() as connection:
        connection.execute(
            "INSERT INTO chat_sessions (user_id, session_key, summary) VALUES (?, ?, ?) ON CONFLICT(user_id, session_key) DO UPDATE SET summary=excluded.summary, updated_at=CURRENT_TIMESTAMP",
            (user_id, session_key, summary[:8000]),
        )
        row = connection.execute("SELECT id FROM chat_sessions WHERE user_id = ? AND session_key = ?", (user_id, session_key)).fetchone()
        return int(row[0])


def record_turn(external_id: str, session_key: str, query: str, answer: str, trace_id: str, *, summary: str = "", knowledge_points: list[str] | None = None, error_type: str = "") -> None:
    user_id = ensure_user(external_id)
    upsert_session(external_id, session_key, summary)
    connection = _mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO chat_messages (user_id, session_key, role, content, trace_id) VALUES (%s,%s,%s,%s,%s)", (user_id, session_key, "student", query[:8000], trace_id))
            cursor.execute("INSERT INTO chat_messages (user_id, session_key, role, content, trace_id) VALUES (%s,%s,%s,%s,%s)", (user_id, session_key, "tutor", answer[:16000], trace_id))
            if error_type or knowledge_points:
                cursor.execute("INSERT INTO mistake_questions (user_id, session_key, question_text, answer_text, knowledge_points, error_type) VALUES (%s,%s,%s,%s,%s,%s)", (user_id, session_key, query[:8000], answer[:16000], json.dumps(knowledge_points or [], ensure_ascii=False), error_type[:64]))
            connection.commit()
        finally:
            connection.close()
        return
    with memory.get_db_connection() as connection:
        connection.execute("INSERT INTO chat_messages (user_id, session_key, role, content, trace_id) VALUES (?,?,?,?,?)", (user_id, session_key, "student", query[:8000], trace_id))
        connection.execute("INSERT INTO chat_messages (user_id, session_key, role, content, trace_id) VALUES (?,?,?,?,?)", (user_id, session_key, "tutor", answer[:16000], trace_id))
        if error_type or knowledge_points:
            connection.execute("INSERT INTO mistake_questions (user_id, session_key, question_text, answer_text, knowledge_points, error_type) VALUES (?,?,?,?,?,?)", (user_id, session_key, query[:8000], answer[:16000], json.dumps(knowledge_points or [], ensure_ascii=False), error_type[:64]))


def record_feedback(external_id: str, trace_id: str, correct: bool, comment: str = "") -> None:
    user_id = ensure_user(external_id)
    connection = _mysql()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("INSERT INTO user_feedback (user_id, trace_id, correct, comment) VALUES (%s,%s,%s,%s)", (user_id, trace_id[:64], correct, comment[:2000]))
            connection.commit()
        finally:
            connection.close()
        return
    with memory.get_db_connection() as connection:
        connection.execute("INSERT INTO user_feedback (user_id, trace_id, correct, comment) VALUES (?,?,?,?)", (user_id, trace_id[:64], int(correct), comment[:2000]))


def learning_report(external_id: str, session_key: str) -> dict[str, Any]:
    user_id = ensure_user(external_id)
    connection = _mysql()
    try:
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT COUNT(*) AS total FROM mistake_questions WHERE user_id=%s AND session_key=%s", (user_id, session_key))
            total = int(cursor.fetchone()["total"])
            cursor.execute("SELECT error_type, knowledge_points FROM mistake_questions WHERE user_id=%s AND session_key=%s ORDER BY created_at DESC, id DESC LIMIT 200", (user_id, session_key))
            rows = cursor.fetchall()
        else:
            with memory.get_db_connection() as sqlite:
                total = int(sqlite.execute("SELECT COUNT(*) FROM mistake_questions WHERE user_id=? AND session_key=?", (user_id, session_key)).fetchone()[0])
                rows = [dict(row) for row in sqlite.execute("SELECT error_type, knowledge_points FROM mistake_questions WHERE user_id=? AND session_key=? ORDER BY created_at DESC, id DESC LIMIT 200", (user_id, session_key)).fetchall()]
        error_counts = Counter(row.get("error_type") or "未分类" for row in rows)
        points = Counter()
        for row in rows:
            raw = row.get("knowledge_points") or []
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = []
            points.update(item for item in raw if item)
        return {
            "mistake_count": total,
            "analyzed_count": len(rows),
            "analysis_limit": 200,
            "top_error_types": error_counts.most_common(5),
            "top_knowledge_points": points.most_common(5),
            "recommendations": [f"优先复习：{item[0]}" for item in points.most_common(3)] or ["继续完成一道题后再生成报告"],
        }
    finally:
        if connection:
            connection.close()


def list_mistakes(external_id: str, session_key: str, limit: int = 50) -> list[dict[str, Any]]:
    user_id = ensure_user(external_id)
    limit = max(1, min(200, int(limit)))
    connection = _mysql()
    try:
        if connection:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT id, question_text, answer_text, knowledge_points, error_type, created_at FROM mistake_questions WHERE user_id=%s AND session_key=%s ORDER BY created_at DESC, id DESC LIMIT %s", (user_id, session_key, limit))
            rows = cursor.fetchall()
        else:
            with memory.get_db_connection() as sqlite:
                rows = [dict(row) for row in sqlite.execute("SELECT id, question_text, answer_text, knowledge_points, error_type, created_at FROM mistake_questions WHERE user_id=? AND session_key=? ORDER BY created_at DESC, id DESC LIMIT ?", (user_id, session_key, limit)).fetchall()]
        for row in rows:
            if isinstance(row.get("knowledge_points"), str):
                try:
                    row["knowledge_points"] = json.loads(row["knowledge_points"])
                except json.JSONDecodeError:
                    row["knowledge_points"] = []
        return rows
    finally:
        if connection:
            connection.close()
