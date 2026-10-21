#!/usr/bin/env python3
"""Apply safe, ownership-aware migrations to an existing MySQL database."""

from __future__ import annotations

import argparse
from urllib.parse import unquote, urlparse

import mysql.connector


def connection_from_url(raw: str):
    parsed = urlparse(raw)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"} or not parsed.hostname:
        raise SystemExit("MYSQL_URL must use mysql://")
    return mysql.connector.connect(
        host=parsed.hostname,
        port=parsed.port or 3306,
        user=unquote(parsed.username or ""),
        password=unquote(parsed.password or ""),
        database=parsed.path.lstrip("/"),
    )


def migrate(connection) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute("SHOW COLUMNS FROM mistake_questions LIKE 'session_key'")
        if cursor.fetchone() is None:
            cursor.execute("ALTER TABLE mistake_questions ADD COLUMN session_key VARCHAR(128) NOT NULL DEFAULT '' AFTER user_id")
        cursor.execute("SHOW INDEX FROM chat_sessions")
        indexes = cursor.fetchall()
        if any(row[2] == "session_key" and int(row[1]) == 0 for row in indexes):
            cursor.execute("ALTER TABLE chat_sessions DROP INDEX session_key")
        if not any(row[2] == "uq_chat_session_owner" and int(row[1]) == 0 for row in indexes):
            cursor.execute("ALTER TABLE chat_sessions ADD UNIQUE KEY uq_chat_session_owner(user_id, session_key)")
        try:
            cursor.execute("CREATE INDEX idx_mistake_user_session ON mistake_questions(user_id, session_key, created_at)")
        except mysql.connector.Error as exc:
            if exc.errno != 1061:
                raise
        connection.commit()
    finally:
        cursor.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="mysql://user:password@host:3306/database")
    args = parser.parse_args()
    connection = connection_from_url(args.url)
    try:
        migrate(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
