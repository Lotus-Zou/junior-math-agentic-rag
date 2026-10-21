#!/usr/bin/env python3
"""Execute fresh MySQL schema and repeat migrations in an owned temporary DB."""

from pathlib import Path
import sys
import uuid

import mysql.connector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.migrate_mysql import migrate
from tools.reproduction_stack import configuration
from tools.reproduce_experiments import write


def main():
    state = configuration()
    connection = mysql.connector.connect(host="127.0.0.1",port=23306,user="root",password=state["database_password"])
    database = "schema_test_"+uuid.uuid4().hex
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()[0]
        cursor.execute(f"CREATE DATABASE {database} CHARACTER SET utf8mb4")
        connection.database = database
        for path in ("sql/schema.sql","sql/002_domain_model.sql"):
            cursor.execute((ROOT / path).read_text())
            while cursor.nextset():
                pass
        connection.commit()
        migrate(connection)
        migrate(connection)
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.execute("SHOW INDEX FROM mistake_questions")
        indexes = cursor.fetchall()
        write("mysql-schema",{"summary":{"passed":True,"mysql_version":version,"fresh_schema":True,"repeatable_migration":True},
                              "tables":tables,"mistake_indexes":indexes,
                              "scope":"Real MySQL 8.4 in an owned temporary database, removed afterwards. Root credential is read locally and never emitted."})
    finally:
        cursor.execute(f"DROP DATABASE IF EXISTS {database}")
        cursor.close()
        connection.close()


if __name__ == "__main__":
    main()
