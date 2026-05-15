import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("DB_PATH", "backend/app.db"))
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
