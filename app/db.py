"""SQLite persistence for models and diagnosis results.

A single connection guarded by a lock is sufficient here: writes are short
and inference happens outside the critical section.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id    TEXT NOT NULL,
    version     TEXT NOT NULL,
    spec_hash   TEXT NOT NULL,
    spec_json   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (model_id, version)
);

CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id     TEXT PRIMARY KEY,
    model_id         TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    idempotency_key  TEXT NOT NULL,
    request_hash     TEXT NOT NULL,
    request_json     TEXT NOT NULL,
    response_json    TEXT NOT NULL,
    elimination_json TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    UNIQUE (model_id, model_version, idempotency_key)
);
"""


class Database:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- models ---------------------------------------------------------------

    def get_model(self, model_id: str, version: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM models WHERE model_id = ? AND version = ?",
                (model_id, version),
            )
            return cur.fetchone()

    def insert_model(
        self, model_id: str, version: str, spec_hash: str, spec_json: str, created_at: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO models (model_id, version, spec_hash, spec_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (model_id, version, spec_hash, spec_json, created_at),
            )
            self._conn.commit()

    # -- diagnoses --------------------------------------------------------------

    def get_diagnosis_by_key(
        self, model_id: str, version: str, idempotency_key: str
    ) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM diagnoses WHERE model_id = ? AND model_version = ?"
                " AND idempotency_key = ?",
                (model_id, version, idempotency_key),
            )
            return cur.fetchone()

    def get_diagnosis_by_id(self, diagnosis_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM diagnoses WHERE diagnosis_id = ?", (diagnosis_id,)
            )
            return cur.fetchone()

    def insert_diagnosis(
        self,
        diagnosis_id: str,
        model_id: str,
        version: str,
        idempotency_key: str,
        request_hash: str,
        request_json: str,
        response_json: str,
        elimination_json: str,
        created_at: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO diagnoses (diagnosis_id, model_id, model_version,"
                " idempotency_key, request_hash, request_json, response_json,"
                " elimination_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    diagnosis_id,
                    model_id,
                    version,
                    idempotency_key,
                    request_hash,
                    request_json,
                    response_json,
                    elimination_json,
                    created_at,
                ),
            )
            self._conn.commit()
