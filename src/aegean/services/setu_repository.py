"""Database-backed persistence for Setu adapter state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional

from aegean.setu_models import SetuTaskRecord


class SetuTaskRepository:
    """Persist Setu tasks and subnet bindings in SQLite."""

    def __init__(self, database_url: str):
        self.db_path = self._resolve_sqlite_path(database_url)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _resolve_sqlite_path(self, database_url: str) -> Path:
        if not database_url.startswith("sqlite:///"):
            raise ValueError(
                "SETU_DB_URL currently supports sqlite URLs only, e.g. sqlite:///./.aegean/setu_tasks.db"
            )
        raw_path = database_url[len("sqlite:///") :]
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS setu_tasks (
                    task_id TEXT PRIMARY KEY,
                    subnet_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    callback_token TEXT NOT NULL,
                    callback_url TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    system_context_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision_json TEXT,
                    consensus_id TEXT,
                    metadata_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS setu_group_bindings (
                    subnet_id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    created_by TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_task(self, task: SetuTaskRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO setu_tasks (
                    task_id, subnet_id, group_id, group_name,
                    callback_token, callback_url, proposal_json, system_context_json,
                    status, decision_json, consensus_id, metadata_json,
                    error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    subnet_id=excluded.subnet_id,
                    group_id=excluded.group_id,
                    group_name=excluded.group_name,
                    callback_token=excluded.callback_token,
                    callback_url=excluded.callback_url,
                    proposal_json=excluded.proposal_json,
                    system_context_json=excluded.system_context_json,
                    status=excluded.status,
                    decision_json=excluded.decision_json,
                    consensus_id=excluded.consensus_id,
                    metadata_json=excluded.metadata_json,
                    error=excluded.error,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    task.task_id,
                    task.subnet_id,
                    task.group_id,
                    task.group_name,
                    task.callback_token,
                    task.callback_url,
                    task.proposal.model_dump_json(),
                    task.system_context.model_dump_json(),
                    task.status.value,
                    task.decision.model_dump_json() if task.decision else None,
                    task.consensus_id,
                    json.dumps(task.metadata, ensure_ascii=False),
                    task.error,
                    task.created_at.isoformat(),
                    task.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def load_task(self, task_id: str) -> Optional[SetuTaskRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM setu_tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def load_all_tasks(self) -> Dict[str, SetuTaskRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM setu_tasks").fetchall()
        return {row["task_id"]: self._row_to_task(row) for row in rows}

    def save_binding(
        self,
        subnet_id: str,
        group_id: str,
        group_name: str,
        created_by: str,
        metadata: Dict,
        created_at: str,
        updated_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO setu_group_bindings (
                    subnet_id, group_id, group_name, created_by,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subnet_id) DO UPDATE SET
                    group_id=excluded.group_id,
                    group_name=excluded.group_name,
                    created_by=excluded.created_by,
                    metadata_json=excluded.metadata_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    subnet_id,
                    group_id,
                    group_name,
                    created_by,
                    json.dumps(metadata, ensure_ascii=False),
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()

    def load_bindings(self) -> Dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT subnet_id, group_id FROM setu_group_bindings"
            ).fetchall()
        return {row["subnet_id"]: row["group_id"] for row in rows}

    def _row_to_task(self, row: sqlite3.Row) -> SetuTaskRecord:
        from datetime import datetime

        decision_json = row["decision_json"]
        return SetuTaskRecord.model_validate(
            {
                "task_id": row["task_id"],
                "subnet_id": row["subnet_id"],
                "group_id": row["group_id"],
                "group_name": row["group_name"],
                "callback_token": row["callback_token"],
                "callback_url": row["callback_url"],
                "proposal": json.loads(row["proposal_json"]),
                "system_context": json.loads(row["system_context_json"]),
                "status": row["status"],
                "decision": json.loads(decision_json) if decision_json else None,
                "consensus_id": row["consensus_id"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "error": row["error"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "updated_at": datetime.fromisoformat(row["updated_at"]),
            }
        )

