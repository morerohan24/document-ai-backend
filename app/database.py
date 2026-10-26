import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def initialize_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_type TEXT NOT NULL,
                extracted_data TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                image_metadata TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_record(
    database_path: Path,
    *,
    record_id: str,
    filename: str,
    content_type: str,
    extracted_data: dict[str, Any],
    raw_text: str,
    image_metadata: dict[str, Any],
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO records
                (id, filename, content_type, extracted_data, raw_text,
                 image_metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                filename,
                content_type,
                json.dumps(extracted_data),
                raw_text,
                json.dumps(image_metadata),
                created_at,
            ),
        )
    return {
        "id": record_id,
        "extracted_data": extracted_data,
        "raw_text": raw_text,
        "image_metadata": image_metadata,
        "created_at": created_at,
    }


def get_record(database_path: Path, record_id: str) -> Optional[dict[str, Any]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM records WHERE id = ?", (record_id,)
        ).fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "extracted_data": json.loads(row["extracted_data"]),
        "raw_text": row["raw_text"],
        "image_metadata": json.loads(row["image_metadata"]),
        "created_at": row["created_at"],
    }
