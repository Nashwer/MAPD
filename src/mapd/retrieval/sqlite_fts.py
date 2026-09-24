from __future__ import annotations

import gzip
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

from mapd.retrieval.schema import RetrievedPassage


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def build_sqlite_fts_index(
    corpus_path: str | Path,
    index_path: str | Path,
    *,
    batch_size: int = 10_000,
    source_revision: str | None = None,
) -> dict[str, object]:
    """Build an atomic, persistent SQLite FTS5 index from wiki-style JSONL."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    corpus = Path(corpus_path)
    destination = Path(index_path)
    if not corpus.is_file():
        raise FileNotFoundError(corpus)
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(destination.name + ".building")
    if building.exists():
        building.unlink()

    connection = sqlite3.connect(building)
    count = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE passages (
                rowid INTEGER PRIMARY KEY,
                id TEXT NOT NULL UNIQUE,
                contents TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE passages_fts USING fts5(
                contents,
                content='passages',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        pending: list[tuple[str, str]] = []
        with _open_text(corpus) as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    identifier = str(raw.get("id", line_number - 1))
                    contents = str(raw["contents"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid corpus record at {corpus}:{line_number}") from exc
                pending.append((identifier, contents))
                if len(pending) >= batch_size:
                    connection.executemany(
                        "INSERT INTO passages(id, contents) VALUES (?, ?)", pending
                    )
                    connection.commit()
                    count += len(pending)
                    pending.clear()
                    if count % 100_000 == 0:
                        print(f"indexed source rows: {count:,}", flush=True)
            if pending:
                connection.executemany(
                    "INSERT INTO passages(id, contents) VALUES (?, ?)", pending
                )
                connection.commit()
                count += len(pending)

        print("building FTS5 postings...", flush=True)
        connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('rebuild')")
        metadata = {
            "schema": "mapd-wiki18-sqlite-fts-v1",
            "corpus": str(corpus.resolve()),
            "documents": count,
            "source_revision": source_revision,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
        connection.execute("INSERT INTO passages_fts(passages_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()

    os.replace(building, destination)
    return {**metadata, "index": str(destination.resolve())}


class SQLiteFTSRetriever:
    """Disk-backed BM25 retriever suitable for the full wiki-18 corpus."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM passages").fetchone()
            self.document_count = int(row[0])
        if self.document_count < 1:
            raise ValueError("retrieval index contains no documents")

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        expression = _fts_expression(query)
        if not expression:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT passages.id, passages.contents, bm25(passages_fts) AS rank
                FROM passages_fts
                JOIN passages ON passages.rowid = passages_fts.rowid
                WHERE passages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (expression, top_k),
            ).fetchall()
        return [
            RetrievedPassage(id=str(row[0]), contents=str(row[1]), score=-float(row[2]))
            for row in rows
        ]

    def metadata(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        return {key: json.loads(value) for key, value in rows}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        return connection


def _fts_expression(query: str) -> str:
    tokens = _TOKEN_PATTERN.findall(query.casefold())
    # FTS syntax is never interpolated directly: each token is a quoted phrase.
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
