from __future__ import annotations

import codecs
import gzip
import json
import os
import re
import sqlite3
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Iterator

from mapd.retrieval.schema import RetrievedPassage


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


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

    def __init__(self, index_path: str | Path, *, query_timeout_seconds: float = 20.0):
        self.index_path = Path(index_path)
        if query_timeout_seconds <= 0:
            raise ValueError("query_timeout_seconds must be positive")
        self.query_timeout_seconds = query_timeout_seconds
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key = 'documents'"
            ).fetchone()
            if row is not None:
                self.document_count = int(json.loads(row[0]))
            else:
                # Compatibility fallback for indexes built before metadata v1.
                row = connection.execute("SELECT COUNT(*) FROM passages").fetchone()
                self.document_count = int(row[0])
        if self.document_count < 1:
            raise ValueError("retrieval index contains no documents")

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        tokens = _query_tokens(query)
        expression = _fts_expression(tokens, operator="AND")
        if not expression:
            return []
        with self._connect() as connection:
            rows = _bounded_search_rows(
                connection,
                expression,
                top_k,
                timeout_seconds=self.query_timeout_seconds,
            )
            if not rows and len(tokens) > 1:
                # Relax only to a few selective terms. OR-ing every word in a
                # natural-language question can scan enormous posting lists
                # (for example "who", "the", "is") on the 21M-doc corpus.
                fallback = sorted(tokens, key=lambda token: (-len(token), token))[:3]
                rows = _bounded_search_rows(
                    connection,
                    _fts_expression(fallback, operator="OR"),
                    top_k,
                    timeout_seconds=self.query_timeout_seconds,
                )
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
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA mmap_size=268435456")
        return connection


def _query_tokens(query: str) -> list[str]:
    raw = _TOKEN_PATTERN.findall(query.casefold())
    filtered = [
        token
        for token in raw
        if token not in _STOP_WORDS and (len(token) > 2 or token.isdigit())
    ]
    selected = filtered or raw
    # Preserve query order, remove repeats, and bound posting-list work.
    return list(dict.fromkeys(selected))[:12]


def _fts_expression(tokens: list[str], *, operator: str) -> str:
    if operator not in {"AND", "OR"}:
        raise ValueError("FTS operator must be AND or OR")
    # FTS syntax is never interpolated directly: each token is a quoted phrase.
    quoted = [f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens]
    return f" {operator} ".join(quoted)


def _search_rows(
    connection: sqlite3.Connection, expression: str, top_k: int
) -> list[tuple[object, ...]]:
    # FTS5's hidden rank column is equivalent to bm25() by default, but its
    # LIMIT-aware query path is substantially faster than ORDER BY bm25().
    return connection.execute(
        """
        SELECT passages.id, passages.contents, passages_fts.rank
        FROM passages_fts
        JOIN passages ON passages.rowid = passages_fts.rowid
        WHERE passages_fts MATCH ?
        ORDER BY passages_fts.rank
        LIMIT ?
        """,
        (expression, top_k),
    ).fetchall()


def _bounded_search_rows(
    connection: sqlite3.Connection,
    expression: str,
    top_k: int,
    *,
    timeout_seconds: float,
) -> list[tuple[object, ...]]:
    """Run ranked retrieval with a wall-clock guard and a fast safe fallback.

    The full wiki index is commonly stored on network-backed home volumes. A
    broad ranked query must not keep the HTTP request alive indefinitely. If
    SQLite exceeds the deadline, return the first matching rows without a
    global relevance sort; the AND-first query still keeps these candidates
    useful, and callers receive a response instead of a multi-minute timeout.
    """
    deadline = time.monotonic() + timeout_seconds
    connection.set_progress_handler(
        lambda: int(time.monotonic() >= deadline),
        1_000,
    )
    try:
        return _search_rows(connection, expression, top_k)
    except sqlite3.OperationalError as exc:
        if "interrupted" not in str(exc).casefold():
            raise
        print(
            f"ranked FTS query exceeded {timeout_seconds:g}s; using unranked fallback",
            flush=True,
        )
    finally:
        connection.set_progress_handler(None, 0)

    fallback_deadline = time.monotonic() + min(timeout_seconds, 5.0)
    connection.set_progress_handler(
        lambda: int(time.monotonic() >= fallback_deadline),
        1_000,
    )
    try:
        return connection.execute(
            """
            SELECT passages.id, passages.contents, 0.0 AS rank
            FROM passages_fts
            JOIN passages ON passages.rowid = passages_fts.rowid
            WHERE passages_fts MATCH ?
            LIMIT ?
            """,
            (expression, top_k),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "interrupted" not in str(exc).casefold():
            raise
        print("unranked FTS fallback also timed out; returning no rows", flush=True)
        return []
    finally:
        connection.set_progress_handler(None, 0)


@contextmanager
def _open_text(path: Path) -> Iterator[IO[str]]:
    if path.suffix == ".gz":
        if _gzip_contains_tar(path):
            # PeterJinGo/wiki-18-corpus names the asset .jsonl.gz, but the
            # decompressed payload is a tar archive containing wiki_dump.jsonl.
            # Stream the member directly to avoid a second ~14 GB disk copy.
            with tarfile.open(path, mode="r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    binary = archive.extractfile(member)
                    if binary is None:
                        continue
                    print(f"reading archive member: {member.name}", flush=True)
                    handle = codecs.getreader("utf-8")(binary)
                    try:
                        yield handle
                    finally:
                        handle.close()
                    return
            raise ValueError(f"gzip tar archive contains no regular file: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield handle
        return
    with path.open("r", encoding="utf-8") as handle:
        yield handle


def _gzip_contains_tar(path: Path) -> bool:
    with gzip.open(path, "rb") as handle:
        header = handle.read(512)
    return len(header) >= 262 and header[257:262] == b"ustar"
