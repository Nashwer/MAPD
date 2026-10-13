from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from mapd.retrieval.schema import RetrievedPassage


INDEX_SCHEMA = "mapd-wiki18-e5-ivfpq-v1"


class EmbeddingEncoder(Protocol):
    dimension: int

    def encode_queries(self, texts: Sequence[str]) -> Any: ...

    def encode_passages(self, texts: Sequence[str]) -> Any: ...


def _optional_imports() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on server extras
        raise RuntimeError(
            "compressed E5 retrieval requires faiss-cpu and numpy; "
            "run `bash mapd.sh bootstrap --skip-model` after updating the repository"
        ) from exc
    return faiss, np


class E5Encoder:
    """Search-R1-compatible E5 mean-pooling encoder.

    Passage construction uses the paper stack's ``passage:`` prefix and a
    default maximum length of 180. Query inference uses ``query:`` and 256
    tokens. Embeddings are L2-normalized so inner product is cosine similarity.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        *,
        device: str = "auto",
        passage_max_length: int = 180,
        query_max_length: int = 256,
        use_fp16: bool = True,
    ) -> None:
        if passage_max_length < 1 or query_max_length < 1:
            raise ValueError("encoder maximum lengths must be positive")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on server extras
            raise RuntimeError(
                "E5 encoding requires torch and transformers from the server environment"
            ) from exc

        resolved_device = device
        if resolved_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        if resolved_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for E5 but torch reports no CUDA device")

        self.torch = torch
        self.device = torch.device(resolved_device)
        self.passage_max_length = passage_max_length
        self.query_max_length = query_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval().to(self.device)
        if use_fp16 and self.device.type == "cuda":
            self.model.half()
        self.dimension = int(self.model.config.hidden_size)
        self.model_name = model_name

    def encode_queries(self, texts: Sequence[str]) -> Any:
        return self._encode([f"query: {text}" for text in texts], self.query_max_length)

    def encode_passages(self, texts: Sequence[str]) -> Any:
        return self._encode([f"passage: {text}" for text in texts], self.passage_max_length)

    def _encode(self, texts: Sequence[str], max_length: int) -> Any:
        if not texts:
            raise ValueError("cannot encode an empty text batch")
        torch = self.torch
        inputs = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model(**inputs, return_dict=True)
            mask = inputs["attention_mask"]
            hidden = output.last_hidden_state.masked_fill(~mask[..., None].bool(), 0.0)
            embeddings = hidden.sum(dim=1) / mask.sum(dim=1)[..., None].clamp_min(1)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.detach().float().cpu().numpy().astype("float32", copy=False)


def metadata_path(index_path: str | Path) -> Path:
    path = Path(index_path)
    return path.with_name(path.name + ".json")


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_index_atomic(faiss: Any, index: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    faiss.write_index(index, str(temporary))
    os.replace(temporary, path)


def _docstore_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    try:
        rows = connection.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(key): json.loads(value) for key, value in rows}


def _validate_docstore(connection: sqlite3.Connection) -> int:
    count, minimum, maximum = connection.execute(
        "SELECT COUNT(*), MIN(rowid), MAX(rowid) FROM passages"
    ).fetchone()
    count = int(count)
    if count < 1:
        raise ValueError("document store contains no passages")
    if int(minimum) != 1 or int(maximum) != count:
        raise ValueError("document store rowids must be contiguous from 1")
    return count


def _iter_batches(
    connection: sqlite3.Connection,
    *,
    start_rowid: int,
    stop_rowid: int,
    batch_size: int,
) -> Iterator[list[tuple[int, str]]]:
    cursor = connection.execute(
        "SELECT rowid, contents FROM passages "
        "WHERE rowid >= ? AND rowid <= ? ORDER BY rowid",
        (start_rowid, stop_rowid),
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        yield [(int(rowid), str(contents)) for rowid, contents in rows]


def _training_texts(
    connection: sqlite3.Connection, *, document_count: int, train_size: int
) -> list[str]:
    sample_size = min(document_count, train_size)
    stride = max(1, document_count // sample_size)
    rows = connection.execute(
        "SELECT contents FROM passages "
        "WHERE ((rowid - 1) % ?) = 0 ORDER BY rowid LIMIT ?",
        (stride, sample_size),
    )
    return [str(row[0]) for row in rows]


def build_e5_ivfpq_index(
    docstore_path: str | Path,
    index_path: str | Path,
    *,
    model_name: str = "intfloat/e5-base-v2",
    device: str = "auto",
    batch_size: int = 128,
    passage_max_length: int = 180,
    nlist: int = 16_384,
    pq_m: int = 96,
    pq_nbits: int = 8,
    train_size: int = 262_144,
    nprobe: int = 64,
    checkpoint_every: int = 1_000_000,
    threads: int | None = None,
    limit: int | None = None,
    force: bool = False,
    encoder: EmbeddingEncoder | None = None,
) -> dict[str, Any]:
    """Build a resumable compressed E5 IVF-PQ index over a SQLite docstore."""
    if min(batch_size, nlist, pq_m, pq_nbits, train_size, nprobe, checkpoint_every) < 1:
        raise ValueError("all index parameters must be positive")
    faiss, np = _optional_imports()
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive")
        faiss.omp_set_num_threads(threads)

    docstore = Path(docstore_path)
    destination = Path(index_path)
    if not docstore.is_file():
        raise FileNotFoundError(docstore)
    destination.parent.mkdir(parents=True, exist_ok=True)
    final_metadata_path = metadata_path(destination)
    building = destination.with_name(destination.name + ".building")
    building_metadata_path = metadata_path(building)

    if not force and destination.exists() != final_metadata_path.exists():
        raise RuntimeError(
            "completed index and metadata sidecar are inconsistent; use --force to rebuild"
        )
    if force:
        for path in (destination, final_metadata_path, building, building_metadata_path):
            if path.exists():
                path.unlink()

    connection = sqlite3.connect(f"file:{docstore}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    try:
        available_documents = _validate_docstore(connection)
        document_count = min(available_documents, limit) if limit else available_documents
        if document_count < 1:
            raise ValueError("limit selected no documents")
        source_metadata = _docstore_metadata(connection)
        reusable_configuration = {
            "schema": INDEX_SCHEMA,
            "model": model_name,
            "passage_max_length": passage_max_length,
            "nlist": nlist,
            "pq_m": pq_m,
            "pq_nbits": pq_nbits,
            "document_store": str(docstore.resolve()),
            "source_revision": source_metadata.get("source_revision"),
            "source_documents": available_documents,
            "documents": document_count,
            "training_passages": min(document_count, train_size),
        }
        if destination.is_file() and final_metadata_path.is_file() and not force:
            value = json.loads(final_metadata_path.read_text(encoding="utf-8"))
            for key, expected in reusable_configuration.items():
                if value.get(key) != expected:
                    raise RuntimeError(
                        f"completed index configuration mismatch for {key}: "
                        f"{value.get(key)!r} != {expected!r}; use --force to rebuild"
                    )
            if int(value.get("indexed_documents", -1)) != document_count:
                raise RuntimeError("completed index metadata has an invalid document count")
            value["index"] = str(destination.resolve())
            return value
        active_encoder = encoder or E5Encoder(
            model_name,
            device=device,
            passage_max_length=passage_max_length,
        )
        dimension = int(active_encoder.dimension)
        if dimension % pq_m:
            raise ValueError(f"embedding dimension {dimension} is not divisible by pq_m={pq_m}")
        if min(document_count, train_size) < nlist:
            raise ValueError("selected training set must contain at least nlist passages")
        if min(document_count, train_size) < 2**pq_nbits:
            raise ValueError("training set is too small for pq_nbits")

        configuration = {
            "schema": INDEX_SCHEMA,
            "model": model_name,
            "dimension": dimension,
            "passage_max_length": passage_max_length,
            "metric": "inner_product",
            "nlist": nlist,
            "pq_m": pq_m,
            "pq_nbits": pq_nbits,
            "nprobe": nprobe,
            "document_store": str(docstore.resolve()),
            "source_revision": source_metadata.get("source_revision"),
            "source_documents": available_documents,
            "documents": document_count,
            "training_passages": min(document_count, train_size),
        }

        if building.is_file():
            if not building_metadata_path.is_file():
                raise RuntimeError(
                    f"partial index has no metadata: {building_metadata_path}; use --force to restart"
                )
            prior = json.loads(building_metadata_path.read_text(encoding="utf-8"))
            for key, expected in configuration.items():
                if prior.get(key) != expected:
                    raise RuntimeError(
                        f"partial index configuration mismatch for {key}: "
                        f"{prior.get(key)!r} != {expected!r}; use --force to restart"
                    )
            index = faiss.read_index(str(building))
            print(f"resuming compressed E5 index at {index.ntotal:,}/{document_count:,}", flush=True)
        else:
            texts = _training_texts(
                connection, document_count=document_count, train_size=train_size
            )
            print(f"encoding {len(texts):,} IVF-PQ training passages", flush=True)
            training_batches = []
            for start in range(0, len(texts), batch_size):
                training_batches.append(
                    active_encoder.encode_passages(texts[start : start + batch_size])
                )
                if start and start % (batch_size * 100) == 0:
                    print(f"training embeddings: {start:,}/{len(texts):,}", flush=True)
            training = np.ascontiguousarray(np.concatenate(training_batches), dtype="float32")
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFPQ(
                quantizer,
                dimension,
                nlist,
                pq_m,
                pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            print(
                f"training IVF{nlist},PQ{pq_m}x{pq_nbits} on {len(training):,} vectors",
                flush=True,
            )
            index.train(training)
            del training, training_batches, texts
            _write_index_atomic(faiss, index, building)
            _write_json_atomic(building_metadata_path, configuration)

        if not index.is_trained:
            raise RuntimeError("FAISS index is not trained")
        indexed = int(index.ntotal)
        if indexed > document_count:
            raise RuntimeError("partial index contains more vectors than the selected corpus")
        last_checkpoint = indexed
        for rows in _iter_batches(
            connection,
            start_rowid=indexed + 1,
            stop_rowid=document_count,
            batch_size=batch_size,
        ):
            expected_rowid = int(index.ntotal) + 1
            if rows[0][0] != expected_rowid:
                raise RuntimeError(
                    f"document order changed: expected rowid {expected_rowid}, found {rows[0][0]}"
                )
            vectors = np.ascontiguousarray(
                active_encoder.encode_passages([contents for _, contents in rows]),
                dtype="float32",
            )
            index.add(vectors)
            indexed = int(index.ntotal)
            if indexed - last_checkpoint >= checkpoint_every:
                _write_index_atomic(faiss, index, building)
                _write_json_atomic(building_metadata_path, configuration)
                last_checkpoint = indexed
                print(f"compressed E5 indexed: {indexed:,}/{document_count:,}", flush=True)

        _write_index_atomic(faiss, index, building)
        completed = {
            **configuration,
            "indexed_documents": int(index.ntotal),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(building_metadata_path, completed)
        os.replace(building, destination)
        os.replace(building_metadata_path, final_metadata_path)
        return {**completed, "index": str(destination.resolve())}
    finally:
        connection.close()


class CompressedE5Retriever:
    """Low-memory E5 IVF-PQ retriever backed by the existing SQLite docstore."""

    backend_name = "e5-ivfpq"

    def __init__(
        self,
        index_path: str | Path,
        docstore_path: str | Path | None = None,
        *,
        model_name: str | None = None,
        device: str = "auto",
        query_max_length: int = 256,
        nprobe: int | None = None,
        threads: int | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> None:
        faiss, np = _optional_imports()
        self._np = np
        self.index_path = Path(index_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        sidecar = metadata_path(self.index_path)
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        self._metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if self._metadata.get("schema") != INDEX_SCHEMA:
            raise ValueError(f"unsupported compressed E5 schema: {self._metadata.get('schema')!r}")
        self.docstore_path = Path(
            docstore_path or self._metadata.get("document_store", "")
        )
        if not self.docstore_path.is_file():
            raise FileNotFoundError(self.docstore_path)
        if threads is not None:
            if threads < 1:
                raise ValueError("threads must be positive")
            faiss.omp_set_num_threads(threads)
        self.index = faiss.read_index(str(self.index_path))
        configured_nprobe = int(nprobe or self._metadata.get("nprobe", 64))
        faiss.extract_index_ivf(self.index).nprobe = configured_nprobe
        self.nprobe = configured_nprobe
        self.document_count = int(self.index.ntotal)
        expected = int(self._metadata["indexed_documents"])
        if self.document_count != expected:
            raise ValueError(
                f"compressed index/metadata count mismatch: {self.document_count} != {expected}"
            )
        connection = sqlite3.connect(f"file:{self.docstore_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        try:
            available_documents = _validate_docstore(connection)
            source_metadata = _docstore_metadata(connection)
        finally:
            connection.close()
        if available_documents != int(self._metadata["source_documents"]):
            raise ValueError("document store count does not match the compressed index")
        indexed_revision = self._metadata.get("source_revision")
        current_revision = source_metadata.get("source_revision")
        if indexed_revision is not None and current_revision != indexed_revision:
            raise ValueError("document store revision does not match the compressed index")
        self.encoder = encoder or E5Encoder(
            model_name or str(self._metadata["model"]),
            device=device,
            passage_max_length=int(self._metadata["passage_max_length"]),
            query_max_length=query_max_length,
        )
        if int(self.encoder.dimension) != int(self._metadata["dimension"]):
            raise ValueError("query encoder dimension does not match compressed index")
        self._lock = threading.Lock()

    def metadata(self) -> dict[str, Any]:
        return {**self._metadata, "backend": self.backend_name, "nprobe": self.nprobe}

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not query.strip():
            return []
        with self._lock:
            vector = self._np.ascontiguousarray(
                self.encoder.encode_queries([query]), dtype="float32"
            )
            scores, indices = self.index.search(vector, min(top_k, self.document_count))
        ranked = [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if int(index) >= 0
        ]
        if not ranked:
            return []
        rowids = [index + 1 for index, _ in ranked]
        placeholders = ",".join("?" for _ in rowids)
        connection = sqlite3.connect(f"file:{self.docstore_path}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        try:
            rows = connection.execute(
                f"SELECT rowid, id, contents FROM passages WHERE rowid IN ({placeholders})",
                rowids,
            ).fetchall()
        finally:
            connection.close()
        documents = {
            int(rowid): (str(identifier), str(contents))
            for rowid, identifier, contents in rows
        }
        results = []
        for index, score in ranked:
            document = documents.get(index + 1)
            if document is not None:
                results.append(
                    RetrievedPassage(id=document[0], contents=document[1], score=score)
                )
        return results
