import io
import json
import tarfile
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from mapd.retrieval.http_service import RetrievalHandler, RetrievalHTTPServer
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.retrieval.sqlite_fts import (
    SQLiteFTSRetriever,
    _fts_expression,
    _query_tokens,
    build_sqlite_fts_index,
)
from mapd.retrieval.e5_ivfpq import CompressedE5Retriever, build_e5_ivfpq_index


class _FixtureEncoder:
    dimension = 8

    def encode_queries(self, texts):
        return self._encode(texts)

    def encode_passages(self, texts):
        return self._encode(texts)

    @staticmethod
    def _encode(texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            vector = np.full(8, 0.0001, dtype="float32")
            if "winter guest" in lowered or "directed" in lowered:
                vector[0] += 1.0
            if "eiffel" in lowered or "paris" in lowered:
                vector[1] += 1.0
            if "die hard" in lowered or "hans gruber" in lowered:
                vector[2] += 1.0
            if not np.any(vector[:3] > 0.5):
                fixture_number = int(lowered.rsplit(" ", 1)[-1].rstrip("."))
                vector[3 + fixture_number % 5] += 1.0
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors).astype("float32")


class _FailingFixtureEncoder(_FixtureEncoder):
    def __init__(self, fail_after_passage_calls):
        self.fail_after_passage_calls = fail_after_passage_calls
        self.passage_calls = 0

    def encode_passages(self, texts):
        self.passage_calls += 1
        if self.passage_calls > self.fail_after_passage_calls:
            raise RuntimeError("simulated indexing interruption")
        return super().encode_passages(texts)


def _compressed_fixture(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    records = [
        {"id": "p1", "contents": "The Eiffel Tower is in Paris."},
        {"id": "p2", "contents": "The Winter Guest was directed by Alan Rickman."},
        {"id": "p3", "contents": "Alan Rickman played Hans Gruber in Die Hard."},
    ]
    records.extend(
        {"id": f"noise-{index}", "contents": f"Unrelated fixture document {index}."}
        for index in range(253)
    )
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    docstore = tmp_path / "wiki.sqlite3"
    build_sqlite_fts_index(corpus, docstore)
    index = tmp_path / "e5.faiss"
    return docstore, index


def test_bm25_returns_relevant_passage():
    retriever = BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl")
    results = retriever.search("Who directed The Winter Guest?", top_k=2)
    assert results[0].id == "p2"
    assert results[0].score > 0


def test_persistent_sqlite_retriever(tmp_path):
    index = tmp_path / "wiki.sqlite3"
    metadata = build_sqlite_fts_index(
        "tests/fixtures/corpus.jsonl", index, batch_size=1, source_revision="fixture-v1"
    )
    retriever = SQLiteFTSRetriever(index)
    results = retriever.search("Who directed The Winter Guest?", top_k=3)
    assert metadata["documents"] == 3
    assert metadata["source_revision"] == "fixture-v1"
    assert retriever.metadata()["schema"] == "mapd-wiki18-sqlite-fts-v1"
    assert results[0].id == "p2"


def test_http_service_uses_search_r1_contract(tmp_path):
    index = tmp_path / "wiki.sqlite3"
    build_sqlite_fts_index("tests/fixtures/corpus.jsonl", index)
    server = RetrievalHTTPServer(("127.0.0.1", 0), RetrievalHandler)
    server.retriever = SQLiteFTSRetriever(index)
    server.default_top_k = 3
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        results = HTTPRetriever(f"http://{host}:{port}/retrieve").search(
            "Who directed The Winter Guest?", 3
        )
        assert results[0].id == "p2"
        assert results[0].score > 0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_compressed_e5_index_build_reuse_and_search(tmp_path):
    docstore, index = _compressed_fixture(tmp_path)
    encoder = _FixtureEncoder()
    result = build_e5_ivfpq_index(
        docstore,
        index,
        model_name="fixture-e5",
        batch_size=8,
        nlist=2,
        pq_m=2,
        pq_nbits=3,
        train_size=256,
        nprobe=2,
        checkpoint_every=64,
        threads=1,
        encoder=encoder,
    )
    assert result["indexed_documents"] == 256
    assert result["schema"] == "mapd-wiki18-e5-ivfpq-v1"
    reused = build_e5_ivfpq_index(
        docstore,
        index,
        model_name="fixture-e5",
        batch_size=8,
        nlist=2,
        pq_m=2,
        pq_nbits=3,
        train_size=256,
        nprobe=2,
        checkpoint_every=64,
        threads=1,
        encoder=encoder,
    )
    assert reused["indexed_documents"] == 256

    retriever = CompressedE5Retriever(
        index, docstore, encoder=encoder, nprobe=2, threads=1
    )
    results = retriever.search("Who directed The Winter Guest?", 3)
    assert "p2" in {result.id for result in results}
    assert [result.score for result in results] == sorted(
        [result.score for result in results], reverse=True
    )
    assert retriever.metadata()["backend"] == "e5-ivfpq"


def test_compressed_e5_index_resumes_from_last_checkpoint(tmp_path):
    docstore, index = _compressed_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="simulated indexing interruption"):
        build_e5_ivfpq_index(
            docstore,
            index,
            model_name="fixture-e5",
            batch_size=32,
            nlist=2,
            pq_m=2,
            pq_nbits=3,
            train_size=256,
            nprobe=2,
            checkpoint_every=64,
            threads=1,
            encoder=_FailingFixtureEncoder(fail_after_passage_calls=10),
        )
    partial = index.with_name(index.name + ".building")
    assert partial.is_file()

    result = build_e5_ivfpq_index(
        docstore,
        index,
        model_name="fixture-e5",
        batch_size=32,
        nlist=2,
        pq_m=2,
        pq_nbits=3,
        train_size=256,
        nprobe=2,
        checkpoint_every=64,
        threads=1,
        encoder=_FixtureEncoder(),
    )
    assert result["indexed_documents"] == 256
    assert not partial.exists()


def test_http_service_accepts_compressed_e5_backend(tmp_path):
    docstore, index = _compressed_fixture(tmp_path)
    encoder = _FixtureEncoder()
    build_e5_ivfpq_index(
        docstore,
        index,
        model_name="fixture-e5",
        batch_size=8,
        nlist=2,
        pq_m=2,
        pq_nbits=3,
        train_size=256,
        nprobe=2,
        checkpoint_every=64,
        threads=1,
        encoder=encoder,
    )
    server = RetrievalHTTPServer(("127.0.0.1", 0), RetrievalHandler)
    server.retriever = CompressedE5Retriever(
        index, docstore, encoder=encoder, nprobe=2, threads=1
    )
    server.default_top_k = 3
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        results = HTTPRetriever(f"http://{host}:{port}/retrieve").search(
            "Who directed The Winter Guest?", 3
        )
        assert "p2" in {result.id for result in results}
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_index_builder_streams_mislabeled_wiki18_tar_gzip(tmp_path):
    corpus = tmp_path / "wiki-18.jsonl.gz"
    payload = Path("tests/fixtures/corpus.jsonl").read_bytes()
    with tarfile.open(corpus, mode="w:gz") as archive:
        member = tarfile.TarInfo("data00/source/wiki_dump.jsonl")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    index = tmp_path / "wiki.sqlite3"
    result = build_sqlite_fts_index(corpus, index)
    assert result["documents"] == 3
    assert SQLiteFTSRetriever(index).search("Eiffel Tower", 1)[0].id == "p1"


def test_full_corpus_query_plan_drops_high_frequency_question_words():
    tokens = _query_tokens("Who directed the 1997 film The Winter Guest?")
    assert tokens == ["directed", "1997", "film", "winter", "guest"]
    assert _fts_expression(tokens, operator="AND") == (
        '"directed" AND "1997" AND "film" AND "winter" AND "guest"'
    )


def test_sqlite_retriever_rejects_nonpositive_query_timeout(tmp_path):
    index = tmp_path / "wiki.sqlite3"
    build_sqlite_fts_index("tests/fixtures/corpus.jsonl", index)
    try:
        SQLiteFTSRetriever(index, query_timeout_seconds=0)
    except ValueError as exc:
        assert "query_timeout_seconds" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("nonpositive timeout should be rejected")
