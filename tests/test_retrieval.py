import io
import tarfile
from pathlib import Path
from threading import Thread

from mapd.retrieval.http_service import RetrievalHandler, RetrievalHTTPServer
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.retrieval.sqlite_fts import (
    SQLiteFTSRetriever,
    _fts_expression,
    _query_tokens,
    build_sqlite_fts_index,
)


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
