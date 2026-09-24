from threading import Thread

from mapd.retrieval.http_service import RetrievalHandler, RetrievalHTTPServer
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.retrieval.sqlite_fts import SQLiteFTSRetriever, build_sqlite_fts_index


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
