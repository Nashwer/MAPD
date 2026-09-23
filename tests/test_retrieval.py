from mapd.retrieval.wiki18 import BM25Retriever


def test_bm25_returns_relevant_passage():
    retriever = BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl")
    results = retriever.search("Who directed The Winter Guest?", top_k=2)
    assert results[0].id == "p2"
    assert results[0].score > 0
