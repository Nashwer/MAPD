import pytest

from mapd.mas.schema import SearchFinding, SubTask
from mapd.mas.searcher import QUERY_GENERATION_ATTEMPTS, Searcher
from mapd.retrieval.schema import RetrievedPassage


class RecordingRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        self.queries.append(query)
        return [RetrievedPassage(id="p1", contents="A supported passage.", score=1.0)]


class RetryTeacher:
    model = "retry-teacher"

    def __init__(self, searcher_responses: list[dict]):
        self.searcher_responses = iter(searcher_responses)
        self.searcher_payloads: list[dict] = []

    def generate_json(self, role: str, payload: dict) -> dict:
        if role == "searcher":
            self.searcher_payloads.append(payload)
            return next(self.searcher_responses)
        if role == "search_summarizer":
            return {"summary": "Supported finding.", "evidence_ids": ["p1"]}
        raise AssertionError(f"unexpected role: {role}")


def _run_search(teacher: RetryTeacher, objective: str = "Find the requested occupation"):
    retriever = RecordingRetriever()
    searcher = Searcher(teacher, retriever, max_queries=3, top_k=1)
    result = searcher.run(
        SubTask(id="s1", objective=objective),
        [SearchFinding(subtask_id="dependency", summary="Known relation", evidence_ids=[])],
        ["athletics director"],
    )
    return retriever, result


def test_searcher_rewrites_queries_rejected_by_leak_guard():
    teacher = RetryTeacher(
        [
            {"queries": ["Was the answer athletics director?"]},
            {"queries": ["Ed Olle university position"]},
        ]
    )

    retriever, (searches, finding) = _run_search(teacher)

    assert retriever.queries == ["Ed Olle university position"]
    assert [item.query for item in searches] == retriever.queries
    assert finding.summary == "Supported finding."
    assert len(teacher.searcher_payloads) == 2
    assert "retry_instruction" not in teacher.searcher_payloads[0]
    assert "retry_instruction" in teacher.searcher_payloads[1]
    assert "athletics director" not in str(teacher.searcher_payloads).lower()


def test_searcher_falls_back_to_safe_orchestrator_objective():
    teacher = RetryTeacher(
        [{"queries": ["athletics director"]}] * QUERY_GENERATION_ATTEMPTS
    )

    retriever, (searches, _) = _run_search(teacher)

    assert len(teacher.searcher_payloads) == QUERY_GENERATION_ATTEMPTS
    assert retriever.queries == ["Find the requested occupation"]
    assert searches[0].query == "Find the requested occupation"


def test_searcher_still_fails_when_objective_leaks_oracle_answer():
    teacher = RetryTeacher(
        [{"queries": ["athletics director"]}] * QUERY_GENERATION_ATTEMPTS
    )

    with pytest.raises(ValueError, match="original objective also failed"):
        _run_search(teacher, objective="Find why the answer is athletics director")
