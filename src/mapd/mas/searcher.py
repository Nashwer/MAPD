from __future__ import annotations

from typing import Any

from mapd.mas.provider import TeacherClient
from mapd.mas.schema import SearchFinding, SearchRecord, SubTask
from mapd.protocol.leak_check import contains_answer
from mapd.retrieval.base import Retriever
from mapd.reward.exact_match import normalize_answer


class Searcher:
    def __init__(self, teacher: TeacherClient, retriever: Retriever, *, max_queries: int, top_k: int):
        self.teacher = teacher
        self.retriever = retriever
        self.max_queries = max_queries
        self.top_k = top_k

    def run(
        self,
        subtask: SubTask,
        findings: list[SearchFinding],
        forbidden_answers: list[str],
    ) -> tuple[list[SearchRecord], SearchFinding]:
        dependencies = [item for item in findings if item.subtask_id in subtask.depends_on]
        result = self.teacher.generate_json(
            "searcher",
            {
                "subtask": subtask.model_dump(mode="json"),
                "dependency_findings": [item.model_dump(mode="json") for item in dependencies],
                "max_queries": self.max_queries,
            },
        )
        query_value = result.get("queries", [])
        if not isinstance(query_value, list):
            raise TypeError("searcher queries must be a JSON array")
        raw_queries = [str(item).strip() for item in query_value]
        queries, seen = [], set()
        for query in raw_queries[: self.max_queries]:
            key = normalize_answer(query)
            if key and key not in seen and not contains_answer(query, forbidden_answers):
                queries.append(query)
                seen.add(key)
        if not queries:
            raise ValueError("searcher produced no usable query after leak filtering")

        searches = [
            SearchRecord(subtask_id=subtask.id, query=query, passages=self.retriever.search(query, self.top_k))
            for query in queries
        ]
        unique_passages: dict[str, dict[str, Any]] = {}
        for search in searches:
            for passage in search.passages:
                unique_passages.setdefault(passage.id, passage.model_dump(mode="json"))
        summary = self.teacher.generate_json(
            "search_summarizer",
            {"subtask": subtask.model_dump(mode="json"), "passages": list(unique_passages.values())},
        )
        evidence_value = summary.get("evidence_ids", [])
        if not isinstance(evidence_value, list):
            raise TypeError("search_summarizer evidence_ids must be a JSON array")
        summary_text = str(summary.get("summary", "")).strip()
        if not summary_text:
            raise ValueError("search_summarizer returned an empty summary")
        evidence_ids = [str(item) for item in evidence_value]
        unknown_ids = sorted(set(evidence_ids) - set(unique_passages))
        if unknown_ids:
            raise ValueError(
                "search_summarizer returned unknown evidence ids: " + ", ".join(unknown_ids)
            )
        if unique_passages and not evidence_ids:
            raise ValueError("search_summarizer must cite at least one retrieved passage")
        return searches, SearchFinding(
            subtask_id=subtask.id,
            summary=summary_text,
            evidence_ids=evidence_ids,
        )
