from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Protocol

import httpx


class TeacherClient(Protocol):
    model: str

    def generate_json(self, role: str, payload: dict[str, Any]) -> dict[str, Any]: ...


ROLE_INSTRUCTIONS = {
    "orchestrator": (
        "Classify and decompose into dependency-aware subtasks with id, objective, and depends_on. "
        "Do not reveal the answer. Return JSON only."
    ),
    "searcher": (
        "Produce up to max_queries retrieval queries for the objective. Do not include an oracle answer. "
        "Return JSON with queries."
    ),
    "search_summarizer": (
        "Compress passages into a supported finding and passage evidence_ids. Return JSON only."
    ),
    "answerer": (
        "Infer the shortest answer strictly from findings and passages. You do not receive ground truth. "
        "Use null when evidence is insufficient."
    ),
    "repair": (
        "Use ground truth only to diagnose expression versus search failure. Return failure_type, diagnosis, "
        "and dependency-aware subtasks without putting the answer in objectives or queries."
    ),
    "protocolizer": (
        "Create a style-normalized MAPD protocol. Grounding facts must be verbatim passage substrings; "
        "the plan must be prospective and contain no answer or hindsight."
    ),
}


def _decode_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise TypeError("teacher response content must be a JSON object or string")
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError("teacher response JSON must be an object")
    return value


class MockTeacher:
    model = "mock-mapd-teacher"

    def generate_json(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        if role == "orchestrator":
            question = payload["question"]
            lowered = question.lower()
            task_type = "multi_hop" if any(term in lowered for term in ("which actor", "also", "both")) else "single_hop"
            return {"task_type": task_type, "subtasks": [{"id": "s1", "objective": question, "depends_on": []}]}
        if role == "searcher":
            return {"queries": [payload["subtask"]["objective"]]}
        if role == "search_summarizer":
            passages = payload["passages"]
            return {
                "summary": " ".join(item["contents"] for item in passages),
                "evidence_ids": [item["id"] for item in passages],
            }
        if role == "answerer":
            evidence = "\n".join(item["contents"] for item in payload["passages"])
            question = payload["question"].lower()
            patterns = []
            if "eiffel tower" in question:
                patterns.append(r"(?:located in|tower in) ([A-Z][A-Za-z .'-]+?)(?:[,.]|$)")
            if "winter guest" in question:
                patterns.append(r"directed by ([A-Z][a-z]+ [A-Z][a-z]+)")
            for pattern in patterns:
                if match := re.search(pattern, evidence):
                    return {"answer": match.group(1).strip()}
            return {"answer": None}
        if role == "repair":
            return {"failure_type": "search", "diagnosis": "No additional mock query available.", "subtasks": []}
        if role == "protocolizer":
            success = payload["success"]
            return {
                "task_type": payload["task_type"],
                "reasoning_plan": [
                    "Search for evidence relevant to each entity in the question.",
                    "Cross-check the retrieved evidence and derive the requested answer.",
                ],
                "grounding_facts": [item["contents"] for item in payload["passages"][:3]],
                "partial_findings": None if success else "Relevant evidence was retrieved, but the answer was not verified.",
                "answer": payload["candidate_answer"] if success else None,
                "answer_grounded": success,
            }
        raise ValueError(f"Unsupported mock teacher role: {role}")


class OpenAICompatibleTeacher:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float,
        max_retries: int,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate_json(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ROLE_INSTRUCTIONS[role]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return _decode_json_object(content)
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Teacher request failed after retries: {error}") from error
