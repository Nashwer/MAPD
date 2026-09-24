from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
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
        "the plan must be prospective and contain no answer or hindsight. Use 2-6 concise reasoning steps. "
        "Include at most 5 grounding facts, each a continuous passage excerpt of at most 300 characters. "
        "Keep partial_findings under 500 characters and keep the answer as short as possible."
    ),
}


_RETRYABLE_HTTP_STATUS_CODES = {408, 409, 425, 429}
_JSON_RESPONSE_INSTRUCTION = "Return a valid JSON object only."


def _http_error_message(
    error: httpx.HTTPStatusError, *, role: str, request_bytes: int
) -> str:
    """Return useful request diagnostics without echoing prompts or credentials."""
    response_body = error.response.text.strip()
    if len(response_body) > 2_000:
        response_body = response_body[:2_000] + "... [truncated]"
    if not response_body:
        response_body = "<empty response>"
    return (
        f"teacher request role={role} request_bytes={request_bytes} failed with "
        f"HTTP {error.response.status_code}: {response_body}"
    )


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
        thinking_mode: str = "default",
        max_output_tokens: int | None = None,
        budget_usd: float | None = None,
        input_price_per_million: float | None = None,
        output_price_per_million: float | None = None,
        usage_log_path: str | Path | None = None,
        budget_ledger_path: str | Path | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.thinking_mode = thinking_mode
        self.max_output_tokens = max_output_tokens
        self.budget_usd = budget_usd
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.usage_log_path = Path(usage_log_path) if usage_log_path else None
        self.budget_ledger_path = (
            Path(budget_ledger_path) if budget_ledger_path else None
        )
        self._usage_lock = threading.Lock()
        self._budget_request_lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0
        self._estimated_cost_usd = 0.0
        self._load_existing_usage(
            self.budget_ledger_path or self.usage_log_path
        )

    def generate_json(self, role: str, payload: dict[str, Any]) -> dict[str, Any]:
        # A configured budget serializes teacher calls so parallel searchers
        # cannot all pass the same pre-request budget check concurrently.
        guard = (
            self._budget_request_lock
            if self.budget_usd is not None
            else nullcontext()
        )
        with guard:
            return self._generate_json_request(role, payload)

    def _generate_json_request(
        self, role: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        self._check_budget()
        user_content = json.dumps(payload, ensure_ascii=False)
        request_bytes = len(user_content.encode("utf-8"))
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"{ROLE_INSTRUCTIONS[role]} {_JSON_RESPONSE_INSTRUCTION}"
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.max_output_tokens is not None:
            body["max_tokens"] = self.max_output_tokens
        if self.thinking_mode == "disabled":
            body["thinking"] = {"type": "disabled"}
        elif self.thinking_mode in {"low", "high", "max"}:
            body["thinking"] = {"type": "enabled"}
            body["reasoning_effort"] = self.thinking_mode
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
                response_payload = response.json()
                choice = response_payload["choices"][0]
                content = choice["message"]["content"]
                self._record_usage(role, response_payload.get("usage"))
                if choice.get("finish_reason") == "length":
                    raise RuntimeError(
                        f"teacher response role={role} was truncated "
                        f"(finish_reason=length, content_chars={len(content)}, "
                        f"max_tokens={body.get('max_tokens')}); increase the output "
                        "limit or constrain the role response"
                    )
                return _decode_json_object(content)
            except httpx.HTTPStatusError as exc:
                error = RuntimeError(
                    _http_error_message(
                        exc, role=role, request_bytes=request_bytes
                    )
                )
                status_code = exc.response.status_code
                retryable = (
                    status_code in _RETRYABLE_HTTP_STATUS_CODES
                    or status_code >= 500
                )
                if not retryable:
                    break
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
            except (httpx.HTTPError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Teacher request failed after retries: {error}") from error

    def usage_summary(self) -> dict[str, int | float | None]:
        with self._usage_lock:
            return {
                "requests": self._request_count,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "estimated_cost_usd": round(self._estimated_cost_usd, 8),
                "budget_usd": self.budget_usd,
            }

    def _check_budget(self) -> None:
        if self.budget_usd is None:
            return
        with self._usage_lock:
            spent = self._estimated_cost_usd
        if spent >= self.budget_usd:
            raise RuntimeError(
                f"teacher API budget reached: ${spent:.4f} >= ${self.budget_usd:.4f}; "
                "completed artifacts and usage log are safe to resume"
            )

    def _record_usage(self, role: str, raw_usage: Any) -> None:
        if not isinstance(raw_usage, dict):
            if self.budget_usd is not None:
                raise RuntimeError(
                    "teacher response omitted token usage; stopping because a budget is configured"
                )
            return
        prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
        completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        with self._usage_lock:
            self._request_count += 1
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._estimated_cost_usd += cost
            cumulative = self._estimated_cost_usd
            record = {
                "time": datetime.now(timezone.utc).isoformat(),
                "role": role,
                "model": self.model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated_cost_usd": round(cost, 8),
                "cumulative_cost_usd": round(cumulative, 8),
            }
            encoded = json.dumps(record, ensure_ascii=False) + "\n"
            paths = {
                path
                for path in (self.usage_log_path, self.budget_ledger_path)
                if path is not None
            }
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_price = self.input_price_per_million or 0.0
        output_price = self.output_price_per_million or 0.0
        # Charge every input token at the cache-miss price. This deliberately
        # overestimates DeepSeek automatic cache hits and keeps the guard safe.
        return (
            prompt_tokens * input_price + completion_tokens * output_price
        ) / 1_000_000

    def _load_existing_usage(self, path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._request_count += 1
                    self._prompt_tokens += int(record.get("prompt_tokens") or 0)
                    self._completion_tokens += int(record.get("completion_tokens") or 0)
                    self._estimated_cost_usd += float(record.get("estimated_cost_usd") or 0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
