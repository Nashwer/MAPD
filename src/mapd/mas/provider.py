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
        "Classify and decompose the question into dependency-aware subtasks. "
        "Use task_type=single_hop for one direct fact, multi_hop for linked facts, comparison for "
        "explicit comparisons, and others only when none applies. Do not reveal the answer. "
        "Return exactly {\"task_type\":\"single_hop|multi_hop|comparison|others\","
        "\"subtasks\":[{\"id\":\"s1\",\"objective\":\"...\",\"depends_on\":[]}]}."
    ),
    "searcher": (
        "Produce up to max_queries retrieval queries for the objective. Do not include an oracle answer. "
        "When retry_instruction is present, follow it and substantially rewrite the rejected candidates. "
        "Return exactly {\"queries\":[\"query\"]}; queries must be a non-empty string array."
    ),
    "search_summarizer": (
        "Compress the retrieved passages into one concise, non-empty supported finding. Cite only ids "
        "present in passages. Return exactly {\"summary\":\"...\",\"evidence_ids\":[\"id\"]}. "
        "If passages is non-empty, evidence_ids must be non-empty."
    ),
    "answerer": (
        "Infer the shortest answer strictly from findings and passages. You do not receive ground truth. "
        "Use null when evidence is insufficient. Return exactly {\"answer\":\"short answer\"} or "
        "{\"answer\":null}."
    ),
    "repair": (
        "Use ground truth only to diagnose expression versus search failure. Do not put the answer in "
        "objectives. Return exactly {\"failure_type\":\"expression|search\",\"diagnosis\":\"...\","
        "\"subtasks\":[{\"id\":\"s1\",\"objective\":\"...\",\"depends_on\":[]}]}."
    ),
    "protocolizer": (
        "Convert the exploration log into the paper's Structured JSON Protocol. Output the protocol "
        "object itself, with no outer protocol/data/result/schema/type wrapper and no extra fields. "
        "task_type must be one of single_hop, multi_hop, comparison, others. reasoning_plan must be an "
        "ordered array of actionable sub-goal strings with no final outcome, answer, or hindsight. "
        "grounding_facts must be an array of strings copied verbatim from retrieved passages. "
        "Copy the input task_type exactly. For a succeeded protocol, copy candidate_answer exactly. "
        "The example strings below show the required JSON types and must be replaced with input-derived values. "
        "If success=true, output exactly: "
        '{"task_type":"multi_hop","reasoning_plan":["Search for the first entity.","Use it to find the requested fact."],'
        '"grounding_facts":["A verbatim retrieved passage substring."],"answer":"short verified answer",'
        '"answer_grounded":true}. Do not output partial_findings. '
        "If success=false, output exactly: "
        '{"task_type":"multi_hop","reasoning_plan":["Search for the first entity.","Find the missing relation."],'
        '"grounding_facts":["A verbatim retrieved passage substring."],'
        '"partial_findings":"Confirmed the supported facts, but reliable evidence for the missing fact is absent.",'
        '"answer_grounded":false}. Do not output answer.'
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


_TASK_TYPES = {"single_hop", "multi_hop", "comparison", "others"}


def _validate_role_response(
    role: str, value: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Validate the JSON contract before role code can silently default fields."""

    if role == "orchestrator":
        _require_exact_fields(role, value, {"task_type", "subtasks"})
        if value["task_type"] not in _TASK_TYPES:
            raise TypeError(f"orchestrator task_type must be one of {sorted(_TASK_TYPES)}")
        _validate_subtasks(value["subtasks"], int(payload.get("max_subquestions") or 0))
    elif role == "searcher":
        _require_exact_fields(role, value, {"queries"})
        queries = _require_string_list(role, "queries", value["queries"], allow_empty=False)
        maximum = int(payload.get("max_queries") or len(queries))
        if len(queries) > maximum:
            raise TypeError(f"searcher queries exceeds max_queries={maximum}")
    elif role == "search_summarizer":
        _require_exact_fields(role, value, {"summary", "evidence_ids"})
        _require_non_empty_string(role, "summary", value["summary"])
        evidence_ids = _require_string_list(
            role, "evidence_ids", value["evidence_ids"], allow_empty=True
        )
        passages = payload.get("passages")
        if not isinstance(passages, list):
            raise TypeError("search_summarizer input passages must be an array")
        allowed_ids = {
            str(item.get("id")) for item in passages if isinstance(item, dict) and item.get("id") is not None
        }
        unknown = sorted(set(evidence_ids) - allowed_ids)
        if unknown:
            raise TypeError(f"search_summarizer cited unknown evidence_ids: {unknown}")
        if passages and not evidence_ids:
            raise TypeError("search_summarizer must cite evidence when passages are available")
    elif role == "answerer":
        _require_exact_fields(role, value, {"answer"})
        if value["answer"] is not None:
            _require_non_empty_string(role, "answer", value["answer"])
    elif role == "repair":
        _require_exact_fields(role, value, {"failure_type", "diagnosis", "subtasks"})
        if value["failure_type"] not in {"expression", "search"}:
            raise TypeError("repair failure_type must be expression or search")
        _require_non_empty_string(role, "diagnosis", value["diagnosis"])
        _validate_subtasks(value["subtasks"], int(payload.get("max_subquestions") or 0))
    elif role == "protocolizer":
        _validate_protocolizer_response(value, payload)
    else:
        raise TypeError(f"unsupported teacher role: {role}")
    return value


def _require_exact_fields(role: str, value: dict[str, Any], expected: set[str]) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TypeError(f"{role} response fields mismatch: missing={missing}, extra={extra}")


def _require_non_empty_string(role: str, field: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{role} {field} must be a non-empty string")
    return value


def _require_string_list(
    role: str, field: str, value: Any, *, allow_empty: bool
) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{role} {field} must be an array")
    if not allow_empty and not value:
        raise TypeError(f"{role} {field} must be non-empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise TypeError(f"{role} {field} must contain only non-empty strings")
    return value


def _validate_subtasks(value: Any, maximum: int) -> None:
    if not isinstance(value, list):
        raise TypeError("subtasks must be an array")
    if maximum and len(value) > maximum:
        raise TypeError(f"subtasks exceeds max_subquestions={maximum}")
    ids: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each subtask must be an object")
        _require_exact_fields("subtask", item, {"id", "objective", "depends_on"})
        ids.append(_require_non_empty_string("subtask", "id", item["id"]))
        _require_non_empty_string("subtask", "objective", item["objective"])
        _require_string_list("subtask", "depends_on", item["depends_on"], allow_empty=True)
    if len(ids) != len(set(ids)):
        raise TypeError("subtask ids must be unique")
    known = set(ids)
    for item in value:
        unknown = sorted(set(item["depends_on"]) - known)
        if unknown:
            raise TypeError(f"subtask depends_on contains unknown ids: {unknown}")
        if item["id"] in item["depends_on"]:
            raise TypeError("subtask cannot depend on itself")


def _validate_protocolizer_response(value: dict[str, Any], payload: dict[str, Any]) -> None:
    success = payload.get("success") is True
    common = {"task_type", "reasoning_plan", "grounding_facts", "answer_grounded"}
    expected = common | ({"answer"} if success else {"partial_findings"})
    _require_exact_fields("protocolizer", value, expected)
    if value["task_type"] != payload.get("task_type"):
        raise TypeError("protocolizer task_type must exactly match the exploration")
    _require_string_list(
        "protocolizer", "reasoning_plan", value["reasoning_plan"], allow_empty=False
    )
    facts = _require_string_list(
        "protocolizer", "grounding_facts", value["grounding_facts"], allow_empty=False
    )
    if type(value["answer_grounded"]) is not bool or value["answer_grounded"] is not success:
        raise TypeError("protocolizer answer_grounded must exactly match exploration success")
    if success:
        _require_non_empty_string("protocolizer", "answer", value["answer"])
        if value["answer"] != payload.get("candidate_answer"):
            raise TypeError("protocolizer answer must exactly match candidate_answer")
    else:
        _require_non_empty_string("protocolizer", "partial_findings", value["partial_findings"])
    passages = payload.get("passages")
    if not isinstance(passages, list):
        raise TypeError("protocolizer input passages must be an array")
    passage_texts = [
        str(item.get("contents") or "") for item in passages if isinstance(item, dict)
    ]
    if any(not any(fact in passage for passage in passage_texts) for fact in facts):
        raise TypeError("protocolizer grounding_facts must be verbatim passage substrings")


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
            protocol = {
                "task_type": payload["task_type"],
                "reasoning_plan": [
                    "Search for evidence relevant to each entity in the question.",
                    "Cross-check the retrieved evidence and derive the requested answer.",
                ],
                "grounding_facts": [item["contents"] for item in payload["passages"][:3]],
                "answer_grounded": success,
            }
            if success:
                protocol["answer"] = payload["candidate_answer"]
            else:
                protocol["partial_findings"] = (
                    "Relevant evidence was retrieved, but the answer was not verified."
                )
            return protocol
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
        self._run_prompt_tokens = 0
        self._run_completion_tokens = 0
        self._run_request_count = 0
        self._run_estimated_cost_usd = 0.0
        self._load_existing_usage(
            self.budget_ledger_path or self.usage_log_path, cumulative=True
        )
        self._load_existing_usage(self.usage_log_path, cumulative=False)

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
        last_decoded: dict[str, Any] | None = None
        base_system_prompt = body["messages"][0]["content"]
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
                decoded = _decode_json_object(content)
                last_decoded = decoded
                return _validate_role_response(role, decoded, payload)
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
                if (
                    role == "protocolizer"
                    and attempt == self.max_retries
                    and last_decoded is not None
                ):
                    # Protocol schema failures belong to the paper's quality
                    # gate. Preserve the last parseable object so one bad
                    # sample is rejected and checkpointed instead of stopping
                    # the entire resumable shard.
                    return last_decoded
                if attempt < self.max_retries:
                    correction = str(exc).replace("\n", " ")[:500]
                    body["messages"][0]["content"] = (
                        base_system_prompt
                        + " Your previous response violated this contract: "
                        + correction
                        + ". Return a corrected JSON object only."
                    )
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Teacher request failed after retries: {error}") from error

    def usage_summary(self) -> dict[str, int | float | None]:
        with self._usage_lock:
            return {
                "requests": self._run_request_count,
                "prompt_tokens": self._run_prompt_tokens,
                "completion_tokens": self._run_completion_tokens,
                "estimated_cost_usd": round(self._run_estimated_cost_usd, 8),
                "budget_usd": self.budget_usd,
                "cumulative_requests": self._request_count,
                "cumulative_prompt_tokens": self._prompt_tokens,
                "cumulative_completion_tokens": self._completion_tokens,
                "cumulative_estimated_cost_usd": round(self._estimated_cost_usd, 8),
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
            self._run_request_count += 1
            self._run_prompt_tokens += prompt_tokens
            self._run_completion_tokens += completion_tokens
            self._run_estimated_cost_usd += cost
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

    def _load_existing_usage(self, path: Path | None, *, cumulative: bool) -> None:
        if path is None or not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    prompt_tokens = int(record.get("prompt_tokens") or 0)
                    completion_tokens = int(record.get("completion_tokens") or 0)
                    cost = float(record.get("estimated_cost_usd") or 0)
                    if cumulative:
                        self._request_count += 1
                        self._prompt_tokens += prompt_tokens
                        self._completion_tokens += completion_tokens
                        self._estimated_cost_usd += cost
                    else:
                        self._run_request_count += 1
                        self._run_prompt_tokens += prompt_tokens
                        self._run_completion_tokens += completion_tokens
                        self._run_estimated_cost_usd += cost
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
