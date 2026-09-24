import json

import httpx
import pytest

from mapd.mas.orchestrator import _parse_task_type, parse_subtasks
from mapd.mas.provider import (
    OpenAICompatibleTeacher,
    _decode_json_object,
    _validate_role_response,
)
from mapd.protocol.schema import TaskType


def test_teacher_json_decoder_accepts_fenced_object():
    assert _decode_json_object('```json\n{"queries": ["tower"]}\n```') == {"queries": ["tower"]}


def test_orchestrator_normalizes_task_type_and_malformed_subtasks():
    assert _parse_task_type("multi-hop") is TaskType.MULTI_HOP
    with pytest.raises(ValueError, match="must be one of"):
        _parse_task_type("unexpected")
    with pytest.raises(TypeError, match="JSON array"):
        parse_subtasks("not-a-list", 1, "round-1", 4)


def test_teacher_role_contract_rejects_empty_finding_and_unknown_evidence():
    payload = {"passages": [{"id": "p1", "contents": "supported text"}]}
    with pytest.raises(TypeError, match="summary must be a non-empty string"):
        _validate_role_response(
            "search_summarizer", {"summary": "", "evidence_ids": ["p1"]}, payload
        )
    with pytest.raises(TypeError, match="unknown evidence_ids"):
        _validate_role_response(
            "search_summarizer",
            {"summary": "supported", "evidence_ids": ["p2"]},
            payload,
        )


def test_teacher_role_contract_rejects_wrapped_protocol():
    with pytest.raises(TypeError, match="fields mismatch"):
        _validate_role_response(
            "protocolizer",
            {"protocol": {"task_type": "single_hop"}, "type": "json_object"},
            {
                "success": True,
                "task_type": "single_hop",
                "candidate_answer": "Paris",
                "passages": [{"id": "p1", "contents": "Paris is in France."}],
            },
        )


def test_teacher_retries_a_semantically_invalid_role_response(monkeypatch):
    requests = []
    responses = iter(
        [
            '{"summary":"","evidence_ids":[]}',
            '{"summary":"supported text","evidence_ids":["p1"]}',
        ]
    )

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": next(responses)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    def fake_post(url, **kwargs):
        requests.append((url, json.loads(json.dumps(kwargs["json"]))))
        return Response()

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    monkeypatch.setattr("mapd.mas.provider.httpx.post", fake_post)
    monkeypatch.setattr("mapd.mas.provider.time.sleep", lambda _: None)
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=1,
    )

    result = teacher.generate_json(
        "search_summarizer",
        {"passages": [{"id": "p1", "contents": "supported text"}]},
    )

    assert result == {"summary": "supported text", "evidence_ids": ["p1"]}
    assert len(requests) == 2
    assert "previous response violated" in requests[1][1]["messages"][0]["content"]


def test_teacher_returns_last_parseable_invalid_protocol_to_quality_gate(monkeypatch):
    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"protocol":{"task_type":"single_hop"},"type":"json_object"}'
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    monkeypatch.setattr("mapd.mas.provider.httpx.post", lambda *_args, **_kwargs: Response())
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=0,
    )

    result = teacher.generate_json(
        "protocolizer",
        {
            "success": True,
            "task_type": "single_hop",
            "candidate_answer": "Paris",
            "passages": [{"id": "p1", "contents": "Paris is in France."}],
        },
    )

    assert result["type"] == "json_object"


def test_deepseek_teacher_disables_thinking_caps_output_and_tracks_budget(
    tmp_path, monkeypatch
):
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": '{"queries": ["tower"]}'}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    monkeypatch.setattr("mapd.mas.provider.httpx.post", fake_post)
    usage_log = tmp_path / "teacher_usage.jsonl"
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=0,
        thinking_mode="disabled",
        max_output_tokens=2048,
        budget_usd=0.000001,
        input_price_per_million=0.30,
        output_price_per_million=1.20,
        usage_log_path=usage_log,
    )

    assert teacher.generate_json("searcher", {"subtask": {}}) == {
        "queries": ["tower"]
    }
    body = requests[0][1]["json"]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 2048
    assert teacher.usage_summary()["estimated_cost_usd"] == pytest.approx(0.000054)
    record = json.loads(usage_log.read_text(encoding="utf-8"))
    assert record["role"] == "searcher"
    assert record["cumulative_cost_usd"] == pytest.approx(0.000054)
    with pytest.raises(RuntimeError, match="budget reached"):
        teacher.generate_json("searcher", {"subtask": {}})


def test_usage_summary_separates_shard_usage_from_cumulative_budget(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "estimated_cost_usd": 0.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=0,
        usage_log_path=tmp_path / "shard.jsonl",
        budget_ledger_path=ledger,
    )

    summary = teacher.usage_summary()

    assert summary["requests"] == 0
    assert summary["estimated_cost_usd"] == 0
    assert summary["cumulative_requests"] == 1
    assert summary["cumulative_estimated_cost_usd"] == pytest.approx(0.25)


def test_deepseek_teacher_reports_400_body_without_retrying_or_leaking_key(
    monkeypatch,
):
    requests = []

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return httpx.Response(
            400,
            json={"error": {"message": "context length exceeded"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "super-secret-key")
    monkeypatch.setattr("mapd.mas.provider.httpx.post", fake_post)
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=5,
    )

    with pytest.raises(RuntimeError) as exc_info:
        teacher.generate_json("answerer", {"question": "Where?"})

    message = str(exc_info.value)
    assert len(requests) == 1
    assert "role=answerer" in message
    assert "request_bytes=" in message
    assert "HTTP 400" in message
    assert "context length exceeded" in message
    assert "super-secret-key" not in message
    system_prompt = requests[0][1]["json"]["messages"][0]["content"]
    assert "JSON" in system_prompt


def test_deepseek_teacher_reports_truncated_response_without_retrying(monkeypatch):
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": '{"answer": "unfinished'},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2048},
            }

    def fake_post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    monkeypatch.setenv("TEST_DEEPSEEK_KEY", "secret")
    monkeypatch.setattr("mapd.mas.provider.httpx.post", fake_post)
    teacher = OpenAICompatibleTeacher(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="TEST_DEEPSEEK_KEY",
        timeout_seconds=10,
        max_retries=5,
        max_output_tokens=2048,
    )

    with pytest.raises(RuntimeError, match=r"role=protocolizer was truncated"):
        teacher.generate_json("protocolizer", {"passages": []})

    assert len(requests) == 1
