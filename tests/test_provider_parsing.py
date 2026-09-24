import json

import httpx
import pytest

from mapd.mas.orchestrator import _parse_task_type, parse_subtasks
from mapd.mas.provider import OpenAICompatibleTeacher, _decode_json_object
from mapd.protocol.schema import TaskType


def test_teacher_json_decoder_accepts_fenced_object():
    assert _decode_json_object('```json\n{"queries": ["tower"]}\n```') == {"queries": ["tower"]}


def test_orchestrator_normalizes_task_type_and_malformed_subtasks():
    assert _parse_task_type("multi-hop") is TaskType.MULTI_HOP
    assert _parse_task_type("unexpected") is TaskType.OTHERS
    assert parse_subtasks("not-a-list", 1, "round-1", 4) == []


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
