from mapd.mas.orchestrator import _parse_task_type, parse_subtasks
from mapd.mas.provider import _decode_json_object
from mapd.protocol.schema import TaskType


def test_teacher_json_decoder_accepts_fenced_object():
    assert _decode_json_object('```json\n{"queries": ["tower"]}\n```') == {"queries": ["tower"]}


def test_orchestrator_normalizes_task_type_and_malformed_subtasks():
    assert _parse_task_type("multi-hop") is TaskType.MULTI_HOP
    assert _parse_task_type("unexpected") is TaskType.OTHERS
    assert parse_subtasks("not-a-list", 1, "round-1", 4) == []
