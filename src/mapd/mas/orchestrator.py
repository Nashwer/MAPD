from __future__ import annotations

from typing import Any

from mapd.mas.provider import TeacherClient
from mapd.mas.schema import SearchFinding, SubTask
from mapd.protocol.schema import TaskType


class Orchestrator:
    def __init__(self, teacher: TeacherClient, max_subquestions: int):
        self.teacher = teacher
        self.max_subquestions = max_subquestions

    def plan(
        self, question: str, round_index: int, previous_findings: list[SearchFinding]
    ) -> tuple[TaskType | None, list[SubTask]]:
        result = self.teacher.generate_json(
            "orchestrator",
            {
                "question": question,
                "round": round_index,
                "max_subquestions": self.max_subquestions,
                "previous_findings": [item.model_dump(mode="json") for item in previous_findings],
            },
        )
        task_type = _parse_task_type(result.get("task_type")) if round_index == 1 else None
        return task_type, parse_subtasks(
            result.get("subtasks", []),
            round_index,
            f"round-{round_index}",
            self.max_subquestions,
        )


def _parse_task_type(value: Any) -> TaskType:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("orchestrator task_type must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return TaskType(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in TaskType)
        raise ValueError(f"orchestrator task_type must be one of: {allowed}") from exc


def parse_subtasks(raw: Any, round_index: int, prefix: str, limit: int) -> list[SubTask]:
    if not isinstance(raw, list):
        raise TypeError("subtasks must be a JSON array")
    if len(raw) > limit:
        raise ValueError(f"subtasks exceeds configured maximum of {limit}")
    if any(not isinstance(item, dict) for item in raw):
        raise TypeError("each subtask must be a JSON object")
    limited = raw
    original_ids = [str(item.get("id", f"s{index + 1}")) for index, item in enumerate(limited)]
    if len(original_ids) != len(set(original_ids)):
        raise ValueError("subtask ids must be unique")
    id_map = {item_id: f"{prefix}-{item_id}" for item_id in original_ids}
    parsed = []
    for index, item in enumerate(limited):
        objective = str(item.get("objective") or item.get("query") or "").strip()
        if not objective:
            raise ValueError("each subtask requires a non-empty objective")
        raw_dependencies = item.get("depends_on", [])
        if not isinstance(raw_dependencies, list):
            raise TypeError("subtask depends_on must be a JSON array")
        dependencies = raw_dependencies
        parsed.append(
            SubTask(
                id=id_map[original_ids[index]],
                objective=objective,
                depends_on=[id_map.get(str(dep), str(dep)) for dep in dependencies],
                round_index=round_index,
            )
        )
    return parsed
