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
    normalized = str(value or "others").strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return TaskType(normalized)
    except ValueError:
        return TaskType.OTHERS


def parse_subtasks(raw: Any, round_index: int, prefix: str, limit: int) -> list[SubTask]:
    if not isinstance(raw, list):
        return []
    limited = [item for item in raw if isinstance(item, dict)][:limit]
    original_ids = [str(item.get("id", f"s{index + 1}")) for index, item in enumerate(limited)]
    id_map = {item_id: f"{prefix}-{item_id}" for item_id in original_ids}
    parsed = []
    for index, item in enumerate(limited):
        objective = str(item.get("objective") or item.get("query") or "").strip()
        if not objective:
            continue
        raw_dependencies = item.get("depends_on", [])
        dependencies = raw_dependencies if isinstance(raw_dependencies, list) else []
        parsed.append(
            SubTask(
                id=id_map[original_ids[index]],
                objective=objective,
                depends_on=[id_map.get(str(dep), str(dep)) for dep in dependencies],
                round_index=round_index,
            )
        )
    return parsed
