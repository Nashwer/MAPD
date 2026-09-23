from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from mapd.mas.provider import TeacherClient
from mapd.mas.schema import ExplorationLog
from mapd.protocol.schema import StructuredProtocol


class Protocolizer:
    def __init__(self, teacher: TeacherClient):
        self.teacher = teacher

    def generate(self, exploration: ExplorationLog) -> tuple[StructuredProtocol | None, dict[str, Any]]:
        unique_passages: dict[str, dict[str, Any]] = {}
        for search in exploration.searches:
            for passage in search.passages:
                unique_passages.setdefault(passage.id, passage.model_dump(mode="json"))
        raw = self.teacher.generate_json(
            "protocolizer",
            {
                "question": exploration.question,
                "task_type": exploration.task_type.value,
                "success": exploration.success,
                "candidate_answer": exploration.candidate_answer if exploration.success else None,
                "subtasks": [item.model_dump(mode="json") for item in exploration.subtasks],
                "findings": [item.model_dump(mode="json") for item in exploration.findings],
                "passages": list(unique_passages.values()),
            },
        )
        try:
            return StructuredProtocol.model_validate(raw), raw
        except ValidationError:
            return None, raw

