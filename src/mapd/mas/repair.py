from __future__ import annotations

import re

from mapd.mas.orchestrator import parse_subtasks
from mapd.mas.provider import TeacherClient
from mapd.mas.schema import RepairRecord, SearchFinding, SubTask
from mapd.protocol.leak_check import contains_answer


class Repair:
    def __init__(self, teacher: TeacherClient, max_subquestions: int):
        self.teacher = teacher
        self.max_subquestions = max_subquestions

    def diagnose(
        self,
        question: str,
        answers: list[str],
        candidate: str | None,
        findings: list[SearchFinding],
        round_index: int,
    ) -> tuple[RepairRecord, list[SubTask]]:
        result = self.teacher.generate_json(
            "repair",
            {
                "question": question,
                "ground_truth": answers,
                "candidate_answer": candidate,
                "findings": [item.model_dump(mode="json") for item in findings],
                "round": round_index,
                "max_subquestions": self.max_subquestions,
            },
        )
        proposed = parse_subtasks(
            result.get("subtasks", []),
            round_index,
            f"repair-{round_index}",
            self.max_subquestions,
        )
        proposed = [item for item in proposed if not contains_answer(item.objective, answers)]
        record = RepairRecord(
            round_index=round_index,
            diagnosis=redact_answers(str(result.get("diagnosis", "")), answers),
            failure_type=str(result.get("failure_type", "search")),
            proposed_subtasks=proposed,
        )
        return record, proposed


def redact_answers(text: str, answers: list[str]) -> str:
    for answer in sorted((item for item in answers if item), key=len, reverse=True):
        text = re.sub(re.escape(answer), "[REDACTED]", text, flags=re.IGNORECASE)
    return text

