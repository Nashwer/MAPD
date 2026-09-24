from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mapd.data.schema import QASample
from mapd.protocol.schema import QualityReport, StructuredProtocol, TaskType
from mapd.retrieval.schema import RetrievedPassage


class SubTask(BaseModel):
    id: str
    objective: str
    depends_on: list[str] = Field(default_factory=list)
    round_index: int = Field(default=1, ge=1)


class SearchRecord(BaseModel):
    subtask_id: str
    query: str
    passages: list[RetrievedPassage]


class SearchFinding(BaseModel):
    subtask_id: str
    summary: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class RepairRecord(BaseModel):
    round_index: int = Field(ge=1)
    diagnosis: str
    failure_type: str
    proposed_subtasks: list[SubTask] = Field(default_factory=list)


class ExplorationLog(BaseModel):
    example_id: str
    question: str
    task_type: TaskType
    subtasks: list[SubTask]
    searches: list[SearchRecord]
    findings: list[SearchFinding] = Field(default_factory=list)
    repairs: list[RepairRecord] = Field(default_factory=list)
    candidate_answer: str | None = None
    success: bool = False
    search_rounds: int = 1
    repair_rounds: int = 0


class SynthesisArtifact(BaseModel):
    sample: QASample
    exploration: ExplorationLog
    protocol: StructuredProtocol | None = None
    raw_protocol: dict[str, Any] = Field(default_factory=dict)
    quality: QualityReport
    metadata: dict[str, Any] = Field(default_factory=dict)
