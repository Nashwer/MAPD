from enum import Enum

from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    OTHERS = "others"


class StructuredProtocol(BaseModel):
    task_type: TaskType
    reasoning_plan: list[str] = Field(min_length=1)
    grounding_facts: list[str] = Field(default_factory=list)
    partial_findings: str | None = None
    answer: str | None = None
    answer_grounded: bool

    @model_validator(mode="after")
    def validate_answer_state(self) -> "StructuredProtocol":
        if self.answer_grounded and not self.answer:
            raise ValueError("answer_grounded=true requires a non-empty answer")
        if not self.answer_grounded and self.answer is not None:
            raise ValueError("un-grounded evidence protocol must omit answer")
        return self


class QualityReport(BaseModel):
    example_id: str
    passed: bool
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    pi_source: str

