from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_serializer, model_validator


class TaskType(str, Enum):
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    OTHERS = "others"


class StructuredProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    reasoning_plan: list[str] = Field(min_length=1)
    grounding_facts: list[str] = Field(min_length=1)
    partial_findings: str | None = None
    answer: str | None = None
    answer_grounded: StrictBool

    @model_validator(mode="after")
    def validate_answer_state(self) -> "StructuredProtocol":
        supplied = self.model_fields_set
        if self.answer_grounded and not self.answer:
            raise ValueError("answer_grounded=true requires a non-empty answer")
        if self.answer_grounded and "partial_findings" in supplied:
            raise ValueError("succeeded protocol must omit partial_findings")
        if not self.answer_grounded and "answer" in supplied:
            raise ValueError("evidence protocol must omit answer")
        if not self.answer_grounded and not self.partial_findings:
            raise ValueError("evidence protocol requires non-empty partial_findings")
        return self

    @model_serializer(mode="wrap")
    def serialize_protocol(self, handler):
        return {
            key: value
            for key, value in handler(self).items()
            if value is not None
        }


class QualityReport(BaseModel):
    example_id: str
    passed: bool
    checks: dict[str, bool]
    errors: list[str] = Field(default_factory=list)
    pi_source: str
