from pydantic import BaseModel, Field, model_validator


class AgentTurn(BaseModel):
    turn_index: int = Field(ge=1)
    model_output: str
    action: str
    query: str | None = None
    answer: str | None = None
    observation: str | None = None
    token_ids: list[int] | None = None
    token_log_probs: list[float] | None = None

    @model_validator(mode="after")
    def validate_generation_trace(self) -> "AgentTurn":
        if self.token_log_probs is not None:
            if self.token_ids is None or len(self.token_log_probs) != len(self.token_ids):
                raise ValueError("token_log_probs must align exactly with token_ids")
        return self


class AgentTrajectory(BaseModel):
    example_id: str
    prompt: str
    system_prompt: str | None = None
    turns: list[AgentTurn]
    final_answer: str | None = None
    reward: float = Field(ge=0.0, le=1.0)
    terminated: bool = False
