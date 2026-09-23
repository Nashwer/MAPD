from pydantic import BaseModel, Field


class AgentTurn(BaseModel):
    turn_index: int = Field(ge=1)
    model_output: str
    action: str
    query: str | None = None
    answer: str | None = None
    observation: str | None = None


class AgentTrajectory(BaseModel):
    example_id: str
    prompt: str
    turns: list[AgentTurn]
    final_answer: str | None = None
    reward: float = Field(ge=0.0, le=1.0)
    terminated: bool = False

