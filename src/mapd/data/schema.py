from pydantic import BaseModel, Field


class QASample(BaseModel):
    id: str
    question: str
    answers: list[str] = Field(min_length=1)
    split: str = "train"
    data_source: str = "unknown"

