from pydantic import BaseModel


class RetrievedPassage(BaseModel):
    id: str
    contents: str
    score: float = 0.0

