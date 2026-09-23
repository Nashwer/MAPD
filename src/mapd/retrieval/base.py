from typing import Protocol

from mapd.retrieval.schema import RetrievedPassage


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedPassage]: ...

