from __future__ import annotations

import httpx

from mapd.retrieval.schema import RetrievedPassage


class HTTPRetriever:
    """Client for the Search-R1-style batched retriever server API."""

    def __init__(self, url: str, timeout_seconds: float = 30):
        self.url = url
        self.timeout_seconds = timeout_seconds

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        response = httpx.post(
            self.url,
            json={"queries": [query], "topk": top_k, "return_scores": True},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        items = response.json()["result"][0]
        passages = []
        for index, item in enumerate(items):
            document = item.get("document", item)
            if isinstance(document, str):
                passage_id, contents = str(index), document
            else:
                passage_id = str(document.get("id", index))
                contents = document.get("contents") or document.get("text") or ""
            passages.append(
                RetrievedPassage(
                    id=passage_id,
                    contents=contents,
                    score=float(item.get("score", 0)),
                )
            )
        return passages

