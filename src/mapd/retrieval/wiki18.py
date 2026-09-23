from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from mapd.io import read_jsonl
from mapd.retrieval.schema import RetrievedPassage


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """In-process BM25 used for fixtures and small wiki-18 shards."""

    def __init__(self, passages: list[RetrievedPassage], k1: float = 1.5, b: float = 0.75):
        if not passages:
            raise ValueError("BM25 corpus must not be empty")
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokens(passage.contents) for passage in passages]
        self.avg_len = sum(map(len, self.doc_tokens)) / len(self.doc_tokens)
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Retriever":
        return cls(read_jsonl(path, RetrievedPassage))

    def search(self, query: str, top_k: int) -> list[RetrievedPassage]:
        query_tokens = _tokens(query)
        count = len(self.passages)
        scored: list[RetrievedPassage] = []
        for passage, tokens in zip(self.passages, self.doc_tokens, strict=True):
            frequencies = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if frequency == 0:
                    continue
                document_frequency = self.doc_freq[token]
                inverse_frequency = math.log(
                    1 + (count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * len(tokens) / self.avg_len)
                score += inverse_frequency * frequency * (self.k1 + 1) / denominator
            scored.append(passage.model_copy(update={"score": score}))
        scored.sort(key=lambda item: (-item.score, item.id))
        return scored[:top_k]

