from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(character for character in lowered if character not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def exact_match(prediction: str | None, answers: list[str]) -> bool:
    if prediction is None:
        return False
    normalized = normalize_answer(prediction)
    return any(normalized == normalize_answer(answer) for answer in answers)


def token_f1(prediction: str, answer: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    overlap = Counter(predicted_tokens) & Counter(answer_tokens)
    shared = sum(overlap.values())
    if not predicted_tokens or not answer_tokens or shared == 0:
        return 0.0
    precision = shared / len(predicted_tokens)
    recall = shared / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)

