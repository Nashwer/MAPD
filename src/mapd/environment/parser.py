import re
from dataclasses import dataclass


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    value: str | None = None


def parse_action(model_output: str) -> ParsedAction:
    answer_match = ANSWER_PATTERN.search(model_output)
    if answer_match:
        return ParsedAction("answer", answer_match.group(1).strip() or None)
    search_match = SEARCH_PATTERN.search(model_output)
    if search_match:
        return ParsedAction("search", search_match.group(1).strip())
    return ParsedAction("invalid")

