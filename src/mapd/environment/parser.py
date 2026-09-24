import re
from dataclasses import dataclass


SEARCH_PATTERN = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
INFORMATION_PATTERN = re.compile(
    r"</?information\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    value: str | None = None


def parse_action(model_output: str) -> ParsedAction:
    # The environment, never the model, owns <information> observations.  A
    # model response containing both actions previously skipped retrieval
    # because the answer branch had priority.  Require one complete action so
    # malformed generations are corrected on the next turn instead.
    if INFORMATION_PATTERN.search(model_output):
        return ParsedAction("invalid")
    search_matches = list(SEARCH_PATTERN.finditer(model_output))
    answer_matches = list(ANSWER_PATTERN.finditer(model_output))
    if len(search_matches) + len(answer_matches) != 1:
        return ParsedAction("invalid")
    if search_matches:
        query = search_matches[0].group(1).strip()
        return ParsedAction("search", query) if query else ParsedAction("invalid")
    if answer_matches:
        answer = answer_matches[0].group(1).strip()
        return ParsedAction("answer", answer) if answer else ParsedAction("invalid")
    return ParsedAction("invalid")
