from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mapd.environment.parser import parse_action
from mapd.reward.exact_match import exact_match


def _answers(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("answers", "answer", "target"):
            if key in value:
                return _answers(value[key])
        return []
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [] if value is None else [str(value)]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """veRL custom reward entry point using terminal strict exact match."""
    del data_source, extra_info
    action = parse_action(solution_str)
    prediction = action.value if action.kind == "answer" else None
    return float(exact_match(prediction, _answers(ground_truth)))

