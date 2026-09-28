#!/usr/bin/env python3
"""Export one failed, shallow agent retrieval trajectory for review."""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "mapd-shallow-retrieval-negative-trace-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"rollout file not found: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"rollout file is empty: {path}")
    return rows


def discover_rollouts(project_root: str | Path) -> Path:
    root = Path(project_root)
    candidates = list(root.glob("artifacts/paper_examples_action_v*/rollouts.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            "no paper-example rollouts found under artifacts/paper_examples_action_v*/rollouts.jsonl"
        )
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def _search_depth(trajectory: dict[str, Any]) -> int:
    return sum(turn.get("action") == "search" for turn in trajectory.get("turns", []))


def select_negative_trace(
    trajectories: list[dict[str, Any]], example_id: str | None = None
) -> dict[str, Any]:
    candidates = []
    for index, trajectory in enumerate(trajectories):
        if example_id and trajectory.get("example_id") != example_id:
            continue
        if (
            float(trajectory.get("reward", 0.0)) == 0.0
            and trajectory.get("terminated") is True
            and trajectory.get("final_answer")
        ):
            candidates.append((_search_depth(trajectory), index, trajectory))
    if not candidates:
        suffix = f" for example_id={example_id!r}" if example_id else ""
        raise ValueError(f"no terminated zero-reward trajectory with an answer was found{suffix}")
    # Prefer a real search->wrong-answer failure over a zero-search direct answer:
    # the former preserves the retrieval observation needed to diagnose why the
    # policy failed to formulate a follow-up query.
    searched = [item for item in candidates if item[0] > 0]
    return min(searched or candidates, key=lambda item: (item[0], item[1]))[2]


def _manifest_example(manifest: dict[str, Any] | None, example_id: str) -> dict[str, Any] | None:
    if not manifest:
        return None
    for item in manifest.get("examples", []):
        if item.get("example_id") == example_id:
            return item
    return None


def _observation_text(trajectory: dict[str, Any]) -> str:
    return "\n".join(
        str(turn.get("observation") or "")
        for turn in trajectory.get("turns", [])
        if turn.get("action") == "search"
    )


def _normalized_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def diagnose(
    trajectory: dict[str, Any], example: dict[str, Any] | None
) -> dict[str, Any]:
    searches = [turn for turn in trajectory["turns"] if turn.get("action") == "search"]
    actual_depth = len(searches)
    expected_depth = int((example or {}).get("expected_reasoning_hops") or 0)
    answer = str(trajectory.get("final_answer") or "").strip()
    observations = _observation_text(trajectory)
    question_words = _normalized_words(str(trajectory.get("prompt") or ""))
    first_query_words = _normalized_words(str(searches[0].get("query") or "")) if searches else set()
    overlap = (
        len(question_words & first_query_words) / len(question_words)
        if question_words
        else 0.0
    )
    return {
        "expected_reasoning_hops": expected_depth or None,
        "actual_search_actions": actual_depth,
        "depth_shortfall": max(expected_depth - actual_depth, 0) if expected_depth else None,
        "terminated_after_one_search": actual_depth == 1 and trajectory.get("terminated") is True,
        "final_answer_supported_verbatim_by_observation": bool(
            answer and answer.casefold() in observations.casefold()
        ),
        "first_query_question_word_coverage": round(overlap, 4),
        "first_query_likely_copies_whole_question": overlap >= 0.65,
        "follow_up_query_count": max(actual_depth - 1, 0),
        "interpretation": [
            "The policy stopped before the expected multi-hop depth."
            if expected_depth and actual_depth < expected_depth
            else "Compare actual search depth with the expected reasoning chain.",
            "The first query is broad and largely copies the original question instead of resolving one relation."
            if overlap >= 0.65
            else "Inspect whether the first query isolates one unresolved relation.",
            "The final answer string does not occur in the retrieved observation, so the answer is an unsupported guess."
            if answer and answer.casefold() not in observations.casefold()
            else "The final answer occurs in the observation; inspect whether the cited relation is actually supported.",
        ],
    }


def build_trace(
    trajectory: dict[str, Any],
    *,
    source: Path,
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    example = _manifest_example(manifest, str(trajectory["example_id"]))
    return {
        "schema": TRACE_SCHEMA,
        "purpose": "Diagnose premature answering and insufficient retrieval depth.",
        "source_rollouts": str(source),
        "run_manifest": manifest,
        "example_manifest": example,
        "diagnosis": diagnose(trajectory, example),
        "trajectory": trajectory,
        "scope_note": (
            "This is a Qwen agent rollout, because search-vs-answer stopping behavior occurs in the "
            "student agent loop. The offline DeepSeek MAS teacher uses scheduled subtasks and cannot "
            "produce this specific premature-stop behavior."
        ),
    }


def _block(value: Any) -> list[str]:
    return ["```text", str(value or ""), "```", ""]


def render_readable(trace: dict[str, Any]) -> str:
    trajectory = trace["trajectory"]
    example = trace["example_manifest"] or {}
    diagnosis = trace["diagnosis"]
    gold = example.get("gold_answers") or ["not recorded in manifest"]
    lines = [
        "# Failed shallow-retrieval trace",
        "",
        "## What this trace demonstrates",
        "",
        f"- Example: `{trajectory['example_id']}`",
        f"- Question: {trajectory['prompt']}",
        f"- Gold answer(s): {', '.join(str(item) for item in gold)}",
        f"- Model answer: `{trajectory.get('final_answer')}`",
        f"- Reward / exact match: `{trajectory.get('reward')}`",
        f"- Expected reasoning hops: `{diagnosis.get('expected_reasoning_hops')}`",
        f"- Actual search actions: `{diagnosis['actual_search_actions']}`",
        f"- Answer found verbatim in retrieved observation: `{diagnosis['final_answer_supported_verbatim_by_observation']}`",
        "",
        trace["scope_note"],
        "",
        "## Concise diagnosis",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis["interpretation"])
    lines.extend(["", "## Exact interaction trace", "", "### System prompt", ""])
    lines.extend(_block(trajectory.get("system_prompt")))
    lines.extend(["### User question", ""])
    lines.extend(_block(trajectory["prompt"]))
    for turn in trajectory["turns"]:
        lines.extend(
            [
                f"### Turn {turn['turn_index']} — `{turn['action']}`",
                "",
                "Raw model output:",
                "",
            ]
        )
        lines.extend(_block(turn.get("model_output")))
        if turn.get("query") is not None:
            lines.extend([f"Parsed query: `{turn['query']}`", ""])
        if turn.get("observation") is not None:
            lines.extend(["Environment observation:", ""])
            lines.extend(_block(turn["observation"]))
        if turn.get("answer") is not None:
            lines.extend([f"Parsed answer: `{turn['answer']}`", ""])
    lines.extend(
        [
            "## What to ask the reviewer",
            "",
            "1. Is the system prompt sufficient to teach relation-by-relation search, or does the 1.7B model need protocol demonstrations/SFT first?",
            "2. Should the environment reject an answer while required protocol hops remain unresolved?",
            "3. Is the first broad query causing poor retrieval, and should query generation be separately supervised?",
            "4. Does the observation contain enough evidence to formulate the next entity-conditioned query?",
            "",
            "The companion JSON preserves the complete trajectory, including token IDs/log-probabilities when recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def export_shallow_trace(
    rollouts_path: str | Path,
    output_root: str | Path,
    *,
    example_id: str | None = None,
    create_archive: bool = True,
) -> dict[str, str]:
    source = Path(rollouts_path)
    trajectories = _read_jsonl(source)
    selected = select_negative_trace(trajectories, example_id)
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    trace = build_trace(selected, source=source, manifest=manifest)
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(selected["example_id"]))
    output_dir = Path(output_root) / f"shallow_negative_{safe_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    readable = output_dir / "trace_readable.md"
    complete = output_dir / "trace_complete.json"
    readable.write_text(render_readable(trace), encoding="utf-8")
    complete.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = {
        "selected_example": str(selected["example_id"]),
        "search_depth": str(_search_depth(selected)),
        "readable": str(readable.resolve()),
        "complete": str(complete.resolve()),
    }
    if create_archive:
        archive = output_dir.with_suffix(".tar.gz")
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(readable, arcname=readable.name)
            bundle.add(complete, arcname=complete.name)
        result["archive"] = str(archive.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--example-id")
    parser.add_argument("--output-root", type=Path, default=Path("exports"))
    args = parser.parse_args()
    rollouts = args.rollouts or discover_rollouts(args.project_root)
    print(
        json.dumps(
            export_shallow_trace(rollouts, args.output_root, example_id=args.example_id),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
