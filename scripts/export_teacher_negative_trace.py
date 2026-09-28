#!/usr/bin/env python3
"""Export persisted DeepSeek/MAS failure traces for effective-depth review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TRACE_SCHEMA = "mapd-teacher-negative-trace-v1"
DEFAULT_EXAMPLE_IDS = ("hotpotqa:train_82354", "hotpotqa:train_56119")


def load_artifact(path: str | Path, example_id: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"teacher artifact file not found: {source}")
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("sample", {}).get("id") == example_id:
            return value
    raise ValueError(f"example_id {example_id!r} was not found in {source}")


def _role_instructions() -> dict[str, str]:
    # Import lazily so simple file-selection tests do not require an installed package.
    from mapd.mas.provider import ROLE_INSTRUCTIONS

    return dict(ROLE_INSTRUCTIONS)


def diagnose(artifact: dict[str, Any]) -> dict[str, Any]:
    exploration = artifact["exploration"]
    searches = exploration["searches"]
    queries = [str(item["query"]) for item in searches]
    zero_score = sum(
        not item.get("passages")
        or max(float(passage.get("score") or 0.0) for passage in item["passages"]) <= 0.0
        for item in searches
    )
    dependent = [item for item in exploration["subtasks"] if item.get("depends_on")]
    return {
        "task_type": exploration["task_type"],
        "exploration_success": exploration["success"],
        "candidate_answer": exploration.get("candidate_answer"),
        "gold_answers": artifact["sample"]["answers"],
        "subtasks": len(exploration["subtasks"]),
        "dependent_subtasks": len(dependent),
        "search_rounds": exploration["search_rounds"],
        "repair_rounds": exploration["repair_rounds"],
        "retrieval_queries": len(searches),
        "unique_queries": len({query.casefold().strip() for query in queries}),
        "zero_score_query_count": zero_score,
        "zero_score_query_fraction": round(zero_score / len(searches), 4) if searches else 0.0,
        "findings": len(exploration["findings"]),
        "interpretation": [
            "This is not a low call-count failure; it is an effective-depth failure.",
            "Inspect whether an incorrect intermediate entity is propagated into dependent searches.",
            "Inspect repeated paraphrases and zero-score results that consume calls without resolving a new relation.",
            "Inspect whether repair changes the entity/relation being searched or merely repeats the original path.",
        ],
    }


def build_trace(
    artifact: dict[str, Any], *, source: Path, run_manifest: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "purpose": "Diagnose ineffective multi-hop depth in the offline DeepSeek/MAS teacher.",
        "source_artifacts": str(source),
        "scope_note": (
            "The teacher pipeline schedules subtasks and searches before answering. Therefore this trace "
            "measures effective dependency depth and query quality, not student-style early stopping."
        ),
        "capture_limitations": [
            "Raw HTTP request/response bodies were not persisted.",
            "The artifact contains validated role outputs, all queries, retrieved passages, findings, repairs, and the final candidate.",
            "Per-request usage is excluded because usage rows are not keyed by example_id.",
        ],
        "teacher_role_instructions": _role_instructions(),
        "run_manifest": run_manifest,
        "diagnosis": diagnose(artifact),
        "artifact": artifact,
    }


def _one_line(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_readable(trace: dict[str, Any]) -> str:
    artifact = trace["artifact"]
    sample = artifact["sample"]
    exploration = artifact["exploration"]
    diagnosis = trace["diagnosis"]
    findings = {item["subtask_id"]: item for item in exploration["findings"]}
    lines = [
        "# Failed teacher/MAS retrieval trace",
        "",
        "## Outcome",
        "",
        f"- Example: `{sample['id']}`",
        f"- Question: {sample['question']}",
        f"- Gold answer(s): {', '.join(sample['answers'])}",
        f"- Teacher candidate: `{exploration.get('candidate_answer')}`",
        f"- Task type: `{exploration['task_type']}`",
        f"- Exploration success: `{exploration['success']}`",
        f"- Search / repair rounds: `{exploration['search_rounds']} / {exploration['repair_rounds']}`",
        f"- Retrieval queries: `{diagnosis['retrieval_queries']}`",
        f"- Zero-score queries: `{diagnosis['zero_score_query_count']}` "
        f"(`{diagnosis['zero_score_query_fraction']:.1%}`)",
        "",
        trace["scope_note"],
        "",
        "## Review hypotheses",
        "",
    ]
    lines.extend(f"- {item}" for item in diagnosis["interpretation"])
    lines.extend(["", "## Complete persisted retrieval chain", ""])
    searches_by_task: dict[str, list[dict[str, Any]]] = {}
    for search in exploration["searches"]:
        searches_by_task.setdefault(search["subtask_id"], []).append(search)
    for subtask in exploration["subtasks"]:
        dependencies = ", ".join(subtask.get("depends_on", [])) or "none"
        lines.extend(
            [
                f"### {subtask['id']}",
                "",
                f"Objective: {subtask['objective']}",
                "",
                f"Depends on: `{dependencies}`",
                "",
            ]
        )
        for query_index, search in enumerate(searches_by_task.get(subtask["id"], []), 1):
            lines.extend([f"Query {query_index}: `{search['query']}`", ""])
            for rank, passage in enumerate(search.get("passages", []), 1):
                lines.append(
                    f"- rank {rank}, id `{passage['id']}`, score `{float(passage.get('score') or 0):.4f}`: "
                    f"{_one_line(passage.get('contents'))}"
                )
            lines.append("")
        finding = findings.get(subtask["id"])
        if finding:
            lines.extend(
                [
                    f"Finding: {finding['summary']}",
                    f"Evidence IDs: {', '.join(finding.get('evidence_ids', []))}",
                    "",
                ]
            )
    lines.extend(["## Repair decisions", ""])
    if exploration["repairs"]:
        lines.extend(
            ["```json", json.dumps(exploration["repairs"], ensure_ascii=False, indent=2), "```", ""]
        )
    else:
        lines.extend(["No repair was recorded.", ""])
    lines.extend(
        [
            "## Protocol emitted after the failed exploration",
            "",
            "```json",
            json.dumps(artifact.get("raw_protocol"), ensure_ascii=False, indent=2),
            "```",
            "",
            "## Exact teacher role instructions",
            "",
        ]
    )
    for role, instruction in trace["teacher_role_instructions"].items():
        lines.extend([f"### {role}", "", instruction, ""])
    lines.extend(
        [
            "## Capture limitation",
            "",
            "The companion JSON is the complete persisted trace. The original run did not retain raw HTTP bodies, "
            "so they cannot be reconstructed exactly after the fact.",
            "",
        ]
    )
    return "\n".join(lines)


def export_teacher_negative(
    artifacts_path: str | Path, example_id: str, output_root: str | Path
) -> dict[str, str]:
    source = Path(artifacts_path)
    artifact = load_artifact(source, example_id)
    manifest_path = source.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    trace = build_trace(artifact, source=source, run_manifest=manifest)
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", example_id)
    output_dir = Path(output_root) / f"teacher_negative_{safe_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    readable = output_dir / "trace_readable.md"
    complete = output_dir / "trace_complete.json"
    readable.write_text(render_readable(trace), encoding="utf-8")
    complete.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "example_id": example_id,
        "readable": str(readable.resolve()),
        "complete": str(complete.resolve()),
    }
