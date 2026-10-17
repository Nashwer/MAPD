from __future__ import annotations

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory, AgentTurn
from scripts.compare_student_evals import paired_report
from scripts.student_eval import _write_manifest, select_cohort, summarize


def _sample(index: int, source: str) -> QASample:
    return QASample(
        id=f"{source}:test_{index}",
        question=f"question {source} {index}",
        answers=[f"answer {index}"],
        split="test",
        data_source=source,
    )


def _trajectory(sample: QASample, *, correct: bool, searches: int) -> AgentTrajectory:
    turns = [
        AgentTurn(
            turn_index=index + 1,
            model_output=f"<search>query {index}</search>",
            action="search",
            query=f"query {index}",
            observation="<information>evidence</information>",
        )
        for index in range(searches)
    ]
    answer = sample.answers[0] if correct else "wrong"
    turns.append(
        AgentTurn(
            turn_index=len(turns) + 1,
            model_output=f"<answer>{answer}</answer>",
            action="answer",
            answer=answer,
        )
    )
    return AgentTrajectory(
        example_id=sample.id,
        prompt=sample.question,
        turns=turns,
        final_answer=answer,
        reward=float(correct),
        terminated=True,
    )


def test_eval_cohort_is_deterministic_and_source_stratified():
    samples = [_sample(index, "nq") for index in range(10)] + [
        _sample(index, "hotpotqa") for index in range(10)
    ]

    first = select_cohort(samples, count=8, seed=2026)
    second = select_cohort(samples, count=8, seed=2026)

    assert [item.id for item in first] == [item.id for item in second]
    assert sum(item.data_source == "nq" for item in first) == 4
    assert sum(item.data_source == "hotpotqa" for item in first) == 4


def test_eval_summary_reports_accuracy_and_retrieval_depth():
    samples = [_sample(1, "nq"), _sample(2, "hotpotqa")]
    trajectories = [
        _trajectory(samples[0], correct=True, searches=1),
        _trajectory(samples[1], correct=False, searches=2),
    ]

    result = summarize(samples, trajectories)

    assert result["strict_accuracy"] == 0.5
    assert result["mean_search_depth"] == 1.5
    assert result["multi_search_rate"] == 0.5
    assert set(result["by_source"]) == {"nq", "hotpotqa"}


def test_paired_report_uses_the_same_examples():
    samples = [_sample(1, "nq"), _sample(2, "hotpotqa")]
    baseline = [
        _trajectory(samples[0], correct=False, searches=1),
        _trajectory(samples[1], correct=True, searches=1),
    ]
    trained = [
        _trajectory(samples[0], correct=True, searches=2),
        _trajectory(samples[1], correct=True, searches=2),
    ]

    result = paired_report(baseline, trained, bootstrap_samples=100, seed=1)

    assert result["baseline_accuracy"] == 0.5
    assert result["trained_accuracy"] == 1.0
    assert result["accuracy_delta"] == 0.5
    assert result["paired_outcomes"]["trained_only_correct"] == 1
    assert result["mean_search_depth_delta"] == 1


def test_partial_eval_manifest_can_be_checkpointed(tmp_path):
    samples = [_sample(1, "nq"), _sample(2, "hotpotqa")]
    trajectory = _trajectory(samples[0], correct=True, searches=1)
    manifest = tmp_path / "manifest.json"

    _write_manifest(
        manifest,
        {"schema": "test"},
        samples,
        [trajectory],
        {"id_overlap": 0, "question_overlap": 0},
    )

    contents = manifest.read_text(encoding="utf-8")
    assert '"completed": false' in contents
    assert '"examples": 1' in contents
