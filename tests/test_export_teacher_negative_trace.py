import json

from scripts.export_teacher_negative_trace import diagnose, load_artifact


def test_teacher_negative_diagnosis_counts_ineffective_queries(tmp_path):
    artifact = {
        "sample": {"id": "hotpotqa:train_82354", "answers": ["Blue Line"]},
        "exploration": {
            "task_type": "multi_hop",
            "success": False,
            "candidate_answer": "Red Line",
            "search_rounds": 2,
            "repair_rounds": 2,
            "subtasks": [
                {"id": "s1", "depends_on": []},
                {"id": "s2", "depends_on": ["s1"]},
            ],
            "searches": [
                {"query": "mall developer", "passages": [{"score": 2.0}]},
                {"query": "mall rail line", "passages": [{"score": 0.0}]},
            ],
            "findings": [{}, {}],
        },
    }
    path = tmp_path / "artifacts.jsonl"
    path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    loaded = load_artifact(path, "hotpotqa:train_82354")
    result = diagnose(loaded)
    assert result["retrieval_queries"] == 2
    assert result["dependent_subtasks"] == 1
    assert result["zero_score_query_fraction"] == 0.5
