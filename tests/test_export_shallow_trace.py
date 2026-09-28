import json
import tarfile

from scripts.export_shallow_trace import export_shallow_trace, select_negative_trace


def _trajectory(*, answer, reward, searches):
    turns = []
    for index in range(searches):
        turns.append(
            {
                "turn_index": index + 1,
                "model_output": "<search>English actor film Die Hard</search>",
                "action": "search",
                "query": "English actor film Die Hard",
                "observation": "<information>Unrelated result</information>",
                "token_ids": [1],
                "token_log_probs": [-0.1],
            }
        )
    turns.append(
        {
            "turn_index": searches + 1,
            "model_output": f"<answer>{answer}</answer>",
            "action": "answer",
            "answer": answer,
            "token_ids": [2],
            "token_log_probs": [-0.2],
        }
    )
    return {
        "example_id": "mapd-paper-example-2",
        "prompt": "Which English actor in a film also appeared in Die Hard?",
        "system_prompt": "Search before answering.",
        "turns": turns,
        "final_answer": answer,
        "reward": reward,
        "terminated": True,
    }


def test_selects_shallowest_failed_answer_and_exports_both_views(tmp_path):
    deep = _trajectory(answer="Wrong A", reward=0.0, searches=3)
    shallow = _trajectory(answer="Wrong B", reward=0.0, searches=1)
    correct = _trajectory(answer="Alan Rickman", reward=1.0, searches=1)
    assert select_negative_trace([deep, shallow, correct])["final_answer"] == "Wrong B"

    run_dir = tmp_path / "artifacts/paper_examples_action_v4"
    run_dir.mkdir(parents=True)
    rollouts = run_dir / "rollouts.jsonl"
    rollouts.write_text(
        "\n".join(json.dumps(item) for item in [deep, shallow, correct]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "examples": [
                    {
                        "example_id": "mapd-paper-example-2",
                        "gold_answers": ["Alan Rickman"],
                        "expected_reasoning_hops": 3,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    outputs = export_shallow_trace(rollouts, tmp_path / "exports")
    readable = open(outputs["readable"], encoding="utf-8").read()
    complete = json.load(open(outputs["complete"], encoding="utf-8"))
    assert "Actual search actions: `1`" in readable
    assert "unsupported guess" in readable
    assert complete["trajectory"]["final_answer"] == "Wrong B"
    assert complete["diagnosis"]["depth_shortfall"] == 2
    with tarfile.open(outputs["archive"], "r:gz") as bundle:
        assert sorted(bundle.getnames()) == ["trace_complete.json", "trace_readable.md"]


def test_prefers_one_search_failure_over_direct_answer():
    direct = _trajectory(answer="Unsupported", reward=0.0, searches=0)
    searched = _trajectory(answer="Wrong after evidence", reward=0.0, searches=1)
    assert select_negative_trace([direct, searched])["final_answer"] == "Wrong after evidence"
