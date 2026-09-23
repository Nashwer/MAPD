from mapd.environment.search_env import AgenticSearchEnvironment, ScriptedPolicy, generated_trajectory_text
from mapd.data.schema import QASample
from mapd.retrieval.wiki18 import BM25Retriever


def test_agentic_search_rollout_executes_tool_and_scores_em():
    sample = QASample(id="q1", question="Where is the Eiffel Tower located?", answers=["Paris"])
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl"), top_k=1, max_turns=2
    )
    trajectory = environment.rollout(
        sample,
        ScriptedPolicy(
            [
                "I should look it up. <search>Eiffel Tower location</search>",
                "The passage verifies it. <answer>Paris</answer>",
            ]
        ),
    )

    assert trajectory.reward == 1.0
    assert trajectory.terminated
    assert trajectory.turns[0].observation.startswith("<information>")
    assert "<information>" not in generated_trajectory_text(trajectory)
