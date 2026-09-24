from mapd.environment.search_env import AgenticSearchEnvironment, ScriptedPolicy, generated_trajectory_text
from mapd.environment.parser import parse_action
from mapd.data.schema import QASample
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.reward.exact_match import exact_match


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


def test_strict_em_rejects_explanatory_sentence_around_correct_answer():
    assert exact_match("Alan Rickman", ["Alan Rickman"])
    assert not exact_match(
        "The 1997 film The Winter Guest was directed by Alan Rickman.",
        ["Alan Rickman"],
    )


def test_parser_rejects_multiple_actions_and_model_generated_information():
    assert parse_action("<search>tower</search><answer>Paris</answer>").kind == "invalid"
    assert (
        parse_action(
            "<search>tower</search><information>Paris</information><answer>Paris</answer>"
        ).kind
        == "invalid"
    )


def test_required_search_reprompts_an_early_answer_before_using_tool():
    sample = QASample(id="q1", question="Where is the Eiffel Tower located?", answers=["Paris"])
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl"),
        top_k=1,
        max_turns=3,
        require_search=True,
    )
    trajectory = environment.rollout(
        sample,
        ScriptedPolicy(
            [
                "<answer>Paris</answer>",
                "<search>Eiffel Tower location</search>",
                "<answer>Paris</answer>",
            ]
        ),
    )

    assert [turn.action for turn in trajectory.turns] == ["invalid", "search", "answer"]
    assert trajectory.reward == 1.0
