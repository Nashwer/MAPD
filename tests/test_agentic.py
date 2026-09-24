from mapd.environment.search_env import (
    AgenticSearchEnvironment,
    ScriptedPolicy,
    generated_trajectory_text,
)
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


def test_agent_can_chain_three_searches_for_a_paper_style_multihop_question():
    sample = QASample(
        id="paper-example-3",
        question=(
            "What year was the barber surgeon who served the last French monarch "
            "of the House of Valois born?"
        ),
        answers=["1510"],
    )
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl"),
        top_k=1,
        max_turns=4,
    )
    trajectory = environment.rollout(
        sample,
        ScriptedPolicy(
            [
                "<search>last French monarch House of Valois</search>",
                "<search>Henry III barber surgeon</search>",
                "<search>Ambroise Pare birth year</search>",
                "<answer>1510</answer>",
            ]
        ),
    )

    assert [turn.action for turn in trajectory.turns] == [
        "search",
        "search",
        "search",
        "answer",
    ]
    assert trajectory.reward == 1.0


def test_response_token_limit_is_shared_across_interaction_turns():
    class BudgetPolicy:
        def __init__(self):
            self.calls = []
            self.outputs = [
                ("<search>first</search>", [1, 2]),
                ("<search>second</search>", [3, 4]),
                ("<answer>Paris</answer>", [5]),
            ]
            self.last_token_ids = None
            self.last_token_log_probs = None

        def generate(self, messages, max_new_tokens):
            del messages
            self.calls.append(max_new_tokens)
            output, token_ids = self.outputs[len(self.calls) - 1]
            self.last_token_ids = token_ids
            self.last_token_log_probs = [-0.1] * len(token_ids)
            return output

    sample = QASample(id="q1", question="Where?", answers=["Paris"])
    policy = BudgetPolicy()
    environment = AgenticSearchEnvironment(
        BM25Retriever.from_jsonl("tests/fixtures/corpus.jsonl"),
        max_turns=4,
    )

    trajectory = environment.rollout(sample, policy, max_new_tokens=5)

    assert policy.calls == [5, 3, 1]
    assert trajectory.reward == 1.0
