from mapd.data.schema import QASample
from mapd.data.verl import to_verl_record
from mapd.environment.search_env import AGENT_SYSTEM_PROMPT
from mapd.mas.schema import ExplorationLog, SynthesisArtifact
from mapd.protocol.schema import QualityReport, StructuredProtocol
from mapd.reward.verl_adapter import compute_score


def test_verl_reward_requires_terminal_answer_and_supports_aliases():
    assert compute_score("nq", "reasoning <answer>Paris</answer>", ["Paris", "Paris, France"]) == 1.0
    assert compute_score("nq", "<search>Paris</search>", ["Paris"]) == 0.0
    assert compute_score("nq", "<answer>London</answer>", {"answers": ["Paris"]}) == 0.0


def test_verl_record_contains_system_prompt_and_protocol():
    sample = QASample(id="q1", question="Where?", answers=["Paris"], data_source="nq")
    protocol = StructuredProtocol(
        task_type="single_hop",
        reasoning_plan=["Search for the location."],
        grounding_facts=["The location is Paris."],
        answer="Paris",
        answer_grounded=True,
    )
    artifact = SynthesisArtifact(
        sample=sample,
        exploration=ExplorationLog(
            example_id="q1",
            question="Where?",
            task_type="single_hop",
            subtasks=[],
            searches=[],
            candidate_answer="Paris",
            success=True,
        ),
        protocol=protocol,
        quality=QualityReport(
            example_id="q1",
            passed=True,
            checks={"schema": True},
            pi_source="protocol",
        ),
    )

    record = to_verl_record(artifact, 0)
    assert record["prompt"][0] == {"role": "system", "content": AGENT_SYSTEM_PROMPT}
    assert record["prompt"][1]["content"] == sample.question
    assert record["extra_info"]["protocol"]["answer"] == "Paris"

