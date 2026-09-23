from mapd.data.schema import QASample
from mapd.mas.schema import ExplorationLog, SearchRecord, SubTask
from mapd.protocol.schema import StructuredProtocol
from mapd.protocol.leak_check import contains_answer
from mapd.protocol.validator import validate_protocol
from mapd.retrieval.schema import RetrievedPassage


def test_quality_gate_accepts_grounded_protocol():
    sample = QASample(id="1", question="Where is the tower?", answers=["Paris"])
    passage = RetrievedPassage(id="p1", contents="The tower is located in Paris.")
    exploration = ExplorationLog(
        example_id="1",
        question=sample.question,
        task_type="single_hop",
        subtasks=[SubTask(id="s1", objective=sample.question)],
        searches=[SearchRecord(subtask_id="s1", query=sample.question, passages=[passage])],
        candidate_answer="Paris",
        success=True,
    )
    protocol = StructuredProtocol(
        task_type="single_hop",
        reasoning_plan=["Search for the location and verify the source."],
        grounding_facts=[passage.contents],
        answer="Paris",
        answer_grounded=True,
    )
    assert validate_protocol(sample, exploration, protocol).passed


def test_quality_gate_rejects_plan_answer_leak():
    sample = QASample(id="1", question="Where is the tower?", answers=["Paris"])
    passage = RetrievedPassage(id="p1", contents="The tower is located in Paris.")
    exploration = ExplorationLog(
        example_id="1",
        question=sample.question,
        task_type="single_hop",
        subtasks=[SubTask(id="s1", objective=sample.question)],
        searches=[SearchRecord(subtask_id="s1", query=sample.question, passages=[passage])],
        candidate_answer="Paris",
        success=True,
    )
    protocol = StructuredProtocol(
        task_type="single_hop",
        reasoning_plan=["Conclude that the answer is Paris."],
        grounding_facts=[passage.contents],
        answer="Paris",
        answer_grounded=True,
    )
    report = validate_protocol(sample, exploration, protocol)
    assert not report.passed
    assert not report.checks["no_answer_leak"]


def test_leak_check_matches_token_sequences_not_substrings():
    assert contains_answer("Conclude that the answer is US.", ["US"])
    assert not contains_answer("Use two sources and compare them.", ["US"])
