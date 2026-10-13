from mapd.data.schema import QASample
from mapd.mas.schema import ExplorationLog, SearchRecord, SubTask
from mapd.protocol.schema import StructuredProtocol
from mapd.protocol.leak_check import contains_answer, hidden_answers
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
    report = validate_protocol(sample, exploration, protocol)
    assert report.passed
    assert set(report.checks) == {
        "schema",
        "em_consistency",
        "extractive_grounding",
        "no_answer_leak",
    }


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


def test_answer_candidate_named_in_question_is_not_an_oracle_leak():
    sample = QASample(
        id="hotpotqa:train_14873",
        question=(
            "Which landmark of Istanbul is located higher, the Column of the Goths "
            "or the Mihrimah Sultan Mosque?"
        ),
        answers=["Mihrimah Sultan Mosque"],
    )
    passage = RetrievedPassage(
        id="p1",
        contents="The Mihrimah Sultan Mosque stands on the Sixth Hill of Istanbul.",
    )
    exploration = ExplorationLog(
        example_id=sample.id,
        question=sample.question,
        task_type="comparison",
        subtasks=[SubTask(id="s1", objective=sample.question)],
        searches=[
            SearchRecord(
                subtask_id="s1",
                query="Mihrimah Sultan Mosque elevation",
                passages=[passage],
            )
        ],
        candidate_answer=sample.answers[0],
        success=True,
    )
    protocol = StructuredProtocol(
        task_type="comparison",
        reasoning_plan=["Compare the elevations of both landmarks named in the question."],
        grounding_facts=[passage.contents],
        answer=sample.answers[0],
        answer_grounded=True,
    )

    assert hidden_answers(sample.question, sample.answers) == []
    assert validate_protocol(sample, exploration, protocol).passed
