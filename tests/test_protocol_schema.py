import pytest
from pydantic import ValidationError

from mapd.mas.protocolizer import Protocolizer
from mapd.mas.schema import ExplorationLog
from mapd.protocol.schema import StructuredProtocol


def test_succeeded_protocol_matches_paper_shape_exactly():
    protocol = StructuredProtocol.model_validate(
        {
            "task_type": "multi_hop",
            "reasoning_plan": ["Find the entity.", "Verify the requested fact."],
            "grounding_facts": ["A verbatim retrieved fact."],
            "answer": "verified answer",
            "answer_grounded": True,
        }
    )

    assert protocol.model_dump(mode="json") == {
        "task_type": "multi_hop",
        "reasoning_plan": ["Find the entity.", "Verify the requested fact."],
        "grounding_facts": ["A verbatim retrieved fact."],
        "answer": "verified answer",
        "answer_grounded": True,
    }


def test_evidence_protocol_matches_paper_shape_exactly():
    protocol = StructuredProtocol.model_validate(
        {
            "task_type": "multi_hop",
            "reasoning_plan": ["Find the entity.", "Search for the missing fact."],
            "grounding_facts": ["A verbatim retrieved fact."],
            "partial_findings": "The entity is known, but the requested fact is missing.",
            "answer_grounded": False,
        }
    )

    assert protocol.model_dump(mode="json") == {
        "task_type": "multi_hop",
        "reasoning_plan": ["Find the entity.", "Search for the missing fact."],
        "grounding_facts": ["A verbatim retrieved fact."],
        "partial_findings": "The entity is known, but the requested fact is missing.",
        "answer_grounded": False,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {
            "protocol": {
                "task_type": "multi_hop",
                "reasoning_plan": ["Search."],
                "grounding_facts": ["Fact."],
                "answer": "answer",
                "answer_grounded": True,
            },
            "type": "json_object",
        },
        {
            "task_type": "multi_hop",
            "reasoning_plan": ["Search."],
            "grounding_facts": ["Fact."],
            "partial_findings": None,
            "answer": "answer",
            "answer_grounded": True,
        },
        {
            "task_type": "multi_hop",
            "reasoning_plan": ["Search."],
            "grounding_facts": ["Fact."],
            "partial_findings": "Missing evidence.",
            "answer": None,
            "answer_grounded": False,
        },
    ],
)
def test_protocol_schema_rejects_wrappers_extras_and_null_variant_fields(payload):
    with pytest.raises(ValidationError):
        StructuredProtocol.model_validate(payload)


def test_protocolizer_preserves_strict_schema_error_details():
    raw = {
        "protocol": {
            "task_type": "multi_hop",
            "reasoning_plan": ["Search."],
            "grounding_facts": ["Fact."],
            "answer": "answer",
            "answer_grounded": True,
        },
        "type": "json_object",
    }

    class WrappedTeacher:
        model = "wrapped-teacher"

        def generate_json(self, role, payload):
            return raw

    exploration = ExplorationLog(
        example_id="1",
        question="Question?",
        task_type="multi_hop",
        subtasks=[],
        searches=[],
    )
    protocol, preserved_raw, errors = Protocolizer(WrappedTeacher()).generate(
        exploration
    )

    assert protocol is None
    assert preserved_raw == raw
    assert any("schema task_type: Field required" in error for error in errors)
    assert any("schema protocol: Extra inputs are not permitted" in error for error in errors)
