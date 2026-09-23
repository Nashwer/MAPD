from mapd.protocol.schema import StructuredProtocol
from mapd.trainer.mapd_trainer import joint_loss
from mapd.trainer.opsd import PrivilegedSequence, protocol_context


def test_privileged_sequence_preserves_response_alignment():
    sequence = PrivilegedSequence(
        student_prompt_ids=[1, 2],
        response_ids=[3, 4, 5],
        response_mask=[True, False, True],
        privileged_context_ids=[9, 9],
    )
    assert sequence.teacher_sequence_ids == [1, 2, 9, 9, 3, 4, 5]


def test_joint_loss_and_protocol_serialization():
    protocol = StructuredProtocol(
        task_type="single_hop",
        reasoning_plan=["Search and verify."],
        grounding_facts=["A grounded fact."],
        answer="answer",
        answer_grounded=True,
    )
    assert '"task_type": "single_hop"' in protocol_context(protocol)
    assert joint_loss(2.0, 4.0, 0.05) == 2.2
