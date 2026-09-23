from __future__ import annotations

from mapd.data.schema import QASample
from mapd.mas.schema import ExplorationLog
from mapd.protocol.grounding_check import facts_are_extractive
from mapd.protocol.leak_check import protocol_has_leak
from mapd.protocol.schema import QualityReport, StructuredProtocol
from mapd.reward.exact_match import exact_match


def validate_protocol(
    sample: QASample,
    exploration: ExplorationLog,
    protocol: StructuredProtocol | None,
) -> QualityReport:
    if protocol is None:
        return QualityReport(
            example_id=sample.id,
            passed=False,
            checks={
                "schema": False,
                "em_consistency": False,
                "extractive_grounding": False,
                "no_answer_leak": False,
                "grounded_flag": False,
                "variant_shape": False,
            },
            errors=["protocol failed schema validation"],
            pi_source="self_rollout_fallback",
        )

    checks: dict[str, bool] = {"schema": True}
    errors: list[str] = []
    checks["em_consistency"] = (
        exact_match(protocol.answer, sample.answers) if exploration.success else protocol.answer is None
    )
    if not checks["em_consistency"]:
        errors.append("protocol answer is inconsistent with exploration success or ground truth")

    passages = [passage.contents for search in exploration.searches for passage in search.passages]
    checks["extractive_grounding"] = facts_are_extractive(protocol.grounding_facts, passages)
    if not checks["extractive_grounding"]:
        errors.append("one or more grounding facts are not verbatim passage substrings")

    checks["no_answer_leak"] = not protocol_has_leak(
        protocol.reasoning_plan,
        [search.query for search in exploration.searches],
        sample.answers,
    )
    if not checks["no_answer_leak"]:
        errors.append("answer leakage detected in reasoning plan or retrieval query")

    checks["grounded_flag"] = protocol.answer_grounded == exploration.success
    if not checks["grounded_flag"]:
        errors.append("answer_grounded does not match exploration success")

    checks["variant_shape"] = bool(protocol.grounding_facts) and (
        protocol.partial_findings is None
        if exploration.success
        else bool(protocol.partial_findings) and protocol.answer is None
    )
    if not checks["variant_shape"]:
        errors.append("protocol does not match the succeeded/evidence variant contract")

    passed = all(checks.values())
    return QualityReport(
        example_id=sample.id,
        passed=passed,
        checks=checks,
        errors=errors,
        pi_source="protocol" if passed else "self_rollout_fallback",
    )

