from mapd.environment.search_env import ScriptedPolicy, AgenticSearchEnvironment
from mapd.config import load_config
from mapd.io import read_jsonl
from mapd.data.schema import QASample
from mapd.mas.provider import MockTeacher
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.mas.pipeline import MASPipeline
from mapd.trainer.opsd import select_privileged_information


def test_valid_protocol_is_preferred_over_self_rollout():
    config = load_config("configs/local_smoke.yaml")
    sample = read_jsonl("tests/fixtures/qa.jsonl", QASample)[0]
    retriever = BM25Retriever.from_jsonl(config.retrieval.corpus_path)
    artifact = MASPipeline(MockTeacher(), retriever, config.mas, config.retrieval.top_k).synthesize(sample)
    trajectory = AgenticSearchEnvironment(retriever, max_turns=1).rollout(
        sample, ScriptedPolicy([f"<answer>{sample.answers[0]}</answer>"])
    )

    pi = select_privileged_information(artifact, [trajectory])
    assert pi is not None
    assert pi.source == "protocol"


def test_invalid_protocol_falls_back_only_to_correct_student_rollout():
    config = load_config("configs/local_smoke.yaml")
    sample = read_jsonl("tests/fixtures/qa.jsonl", QASample)[0]
    retriever = BM25Retriever.from_jsonl(config.retrieval.corpus_path)
    artifact = MASPipeline(MockTeacher(), retriever, config.mas, config.retrieval.top_k).synthesize(sample)
    artifact.quality.passed = False
    good = AgenticSearchEnvironment(retriever, max_turns=1).rollout(
        sample, ScriptedPolicy([f"<answer>{sample.answers[0]}</answer>"])
    )
    bad = AgenticSearchEnvironment(retriever, max_turns=1).rollout(
        sample, ScriptedPolicy(["<answer>wrong</answer>"])
    )

    assert select_privileged_information(artifact, [bad]) is None
    fallback = select_privileged_information(artifact, [bad, good])
    assert fallback is not None
    assert fallback.source == "self_rollout"
