import json

from mapd.cli import _run_synthesis
from mapd.config import load_config
from mapd.io import read_jsonl
from mapd.data.schema import QASample
from mapd.mas.provider import MockTeacher
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.mas.pipeline import MASPipeline


class RecordingTeacher(MockTeacher):
    def __init__(self):
        self.calls = []

    def generate_json(self, role, payload):
        self.calls.append((role, payload))
        if role == "answerer":
            return {"answer": None}
        return super().generate_json(role, payload)


def test_mock_pipeline_passes_quality_gate():
    config = load_config("configs/local_smoke.yaml")
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)
    pipeline = MASPipeline(
        MockTeacher(),
        BM25Retriever.from_jsonl(config.retrieval.corpus_path),
        config.mas,
        config.retrieval.top_k,
    )
    artifacts = [pipeline.synthesize(sample) for sample in samples]
    assert all(artifact.quality.passed for artifact in artifacts)
    assert all(artifact.quality.pi_source == "protocol" for artifact in artifacts)


def test_synthesis_cache_is_reused(tmp_path):
    config = load_config("configs/local_smoke.yaml")
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)
    _run_synthesis(config, samples, tmp_path)
    _run_synthesis(config, samples, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cache_hits"] == len(samples)


def test_ground_truth_is_confined_to_repair_and_never_reaches_search_or_answerer():
    config = load_config("configs/local_smoke.yaml")
    sample = read_jsonl("tests/fixtures/qa.jsonl", QASample)[0]
    teacher = RecordingTeacher()
    pipeline = MASPipeline(
        teacher,
        BM25Retriever.from_jsonl(config.retrieval.corpus_path),
        config.mas,
        config.retrieval.top_k,
    )

    pipeline.synthesize(sample)

    assert any(role == "repair" and payload["ground_truth"] == sample.answers for role, payload in teacher.calls)
    for role, payload in teacher.calls:
        if role != "repair":
            assert "ground_truth" not in payload
            assert "answers" not in payload
