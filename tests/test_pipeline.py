import json
import threading
import time

import pytest

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


def test_synthesis_uses_configured_sample_workers(tmp_path, monkeypatch):
    config = load_config("configs/local_smoke.yaml")
    config.run.synthesis_workers = 2
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)
    active = 0
    peak = 0
    lock = threading.Lock()
    original = MASPipeline.synthesize

    def recording_synthesize(self, sample):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        try:
            return original(self, sample)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(MASPipeline, "synthesize", recording_synthesize)
    _run_synthesis(config, samples, tmp_path)

    assert peak == 2


def test_synthesis_records_sample_rejection_and_continues(tmp_path, monkeypatch):
    config = load_config("configs/local_smoke.yaml")
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)
    original = MASPipeline.synthesize

    def reject_one(self, sample):
        if sample.id == "smoke-1":
            raise ValueError("sample-local leak guard rejection")
        return original(self, sample)

    monkeypatch.setattr(MASPipeline, "synthesize", reject_one)
    artifacts = _run_synthesis(config, samples, tmp_path)

    failures = [
        json.loads(line)
        for line in (tmp_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [artifact.sample.id for artifact in artifacts] == ["smoke-2"]
    assert failures[0]["example_id"] == "smoke-1"
    assert failures[0]["error_type"] == "ValueError"
    assert manifest["requested_count"] == 2
    assert manifest["count"] == 1
    assert manifest["failed"] == 1


def test_synthesis_still_stops_on_operational_failure(tmp_path, monkeypatch):
    config = load_config("configs/local_smoke.yaml")
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)

    def fail_service(self, sample):
        raise RuntimeError("teacher API unavailable")

    monkeypatch.setattr(MASPipeline, "synthesize", fail_service)
    with pytest.raises(RuntimeError, match="teacher API unavailable"):
        _run_synthesis(config, samples, tmp_path)


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
