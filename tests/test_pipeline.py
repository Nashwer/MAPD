import json
import threading
import time

import pytest

from mapd.cli import _run_synthesis
from mapd.config import load_config
from mapd.io import read_jsonl
from mapd.data.schema import QASample
from mapd.mas.provider import MockTeacher
from mapd.mas.schema import SearchFinding, SearchRecord
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.retrieval.schema import RetrievedPassage
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


class ExpressionRepairTeacher:
    model = "expression-repair-teacher"

    def __init__(self):
        self.calls = []

    def generate_json(self, role, payload):
        self.calls.append((role, payload))
        if role == "orchestrator":
            return {
                "task_type": "single_hop",
                "subtasks": [
                    {
                        "id": "s1",
                        "objective": "Identify the complete plate-boundary type",
                        "depends_on": [],
                    }
                ],
            }
        if role == "searcher":
            return {"queries": ["East African Rift plate boundary type"]}
        if role == "search_summarizer":
            return {
                "summary": "The East African Rift is a divergent tectonic plate boundary.",
                "evidence_ids": ["p1"],
            }
        if role == "answerer":
            if "revision_instruction" in payload:
                return {"answer": "divergent tectonic plate boundary"}
            return {"answer": "divergent"}
        if role == "repair":
            return {
                "failure_type": "expression",
                "diagnosis": "The candidate omitted the complete relation name.",
                "subtasks": [],
            }
        if role == "protocolizer":
            return {
                "task_type": "single_hop",
                "reasoning_plan": ["Identify the complete plate-boundary type."],
                "grounding_facts": [
                    "The East African Rift is a divergent tectonic plate boundary."
                ],
                "answer": payload["candidate_answer"],
                "answer_grounded": True,
            }
        raise AssertionError(f"unexpected role: {role}")


class FixedPassageRetriever:
    def search(self, query, top_k):
        return [
            RetrievedPassage(
                id="p1",
                contents="The East African Rift is a divergent tectonic plate boundary.",
                score=1.0,
            )
        ]


def test_expression_repair_reanswers_without_redundant_retrieval():
    config = load_config("configs/local_smoke.yaml")
    config.mas.max_rounds = 1
    config.mas.max_repair_rounds = 2
    teacher = ExpressionRepairTeacher()
    pipeline = MASPipeline(teacher, FixedPassageRetriever(), config.mas, top_k=1)
    sample = QASample(
        id="expression-1",
        question="What type of plate boundary is the East African Rift?",
        answers=["divergent tectonic plate boundary"],
    )

    artifact = pipeline.synthesize(sample)

    assert artifact.exploration.success is True
    assert artifact.exploration.candidate_answer == "divergent tectonic plate boundary"
    assert artifact.exploration.repair_rounds == 1
    assert len(artifact.exploration.searches) == 1
    answer_payloads = [payload for role, payload in teacher.calls if role == "answerer"]
    assert len(answer_payloads) == 2
    assert answer_payloads[1]["revision_instruction"] == (
        "The candidate omitted the complete relation name."
    )
    assert all("ground_truth" not in payload for payload in answer_payloads)


def test_answerer_receives_only_unique_passages_cited_by_findings():
    config = load_config("configs/local_smoke.yaml")
    teacher = RecordingTeacher()
    pipeline = MASPipeline(teacher, FixedPassageRetriever(), config.mas, top_k=1)
    cited = RetrievedPassage(id="p1", contents="Useful evidence.", score=1.0)
    irrelevant = RetrievedPassage(id="p2", contents="Distracting evidence.", score=0.5)
    searches = [
        SearchRecord(subtask_id="s1", query="first", passages=[cited, irrelevant]),
        SearchRecord(subtask_id="s2", query="second", passages=[cited]),
    ]
    findings = [
        SearchFinding(
            subtask_id="s1", summary="Useful supported fact.", evidence_ids=["p1"]
        )
    ]

    pipeline._answer("What is the answer?", searches, findings)

    payload = next(payload for role, payload in teacher.calls if role == "answerer")
    assert [passage["id"] for passage in payload["passages"]] == ["p1"]
