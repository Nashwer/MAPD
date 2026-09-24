import shutil
import json

from mapd.cli import _run_synthesis
from mapd.config import load_config
from mapd.data.schema import QASample
from mapd.io import read_jsonl
from mapd.mas.schema import SynthesisArtifact
from scripts.protocol_bundle import (
    export_protocol_bundle,
    merge_protocol_bundles,
    restore_protocol_bundle,
)


def test_protocol_bundle_round_trip_and_deduplicating_merge(tmp_path):
    samples = read_jsonl("tests/fixtures/qa.jsonl", QASample)
    config = load_config("configs/local_smoke.yaml")
    source = tmp_path / "artifacts/protocol_shards/offset_0_count_2"
    _run_synthesis(config, samples, source)
    usage_record = {
        "role": "orchestrator",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "estimated_cost_usd": 0.000054,
    }
    (source / "teacher_usage.jsonl").write_text(
        json.dumps(usage_record) + "\n", encoding="utf-8"
    )

    bundle = tmp_path / "exports/protocols.tar.gz"
    exported = export_protocol_bundle(
        source,
        bundle,
        project_root=tmp_path,
        dataset_offset=0,
        requested_count=2,
    )
    assert exported["completed_count"] == 2
    assert len(exported["bundle_sha256"]) == 64

    shutil.rmtree(source)
    restored = restore_protocol_bundle(bundle, project_root=tmp_path)
    assert restored["completed_count"] == 2
    assert len(read_jsonl(source / "artifacts.jsonl", SynthesisArtifact)) == 2
    ledger = tmp_path / "artifacts/teacher_usage.jsonl"
    assert json.loads(ledger.read_text(encoding="utf-8"))["role"] == "orchestrator"

    merged = tmp_path / "artifacts/protocol_merged"
    result = merge_protocol_bundles([bundle, bundle], merged)
    assert result["bundle_count"] == 2
    assert result["count"] == 2
    assert len(read_jsonl(merged / "artifacts.jsonl", SynthesisArtifact)) == 2
