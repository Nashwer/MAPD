import json
import tarfile

import pytest

from scripts.data_bundle import export_data_bundle


def _write_fixture(root, *, count=25_600):
    (root / "training").mkdir(parents=True)
    (root / "evaluation").mkdir(parents=True)
    train = root / "training" / "mapd_train_25600.jsonl"
    with train.open("w", encoding="utf-8") as handle:
        for index in range(count):
            source = "nq" if index < 12_800 else "hotpotqa"
            record = {
                "id": f"{source}:train_{index}",
                "question": f"Question {index}?",
                "answers": [f"Answer {index}"],
                "split": "train",
                "data_source": source,
            }
            handle.write(json.dumps(record) + "\n")
    manifest = {
        "schema": "mapd-data-v1",
        "source": "PeterJinGo/nq_hotpotqa_train",
        "source_revision": "fixed-revision",
        "seed": 42,
        "target_size": 25_600,
        "selected_counts": {"nq": 12_800, "hotpotqa": 12_800},
    }
    (root / "training" / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (root / "evaluation" / "searchr1_test.jsonl").write_text(
        json.dumps(
            {
                "id": "nq:test_0",
                "question": "Held out?",
                "answers": ["Yes"],
                "split": "test",
                "data_source": "nq",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_data_bundle_validates_and_exports(tmp_path):
    data = tmp_path / "data"
    _write_fixture(data)
    bundle = tmp_path / "dataset.tar.gz"

    result = export_data_bundle(
        data, bundle, expected_revision="fixed-revision"
    )

    assert result["training_examples"] == 25_600
    assert result["selected_counts"] == {"hotpotqa": 12_800, "nq": 12_800}
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
        assert "bundle-manifest.json" in names
        assert "payload/data/training/mapd_train_25600.jsonl" in names
        assert "payload/data/evaluation/searchr1_test.jsonl" in names
        assert "payload/data/README.txt" in names


def test_data_bundle_rejects_incomplete_training_set(tmp_path):
    data = tmp_path / "data"
    _write_fixture(data, count=1)

    with pytest.raises(ValueError, match="expected 25600 training records"):
        export_data_bundle(data, tmp_path / "dataset.tar.gz")
