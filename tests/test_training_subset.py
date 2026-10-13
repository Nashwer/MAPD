import json

from mapd.data.schema import QASample
from mapd.io import read_jsonl
from scripts.sample_training_data import sample_training_subset


def _write_samples(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for source in ("nq", "hotpotqa"):
            for index in range(10):
                handle.write(
                    QASample(
                        id=f"{source}:{index}",
                        question=f"{source} question {index}",
                        answers=[str(index)],
                        data_source=source,
                    ).model_dump_json()
                    + "\n"
                )


def test_training_subset_is_deterministic_balanced_and_manifested(tmp_path):
    source = tmp_path / "full.jsonl"
    _write_samples(source)

    manifests = []
    outputs = []
    for suffix in ("a", "b"):
        output = tmp_path / f"subset-{suffix}.jsonl"
        manifest = tmp_path / f"subset-{suffix}.manifest.json"
        manifests.append(
            sample_training_subset(
                source, output, manifest, fraction=0.3, seed=42
            )
        )
        outputs.append(read_jsonl(output, QASample))

    assert [sample.id for sample in outputs[0]] == [sample.id for sample in outputs[1]]
    assert len(outputs[0]) == 6
    assert manifests[0]["selected_counts"] == {"hotpotqa": 3, "nq": 3}
    saved = json.loads((tmp_path / "subset-a.manifest.json").read_text("utf-8"))
    assert saved["output_sha256"] == manifests[0]["output_sha256"]
