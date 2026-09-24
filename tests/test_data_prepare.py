from mapd.data.prepare import (
    canonical_question,
    deduplicate_training_samples,
    normalize_record,
    prepare_searchr1_dataset,
    select_balanced_subset,
)
from mapd.data.schema import QASample


def test_searchr1_record_normalization():
    sample = normalize_record(
        {
            "id": "train_7",
            "question": " Who directed The Winter Guest? ",
            "golden_answers": ["Alan Rickman", "Alan Rickman"],
            "data_source": "hotpot_qa",
            "extra_info": {"split": "train"},
        },
        default_split="train",
    )
    assert sample.id == "hotpotqa:train_7"
    assert sample.question == "Who directed The Winter Guest?"
    assert sample.answers == ["Alan Rickman"]
    assert canonical_question(sample.question) == "who directed the winter guest"


def test_deduplication_and_heldout_separation():
    samples = [
        QASample(id="1", question="Same question?", answers=["a"], data_source="nq"),
        QASample(id="2", question="same question", answers=["a"], data_source="hotpotqa"),
        QASample(id="3", question="Held out!", answers=["b"], data_source="nq"),
        QASample(id="4", question="Keep me", answers=["c"], data_source="hotpotqa"),
    ]
    kept, counts = deduplicate_training_samples(
        samples, heldout_questions={canonical_question("held out")}
    )
    assert [sample.id for sample in kept] == ["1", "4"]
    assert counts == {"duplicates_removed": 1, "heldout_overlaps_removed": 1}


def test_balanced_subset_is_deterministic():
    samples = [
        QASample(id=f"nq-{index}", question=f"nq {index}", answers=["a"], data_source="nq")
        for index in range(10)
    ] + [
        QASample(
            id=f"hp-{index}",
            question=f"hotpot {index}",
            answers=["b"],
            data_source="hotpotqa",
        )
        for index in range(10)
    ]
    first = select_balanced_subset(samples, target_size=8, seed=17)
    second = select_balanced_subset(samples, target_size=8, seed=17)
    assert [sample.id for sample in first] == [sample.id for sample in second]
    assert sum(sample.data_source == "nq" for sample in first) == 4
    assert sum(sample.data_source == "hotpotqa" for sample in first) == 4


def test_parquet_pipeline_writes_manifest_and_separates_heldout(tmp_path):
    pytest = __import__("pytest")
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    train = tmp_path / "train.parquet"
    heldout = tmp_path / "test.parquet"
    rows = [
        {
            "id": f"row-{index}",
            "question": f"Question {index}?",
            "golden_answers": [f"answer {index}"],
            "data_source": "nq" if index % 2 == 0 else "hotpotqa",
        }
        for index in range(8)
    ]
    parquet.write_table(pyarrow.Table.from_pylist(rows), train)
    parquet.write_table(pyarrow.Table.from_pylist([rows[0]]), heldout)
    manifest = prepare_searchr1_dataset(
        train,
        tmp_path / "output",
        heldout_parquet=heldout,
        target_size=6,
        seed=9,
        source_revision="test-revision",
    )
    assert manifest["heldout_overlaps_removed"] == 1
    assert manifest["target_size"] == 6
    assert manifest["source_revision"] == "test-revision"
    output = tmp_path / "output" / "training" / "mapd_train_6.jsonl"
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6
