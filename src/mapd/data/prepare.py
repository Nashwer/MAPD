from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from mapd.data.schema import QASample
from mapd.io import write_jsonl


_SPACE_PATTERN = re.compile(r"\s+")
_QUESTION_KEY_PATTERN = re.compile(r"[^\w]+", re.UNICODE)
SUPPORTED_SOURCES = ("nq", "hotpotqa")


def canonical_question(question: str) -> str:
    """Return a stable key used only for leakage checks and de-duplication."""
    normalized = unicodedata.normalize("NFKC", question).casefold().strip()
    return _QUESTION_KEY_PATTERN.sub(" ", normalized).strip()


def normalize_record(record: Mapping[str, Any], *, default_split: str) -> QASample:
    question = _first_text(record.get("question"))
    if not question:
        question = _question_from_prompt(record.get("prompt"))
    if not question:
        raise ValueError("record has no usable question")

    answers = _answer_list(record.get("golden_answers"))
    if not answers:
        answers = _answer_list(record.get("answers"))
    if not answers:
        answers = _answer_list(record.get("answer"))
    if not answers:
        reward_model = record.get("reward_model")
        if isinstance(reward_model, Mapping):
            ground_truth = reward_model.get("ground_truth")
            if isinstance(ground_truth, Mapping):
                answers = _answer_list(ground_truth.get("target"))
            else:
                answers = _answer_list(ground_truth)
    if not answers:
        raise ValueError("record has no usable answer")

    source = str(record.get("data_source") or record.get("source") or "unknown").lower()
    if "hotpot" in source:
        source = "hotpotqa"
    elif source in {"natural_questions", "naturalquestions"} or source.startswith("nq"):
        source = "nq"
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"unsupported data source: {source!r}")

    split = default_split
    extra_info = record.get("extra_info")
    if isinstance(extra_info, Mapping) and extra_info.get("split"):
        split = str(extra_info["split"])
    elif record.get("split"):
        split = str(record["split"])

    raw_id = record.get("id") or record.get("example_id")
    if raw_id is None and isinstance(extra_info, Mapping):
        raw_id = extra_info.get("index")
    if raw_id is None:
        raw_id = hashlib.sha256(canonical_question(question).encode("utf-8")).hexdigest()[:16]
    return QASample(
        id=f"{source}:{raw_id}",
        question=_SPACE_PATTERN.sub(" ", question).strip(),
        answers=answers,
        split=split,
        data_source=source,
    )


def iter_parquet_records(path: str | Path, *, batch_size: int = 8192) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - covered on the GPU server
        raise RuntimeError("pyarrow is required; run `bash mapd.sh bootstrap` first") from exc

    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    useful = [
        name
        for name in (
            "id",
            "question",
            "golden_answers",
            "data_source",
            "prompt",
            "reward_model",
            "extra_info",
            "split",
        )
        if name in available
    ]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=useful):
        yield from batch.to_pylist()


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    default_split: str,
) -> tuple[list[QASample], Counter[str]]:
    samples: list[QASample] = []
    rejected: Counter[str] = Counter()
    for record in records:
        try:
            samples.append(normalize_record(record, default_split=default_split))
        except ValueError as exc:
            rejected[str(exc)] += 1
    return samples, rejected


def deduplicate_training_samples(
    samples: Iterable[QASample],
    *,
    heldout_questions: set[str] | None = None,
) -> tuple[list[QASample], dict[str, int]]:
    heldout = heldout_questions or set()
    seen: set[str] = set()
    kept: list[QASample] = []
    duplicate_count = 0
    leakage_count = 0
    for sample in samples:
        key = canonical_question(sample.question)
        if key in heldout:
            leakage_count += 1
            continue
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        kept.append(sample.model_copy(update={"split": "train"}))
    return kept, {
        "duplicates_removed": duplicate_count,
        "heldout_overlaps_removed": leakage_count,
    }


def select_balanced_subset(
    samples: Iterable[QASample],
    *,
    target_size: int,
    seed: int,
) -> list[QASample]:
    if target_size < 1:
        raise ValueError("target_size must be positive")
    grouped: dict[str, list[QASample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.data_source].append(sample)
    if not grouped:
        raise ValueError("no training samples remain after filtering")

    sources = sorted(grouped)
    rng = random.Random(seed)
    for source in sources:
        rng.shuffle(grouped[source])

    base, remainder = divmod(target_size, len(sources))
    selected: list[QASample] = []
    unused: list[QASample] = []
    for index, source in enumerate(sources):
        quota = base + (1 if index < remainder else 0)
        take = min(quota, len(grouped[source]))
        selected.extend(grouped[source][:take])
        unused.extend(grouped[source][take:])
    if len(selected) < target_size:
        rng.shuffle(unused)
        selected.extend(unused[: target_size - len(selected)])
    if len(selected) != target_size:
        raise ValueError(
            f"requested {target_size} samples but only {len(selected)} unique samples are available"
        )
    rng.shuffle(selected)
    return selected


def prepare_searchr1_dataset(
    train_parquet: str | Path,
    output_root: str | Path,
    *,
    heldout_parquet: str | Path | None = None,
    evaluation_jsonl: Iterable[str | Path] = (),
    target_size: int = 25_600,
    seed: int = 42,
    source_revision: str | None = None,
) -> dict[str, Any]:
    output = Path(output_root)
    train_samples, train_rejected = normalize_records(
        iter_parquet_records(train_parquet), default_split="train"
    )

    heldout_samples: list[QASample] = []
    heldout_rejected: Counter[str] = Counter()
    if heldout_parquet is not None:
        heldout_samples, heldout_rejected = normalize_records(
            iter_parquet_records(heldout_parquet), default_split="test"
        )

    heldout_questions = {canonical_question(sample.question) for sample in heldout_samples}
    for jsonl_path in evaluation_jsonl:
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                question = _first_text(record.get("question"))
                if question:
                    heldout_questions.add(canonical_question(question))

    deduplicated, filter_counts = deduplicate_training_samples(
        train_samples, heldout_questions=heldout_questions
    )
    subset = select_balanced_subset(deduplicated, target_size=target_size, seed=seed)

    by_source: dict[str, list[QASample]] = defaultdict(list)
    for sample in deduplicated:
        by_source[sample.data_source].append(sample)
    for source in SUPPORTED_SOURCES:
        write_jsonl(output / source / "train.jsonl", by_source.get(source, []))
    write_jsonl(output / "evaluation" / "searchr1_test.jsonl", heldout_samples)
    subset_path = output / "training" / f"mapd_train_{target_size}.jsonl"
    write_jsonl(subset_path, subset)

    manifest = {
        "schema": "mapd-data-v1",
        "source": "PeterJinGo/nq_hotpotqa_train",
        "source_revision": source_revision,
        "seed": seed,
        "target_size": target_size,
        "input_counts": dict(Counter(sample.data_source for sample in train_samples)),
        "deduplicated_counts": dict(Counter(sample.data_source for sample in deduplicated)),
        "selected_counts": dict(Counter(sample.data_source for sample in subset)),
        "heldout_count": len(heldout_samples),
        "heldout_question_keys": len(heldout_questions),
        **filter_counts,
        "train_rejected": dict(train_rejected),
        "heldout_rejected": dict(heldout_rejected),
        "output": str(subset_path),
    }
    manifest_path = output / "training" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _first_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _question_from_prompt(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for message in reversed(value):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = _first_text(message.get("content"))
            marker = "Question:"
            if marker in content:
                return content.rsplit(marker, 1)[-1].strip()
            return content
    return ""


def _answer_list(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        return []
    answers: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        answer = _SPACE_PATTERN.sub(" ", item).strip()
        if answer and answer not in seen:
            seen.add(answer)
            answers.append(answer)
    return answers
