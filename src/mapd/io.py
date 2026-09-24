from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


def read_jsonl(path: str | Path, model: type[ModelT]) -> list[ModelT]:
    records: list[ModelT] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)


def stable_hash(value: BaseModel | dict) -> str:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
