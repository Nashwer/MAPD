from __future__ import annotations

from pathlib import Path
from typing import Any

from mapd.environment.search_env import AGENT_SYSTEM_PROMPT
from mapd.mas.schema import SynthesisArtifact


def to_verl_record(artifact: SynthesisArtifact, index: int) -> dict[str, Any]:
    sample = artifact.sample
    protocol = (
        artifact.protocol.model_dump(mode="json")
        if artifact.quality.passed and artifact.protocol is not None
        else None
    )
    return {
        "data_source": sample.data_source,
        "prompt": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": sample.question},
        ],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": sample.answers},
        "extra_info": {
            "split": sample.split,
            "index": index,
            "example_id": sample.id,
            "protocol": protocol,
            "protocol_valid": artifact.quality.passed,
            "pi_source": artifact.quality.pi_source,
        },
    }


def write_verl_parquet(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write records using the nested Arrow schema expected by veRL datasets."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow; install the project data extra or `pip install pyarrow`."
        ) from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), output)

