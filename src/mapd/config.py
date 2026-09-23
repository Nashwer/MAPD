from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: object) -> object:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


class RunConfig(BaseModel):
    seed: int = 42
    artifact_dir: Path = Path("artifacts/default")
    progress_every: int = Field(default=100, ge=1)


class TeacherConfig(BaseModel):
    backend: Literal["mock", "openai_compatible"] = "mock"
    model: str = "mock-mapd-teacher"
    base_url: str | None = None
    api_key_env: str = "MAPD_TEACHER_API_KEY"
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def validate_real_backend(self) -> "TeacherConfig":
        if self.backend == "openai_compatible" and (not self.base_url or not self.model):
            raise ValueError("openai_compatible teacher requires base_url and model")
        return self


class RetrievalConfig(BaseModel):
    backend: Literal["bm25", "http"] = "bm25"
    corpus_path: Path | None = None
    url: str | None = None
    top_k: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_backend(self) -> "RetrievalConfig":
        if self.backend == "bm25" and self.corpus_path is None:
            raise ValueError("bm25 retrieval requires corpus_path")
        if self.backend == "http" and not self.url:
            raise ValueError("http retrieval requires url")
        return self


class MASConfig(BaseModel):
    max_subquestions: int = Field(default=4, ge=1)
    max_rounds: int = Field(default=2, ge=1)
    max_queries_per_searcher: int = Field(default=3, ge=1)
    max_repair_rounds: int = Field(default=2, ge=0)


class DataConfig(BaseModel):
    training_datasets: list[str] = ["nq", "hotpotqa"]
    evaluation_datasets: list[str] = [
        "nq",
        "triviaqa",
        "popqa",
        "hotpotqa",
        "2wikimultihopqa",
        "musique",
        "bamboogle",
    ]
    retrieval_corpus: str = "wiki-18"


class TrainingConfig(BaseModel):
    model: str = "Qwen/Qwen3-1.7B"
    group_size: int = Field(default=8, ge=1)
    batch_size: int = Field(default=128, ge=1)
    max_prompt_length: int = Field(default=4096, ge=1)
    max_response_length: int = Field(default=512, ge=1)
    max_turns: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=1e-6, gt=0)
    warmup_ratio: float = Field(default=0.1, ge=0, le=1)
    total_steps: int = Field(default=200, ge=1)
    lambda_opsd: float = Field(default=0.05, ge=0)
    clip_low: float = Field(default=0.2, ge=0)
    clip_high: float = Field(default=0.2, ge=0)
    reference_kl_beta: float = Field(default=0.0, ge=0)
    num_gpus: int = Field(default=8, ge=1)


class EvaluationConfig(BaseModel):
    seeds: list[int] = [0, 1, 2, 3, 4]
    metric: Literal["exact_match"] = "exact_match"


class AppConfig(BaseModel):
    run: RunConfig = RunConfig()
    teacher: TeacherConfig = TeacherConfig()
    retrieval: RetrievalConfig
    mas: MASConfig = MASConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(_expand_env(raw))
