from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import typer

from mapd.config import AppConfig, load_config
from mapd.environment.schema import AgentTrajectory
from mapd.environment.search_env import AgenticSearchEnvironment, ScriptedPolicy
from mapd.data.verl import to_verl_record, write_verl_parquet
from mapd.evaluation.qa_eval import evaluate_trajectories
from mapd.io import read_jsonl, stable_hash, write_jsonl
from mapd.data.schema import QASample
from mapd.mas.pipeline import MASPipeline
from mapd.mas.provider import MockTeacher, OpenAICompatibleTeacher, TeacherClient
from mapd.mas.schema import SynthesisArtifact
from mapd.protocol.validator import validate_protocol
from mapd.retrieval.base import Retriever
from mapd.retrieval.retriever_server import HTTPRetriever
from mapd.retrieval.wiki18 import BM25Retriever
from mapd.trainer.grpo import RolloutLossInput
from mapd.trainer.mapd_trainer import compute_mapd_loss
from mapd.trainer.opsd import select_privileged_information


app = typer.Typer(no_args_is_help=True, help="MAPD reproduction utilities")
ARTIFACT_SCHEMA_VERSION = 3
SYNTHESIS_REVISION = 8


def _build_teacher(
    config: AppConfig, *, usage_log_path: Path | None = None
) -> TeacherClient:
    if config.teacher.backend == "mock":
        return MockTeacher()
    return OpenAICompatibleTeacher(
        model=config.teacher.model,
        base_url=config.teacher.base_url or "",
        api_key_env=config.teacher.api_key_env,
        timeout_seconds=config.teacher.timeout_seconds,
        max_retries=config.teacher.max_retries,
        thinking_mode=config.teacher.thinking_mode,
        max_output_tokens=config.teacher.max_output_tokens,
        budget_usd=config.teacher.budget_usd,
        input_price_per_million=config.teacher.input_price_per_million,
        output_price_per_million=config.teacher.output_price_per_million,
        usage_log_path=usage_log_path,
        budget_ledger_path=Path("artifacts/teacher_usage.jsonl"),
    )


def _build_retriever(config: AppConfig) -> Retriever:
    if config.retrieval.backend == "bm25":
        return BM25Retriever.from_jsonl(config.retrieval.corpus_path or "")
    return HTTPRetriever(config.retrieval.url or "")


def _run_synthesis(config: AppConfig, samples: list[QASample], output_dir: Path) -> list[SynthesisArtifact]:
    teacher = _build_teacher(
        config, usage_log_path=output_dir / "teacher_usage.jsonl"
    )
    pipeline = MASPipeline(
        teacher=teacher,
        retriever=_build_retriever(config),
        config=config.mas,
        top_k=config.retrieval.top_k,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Operational controls such as retries, price tables and the spend limit
    # must not invalidate completed protocols. In particular, raising a budget
    # after a safe stop should resume instead of paying to synthesize them again.
    config_hash = stable_hash(
        {
            "seed": config.run.seed,
            "teacher": {
                "backend": config.teacher.backend,
                "model": config.teacher.model,
                "base_url": config.teacher.base_url,
                "thinking_mode": config.teacher.thinking_mode,
                "max_output_tokens": config.teacher.max_output_tokens,
            },
            "retrieval": config.retrieval.model_dump(mode="json"),
            "mas": config.mas.model_dump(mode="json"),
        }
    )
    artifact_path = output_dir / "artifacts.jsonl"
    cached_by_id: dict[str, SynthesisArtifact] = {}
    if artifact_path.exists():
        try:
            for cached in read_jsonl(artifact_path, SynthesisArtifact):
                cached_by_id[cached.sample.id] = cached
        except (ValueError, TypeError):
            # Older development artifacts are intentionally invalidated when
            # the on-disk schema changes.
            cached_by_id = {}

    artifacts: list[SynthesisArtifact] = []
    cache_hits = 0
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        sample_hash = stable_hash(sample)
        cached = cached_by_id.get(sample.id)
        if (
            cached
            and cached.metadata.get("artifact_schema_version") == ARTIFACT_SCHEMA_VERSION
            and cached.metadata.get("synthesis_revision") == SYNTHESIS_REVISION
            and cached.metadata.get("sample_hash") == sample_hash
            and cached.metadata.get("config_hash") == config_hash
        ):
            artifacts.append(cached)
            cache_hits += 1
        else:
            artifact = pipeline.synthesize(sample)
            artifact.metadata.update(
                {
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "synthesis_revision": SYNTHESIS_REVISION,
                    "sample_hash": sample_hash,
                    "config_hash": config_hash,
                }
            )
            artifacts.append(artifact)

        if index % config.run.progress_every == 0 or index == total:
            # Rewriting the checkpoint keeps JSONL valid if an SSH session or
            # teacher request is interrupted between progress boundaries.
            write_jsonl(artifact_path, artifacts)
            progress = {
                "event": "synthesis_progress",
                "processed": index,
                "total": total,
                "passed": sum(item.quality.passed for item in artifacts),
                "cache_hits": cache_hits,
                "checkpoint": str(artifact_path),
                "time": datetime.now(timezone.utc).isoformat(),
            }
            if isinstance(teacher, OpenAICompatibleTeacher):
                progress["teacher_usage"] = teacher.usage_summary()
            typer.echo(json.dumps(progress, ensure_ascii=False))

    write_jsonl(artifact_path, artifacts)
    write_jsonl(output_dir / "explorations.jsonl", [artifact.exploration for artifact in artifacts])
    write_jsonl(
        output_dir / "protocols.jsonl",
        [artifact.protocol if artifact.protocol is not None else artifact.raw_protocol for artifact in artifacts],
    )
    write_jsonl(output_dir / "quality.jsonl", [artifact.quality for artifact in artifacts])
    write_jsonl(output_dir / "training.jsonl", [to_verl_record(artifact, index) for index, artifact in enumerate(artifacts)])
    manifest = {
        "count": len(artifacts),
        "passed": sum(artifact.quality.passed for artifact in artifacts),
        "exploration_success": sum(artifact.exploration.success for artifact in artifacts),
        "succeeded_protocols": sum(
            artifact.protocol is not None and artifact.protocol.answer_grounded
            for artifact in artifacts
        ),
        "evidence_protocols": sum(
            artifact.protocol is not None and not artifact.protocol.answer_grounded
            for artifact in artifacts
        ),
        "task_types": dict(
            sorted(Counter(artifact.exploration.task_type.value for artifact in artifacts).items())
        ),
        "searches": sum(len(artifact.exploration.searches) for artifact in artifacts),
        "retrieved_passages": sum(
            len(search.passages)
            for artifact in artifacts
            for search in artifact.exploration.searches
        ),
        "findings": sum(len(artifact.exploration.findings) for artifact in artifacts),
        "nonempty_findings": sum(
            bool(finding.summary.strip())
            for artifact in artifacts
            for finding in artifact.exploration.findings
        ),
        "cache_hits": cache_hits,
        "seed": config.run.seed,
        "config_hash": config_hash,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "synthesis_revision": SYNTHESIS_REVISION,
    }
    if isinstance(teacher, OpenAICompatibleTeacher):
        manifest["teacher_usage"] = teacher.usage_summary()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifacts


@app.command()
def doctor(
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True),
    require_gpu: bool = typer.Option(False, "--require-gpu", help="Fail unless CUDA is usable."),
) -> None:
    """Check the lightweight runtime and configured external services."""
    settings = load_config(config)
    torch_version = None
    cuda_available = False
    cuda_device = None
    if importlib.util.find_spec("torch") is not None:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None

    checks = {
        "python": platform.python_version(),
        "python_supported": (3, 10) <= sys.version_info[:2] < (3, 14),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "verl_importable": importlib.util.find_spec("verl") is not None,
        "vllm_importable": importlib.util.find_spec("vllm") is not None,
        "teacher_backend": settings.teacher.backend,
        "teacher_ready": settings.teacher.backend == "mock" or bool(os.getenv(settings.teacher.api_key_env)),
        "retrieval_backend": settings.retrieval.backend,
        "retrieval_ready": settings.retrieval.backend == "http"
        or bool(settings.retrieval.corpus_path and settings.retrieval.corpus_path.exists()),
        "gpu_required": require_gpu,
        "training_environment": f"{platform.system()} Python {platform.python_version()}",
    }
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))
    required = [checks["python_supported"], checks["teacher_ready"], checks["retrieval_ready"]]
    if require_gpu:
        required.extend((checks["verl_importable"], checks["vllm_importable"], checks["cuda_available"]))
    if not all(required):
        raise typer.Exit(code=1)


@app.command()
def synthesize(
    input_path: Path = typer.Option(..., "--input", exists=True),
    output_dir: Path = typer.Option(..., "--output"),
    config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True),
) -> None:
    """Run offline MAS protocol synthesis."""
    artifacts = _run_synthesis(load_config(config), read_jsonl(input_path, QASample), output_dir)
    passed = sum(artifact.quality.passed for artifact in artifacts)
    typer.echo(f"Synthesized {len(artifacts)} samples; quality gate passed: {passed}/{len(artifacts)}")


@app.command()
def validate(artifacts_path: Path = typer.Option(..., "--artifacts", exists=True)) -> None:
    """Re-run deterministic quality checks for synthesis artifacts."""
    artifacts = read_jsonl(artifacts_path, SynthesisArtifact)
    reports = [validate_protocol(item.sample, item.exploration, item.protocol) for item in artifacts]
    failed = [report for report in reports if not report.passed]
    typer.echo(json.dumps({"count": len(reports), "passed": len(reports) - len(failed)}, indent=2))
    if failed:
        raise typer.Exit(code=1)


@app.command("prepare-data")
def prepare_data(
    artifacts_path: Path = typer.Option(..., "--artifacts", exists=True),
    output_path: Path = typer.Option(..., "--output"),
) -> None:
    """Convert synthesis artifacts to veRL-compatible JSONL or Parquet."""
    artifacts = read_jsonl(artifacts_path, SynthesisArtifact)
    records = [to_verl_record(item, index) for index, item in enumerate(artifacts)]
    if output_path.suffix.lower() == ".parquet":
        write_verl_parquet(output_path, records)
    elif output_path.suffix.lower() == ".jsonl":
        write_jsonl(output_path, records)
    else:
        raise typer.BadParameter("output must end in .jsonl or .parquet", param_hint="--output")
    typer.echo(f"Wrote {len(artifacts)} training records to {output_path}")


@app.command("evaluate-trajectories")
def evaluate_trajectory_file(
    input_path: Path = typer.Option(..., "--input", exists=True),
) -> None:
    """Aggregate strict-EM results from saved agent trajectories."""
    trajectories = read_jsonl(input_path, AgentTrajectory)
    summary = evaluate_trajectories(trajectories)
    typer.echo(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


@app.command("status")
def artifact_status(
    artifact_dir: Path = typer.Option(Path("artifacts/smoke"), "--artifact-dir"),
) -> None:
    """Show machine-readable progress for a synthesis artifact directory."""
    manifest_path = artifact_dir / "manifest.json"
    artifact_path = artifact_dir / "artifacts.jsonl"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_count = 0
    if artifact_path.exists():
        with artifact_path.open("r", encoding="utf-8") as handle:
            artifact_count = sum(bool(line.strip()) for line in handle)
    payload = {
        "artifact_dir": str(artifact_dir),
        "artifact_count": artifact_count,
        "complete": manifest is not None,
        "manifest": manifest,
        "last_modified": (
            datetime.fromtimestamp(artifact_path.stat().st_mtime, tz=timezone.utc).isoformat()
            if artifact_path.exists()
            else None
        ),
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def smoke(config: Path = typer.Option(Path("configs/local_smoke.yaml"), exists=True)) -> None:
    """Run offline synthesis and the online MAPD data/loss path on CPU mocks."""
    settings = load_config(config)
    samples = read_jsonl(Path("tests/fixtures/qa.jsonl"), QASample)
    artifacts = _run_synthesis(settings, samples, settings.run.artifact_dir)
    failed = [artifact.quality for artifact in artifacts if not artifact.quality.passed]
    environment = AgenticSearchEnvironment(
        _build_retriever(settings),
        top_k=settings.retrieval.top_k,
        max_turns=settings.training.max_turns,
    )
    online_steps = []
    for artifact in artifacts:
        sample = artifact.sample
        trajectories = [
            environment.rollout(
                sample,
                ScriptedPolicy([f"<search>{sample.question}</search>", f"<answer>{sample.answers[0]}</answer>"]),
            ),
            environment.rollout(sample, ScriptedPolicy(["<answer>intentionally wrong</answer>"])),
        ][: settings.training.group_size]
        while len(trajectories) < settings.training.group_size:
            trajectories.append(environment.rollout(sample, ScriptedPolicy(["<answer>intentionally wrong</answer>"])))
        pi = select_privileged_information(artifact, trajectories)
        student_distribution = [math.log(0.6), math.log(0.4)]
        privileged_distribution = [math.log(0.7), math.log(0.3)]
        loss_inputs = [
            RolloutLossInput(
                reward=trajectory.reward,
                old_action_log_probs=[math.log(0.6)],
                new_action_log_probs=[math.log(0.6)],
                reference_token_kls=[0.0],
                student_log_distributions=[student_distribution],
                privileged_log_distributions=[privileged_distribution] if pi else None,
                action_mask=[True],
            )
            for trajectory in trajectories
        ]
        loss = compute_mapd_loss(loss_inputs, lambda_opsd=settings.training.lambda_opsd)
        online_steps.append(
            {
                "example_id": sample.id,
                "rewards": [item.reward for item in trajectories],
                "pi_source": pi.source if pi else "disabled",
                "grpo_loss": loss.grpo,
                "opsd_loss": loss.opsd,
                "joint_loss": loss.total,
            }
        )
    (settings.run.artifact_dir / "online_smoke.json").write_text(
        json.dumps(online_steps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "samples": len(artifacts),
        "passed": len(artifacts) - len(failed),
        "online_training_steps": len(online_steps),
        "student_rollouts": sum(len(item["rewards"]) for item in online_steps),
        "pi_sources": sorted({item["pi_source"] for item in online_steps}),
        "artifact_dir": str(settings.run.artifact_dir),
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    if failed:
        raise typer.Exit(code=1)
