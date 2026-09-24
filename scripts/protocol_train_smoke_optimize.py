#!/usr/bin/env python3
"""Run a small real MAPD training loop over one generated protocol shard."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mapd.environment.schema import AgentTrajectory
from mapd.io import read_jsonl
from mapd.mas.schema import SynthesisArtifact
from mapd.trainer.opsd import select_privileged_information
from mapd.trainer.single_device import (
    enable_last_decoder_layer,
    load_smoke_checkpoint,
    optimize_replay_group,
    save_smoke_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize Qwen on real rollouts and generated MAPD protocols"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lambda-opsd", type=float, default=0.05)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.2)
    parser.add_argument("--reference-kl-beta", type=float, default=0.001)
    parser.add_argument("--max-sequence-length", type=int, default=4608)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the protocol training smoke test")
    artifacts = read_jsonl(args.artifacts, SynthesisArtifact)
    trajectories = read_jsonl(args.rollouts, AgentTrajectory)
    groups = _group_rollouts(trajectories)
    _validate_inputs(artifacts, groups)

    print(f"loading train model in float32: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    trainable = enable_last_decoder_layer(model)
    reference_model = None
    if args.reference_kl_beta > 0:
        print(
            f"loading frozen reference model in bfloat16 (beta={args.reference_kl_beta}): "
            f"{args.model}",
            flush=True,
        )
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda")
        reference_model.config.use_cache = False
        reference_model.eval()
        reference_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable),
        lr=args.learning_rate,
        foreach=False,
    )

    history: list[dict[str, Any]] = []
    optimizer_step = 0
    last_result = None
    for artifact in artifacts:
        group = groups[artifact.sample.id]
        privileged = select_privileged_information(artifact, group)
        rewards = [item.reward for item in group]
        mixed_rewards = len(set(rewards)) > 1
        if privileged is None and not mixed_rewards:
            record = {
                "example_id": artifact.sample.id,
                "protocol_passed": artifact.quality.passed,
                "privileged_source": "disabled",
                "rewards": rewards,
                "optimized": False,
                "reason": "no valid PI and zero group-relative advantage",
            }
            history.append(record)
            _write_history(args.output_dir / "steps.json", history)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue

        result = optimize_replay_group(
            model,
            tokenizer,
            optimizer,
            artifact.sample,
            group,
            privileged,
            reference_model=reference_model,
            clip_low=args.clip_low,
            clip_high=args.clip_high,
            beta=args.reference_kl_beta,
            lambda_opsd=args.lambda_opsd,
            max_sequence_length=args.max_sequence_length,
        )
        optimizer_step += 1
        last_result = result
        record = {
            "example_id": artifact.sample.id,
            "protocol_passed": artifact.quality.passed,
            "privileged_source": privileged.source if privileged is not None else "disabled",
            "rewards": rewards,
            "optimized": True,
            "optimizer_step": optimizer_step,
            **asdict(result),
            "loss": asdict(result.loss),
        }
        history.append(record)
        _write_history(args.output_dir / "steps.json", history)
        print(
            json.dumps(
                {
                    "event": "protocol_train_optimize_progress",
                    "processed": len(history),
                    "total": len(artifacts),
                    "example_id": artifact.sample.id,
                    "optimizer_step": optimizer_step,
                    "privileged_source": record["privileged_source"],
                    "loss": record["loss"],
                    "grad_norm": result.grad_norm,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    if last_result is None:
        raise RuntimeError("no sample supplied either OPSD or GRPO training signal")
    checkpoint_dir = args.output_dir / "checkpoint-final"
    checkpoint = save_smoke_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        last_result,
        base_model=str(args.model.resolve()),
        step=optimizer_step,
    )

    _, probe = trainable[0]
    expected = float(probe.detach().reshape(-1)[0].cpu())
    with torch.no_grad():
        probe.reshape(-1)[0].add_(1.0)
    loaded_step = load_smoke_checkpoint(checkpoint, model, optimizer)
    restored = float(probe.detach().reshape(-1)[0].cpu())
    if loaded_step != optimizer_step or restored != expected:
        raise RuntimeError("final checkpoint reload verification failed")

    summary = {
        "schema": "mapd-real-protocol-train-smoke-v2",
        "examples": len(artifacts),
        "passed_protocols": sum(item.quality.passed for item in artifacts),
        "optimizer_steps": optimizer_step,
        "skipped_no_signal": sum(not item["optimized"] for item in history),
        "protocol_pi_steps": sum(
            item["privileged_source"] == "protocol" and item["optimized"] for item in history
        ),
        "self_rollout_pi_steps": sum(
            item["privileged_source"] == "self_rollout" and item["optimized"]
            for item in history
        ),
        "grpo_signal_steps": sum(
            len(set(item["rewards"])) > 1 and item["optimized"] for item in history
        ),
        "mean_reference_kl": (
            sum(item["loss"]["reference_kl"] for item in history if item["optimized"])
            / optimizer_step
        ),
        "training_config": {
            "group_size": len(next(iter(groups.values()))),
            "learning_rate": args.learning_rate,
            "clip_low": args.clip_low,
            "clip_high": args.clip_high,
            "reference_kl_beta": args.reference_kl_beta,
            "lambda_opsd": args.lambda_opsd,
            "max_sequence_length": args.max_sequence_length,
        },
        "checkpoint": str(checkpoint.resolve()),
        "reload_verified": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("REAL PROTOCOL MAPD TRAINING OK")
    return 0


def _group_rollouts(
    trajectories: list[AgentTrajectory],
) -> dict[str, list[AgentTrajectory]]:
    groups: dict[str, list[AgentTrajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[trajectory.example_id].append(trajectory)
    return dict(groups)


def _validate_inputs(
    artifacts: list[SynthesisArtifact],
    groups: dict[str, list[AgentTrajectory]],
) -> None:
    if not artifacts:
        raise ValueError("protocol artifact shard is empty")
    artifact_ids = [item.sample.id for item in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("protocol artifact shard contains duplicate example ids")
    if set(groups) != set(artifact_ids):
        missing = sorted(set(artifact_ids) - set(groups))
        extra = sorted(set(groups) - set(artifact_ids))
        raise ValueError(f"rollout/artifact mismatch: missing={missing}, extra={extra}")
    group_sizes = {len(group) for group in groups.values()}
    if len(group_sizes) != 1 or next(iter(group_sizes)) < 2:
        raise ValueError("every example must have the same rollout group size of at least two")
    for artifact in artifacts:
        if artifact.quality.passed and artifact.protocol is None:
            raise ValueError(f"passed artifact {artifact.sample.id!r} has no protocol")
        for trajectory in groups[artifact.sample.id]:
            for turn in trajectory.turns:
                if turn.token_ids is None or turn.token_log_probs is None:
                    raise ValueError("rollout is missing exact token metadata")


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".building")
    temporary.write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
