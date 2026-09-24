#!/usr/bin/env python3
"""Run one low-memory, full-vocabulary MAPD optimizer step and reload it."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/smoke/artifacts.jsonl"))
    parser.add_argument("--rollouts", type=Path, default=Path("artifacts/train_smoke/rollouts.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train_smoke/checkpoint-step-1"))
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lambda-opsd", type=float, default=0.05)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the optimizer smoke test")
    trajectories = read_jsonl(args.rollouts, AgentTrajectory)
    if not trajectories:
        raise ValueError("rollout file is empty")
    example_ids = {item.example_id for item in trajectories}
    if len(example_ids) != 1:
        raise ValueError("all rollout trajectories must belong to one example")
    artifacts = read_jsonl(args.artifacts, SynthesisArtifact)
    artifact = next((item for item in artifacts if item.sample.id in example_ids), None)
    if artifact is None:
        raise ValueError("no synthesis artifact matches the rollout group")
    privileged = select_privileged_information(artifact, trajectories)
    if privileged is None:
        raise RuntimeError("no valid protocol or successful self-rollout is available")

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
    optimizer = torch.optim.AdamW(
        (parameter for _, parameter in trainable),
        lr=args.learning_rate,
        foreach=False,
    )
    result = optimize_replay_group(
        model,
        tokenizer,
        optimizer,
        artifact.sample,
        trajectories,
        privileged,
        lambda_opsd=args.lambda_opsd,
        max_sequence_length=args.max_sequence_length,
    )
    if result.sampled_parameter_delta <= 0:
        raise RuntimeError("optimizer step did not change a sampled trainable parameter")
    checkpoint = save_smoke_checkpoint(
        args.output,
        model,
        optimizer,
        result,
        base_model=str(args.model.resolve()),
        step=1,
    )

    _, probe = trainable[0]
    expected = float(probe.detach().reshape(-1)[0].cpu())
    with torch.no_grad():
        probe.reshape(-1)[0].add_(1.0)
    loaded_step = load_smoke_checkpoint(checkpoint, model, optimizer)
    restored = float(probe.detach().reshape(-1)[0].cpu())
    if loaded_step != 1 or restored != expected:
        raise RuntimeError("checkpoint reload verification failed")

    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reload_verified"] = True
    manifest["privileged_source"] = privileged.source
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**asdict(result), "loss": asdict(result.loss)}, ensure_ascii=False, indent=2))
    print(f"checkpoint: {checkpoint.resolve()}")
    print("MAPD OPTIMIZER SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
