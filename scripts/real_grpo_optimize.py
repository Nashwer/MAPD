#!/usr/bin/env python3
"""Run and checkpoint a true non-zero GRPO-only update from real rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory
from mapd.io import read_jsonl
from mapd.trainer.single_device import (
    enable_last_decoder_layer,
    optimize_replay_group,
    save_smoke_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/real_grpo_smoke"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/real_grpo_smoke/checkpoint-step-1"))
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-low", type=float, default=0.2)
    parser.add_argument("--clip-high", type=float, default=0.2)
    parser.add_argument("--reference-kl-beta", type=float, default=0.001)
    parser.add_argument("--max-sequence-length", type=int, default=4608)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GRPO smoke test")
    samples = read_jsonl(args.input_dir / "sample.jsonl", QASample)
    trajectories = read_jsonl(args.input_dir / "rollouts.jsonl", AgentTrajectory)
    if len(samples) != 1 or not trajectories:
        raise ValueError("expected one sample and a non-empty rollout group")
    rewards = [item.reward for item in trajectories]
    if len(set(rewards)) < 2:
        raise ValueError("rollout group has zero reward variance")
    if any(item.example_id != samples[0].id for item in trajectories):
        raise ValueError("rollout group does not match the selected sample")

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
    result = optimize_replay_group(
        model,
        tokenizer,
        optimizer,
        samples[0],
        trajectories,
        None,
        reference_model=reference_model,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        beta=args.reference_kl_beta,
        lambda_opsd=0.0,
        max_sequence_length=args.max_sequence_length,
    )
    # With normalized group advantages and an initial ratio of one, the scalar
    # GRPO loss can sum to zero even though its policy gradient is non-zero.
    if result.grad_norm <= 0:
        raise RuntimeError("mixed-reward GRPO produced no trainable gradient")
    if result.sampled_parameter_delta <= 0:
        raise RuntimeError("GRPO optimizer step did not change a trainable parameter")
    checkpoint = save_smoke_checkpoint(
        args.output,
        model,
        optimizer,
        result,
        base_model=str(args.model.resolve()),
        step=1,
    )
    summary = {
        **asdict(result),
        "loss": asdict(result.loss),
        "rewards": rewards,
        "grpo_gradient_verified": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"checkpoint: {checkpoint.resolve()}")
    print("REAL GRPO OPTIMIZER SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
