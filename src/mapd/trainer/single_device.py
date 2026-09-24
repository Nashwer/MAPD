from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mapd.data.schema import QASample
from mapd.environment.schema import AgentTrajectory
from mapd.trainer.grpo import group_relative_advantages
from mapd.trainer.mapd_trainer import LossBreakdown
from mapd.trainer.opsd import PrivilegedInformation
from mapd.trainer.replay import TokenReplayTurn, build_token_replay


@dataclass(frozen=True)
class SingleDeviceStepResult:
    loss: LossBreakdown
    grad_norm: float
    sampled_parameter_delta: float
    response_tokens: int
    trainable_parameters: int


def enable_last_decoder_layer(model: Any) -> list[tuple[str, Any]]:
    """Freeze the model except its last decoder layer for a low-memory smoke step."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model does not expose model.layers for the smoke optimizer")
    for parameter in layers[-1].parameters():
        parameter.requires_grad_(True)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters were selected")
    return trainable


def optimize_replay_group(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    sample: QASample,
    trajectories: list[AgentTrajectory],
    privileged_information: PrivilegedInformation | None,
    *,
    reference_model: Any | None = None,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    beta: float = 0.0,
    lambda_opsd: float = 0.05,
    max_sequence_length: int = 4096,
    max_grad_norm: float = 1.0,
) -> SingleDeviceStepResult:
    """Run one exact-token, full-vocabulary MAPD update on a single device.

    This is an acceptance backend, not the distributed paper trainer. It uses
    the rollout token ids/log-probabilities when available, replays identical
    response ids under both contexts, and detaches the privileged branch.
    """
    import torch

    if not trajectories:
        raise ValueError("a rollout group cannot be empty")
    if beta > 0 and reference_model is None:
        raise ValueError("a frozen reference model is required when beta is positive")
    if min(clip_low, clip_high, beta, lambda_opsd, max_grad_norm) < 0:
        raise ValueError("loss coefficients and max_grad_norm must be non-negative")

    replays = [
        build_token_replay(sample, trajectory, tokenizer, privileged_information)
        for trajectory in trajectories
    ]
    trajectory_tokens = [sum(len(turn.response_ids) for turn in replay) for replay in replays]
    if any(count == 0 for count in trajectory_tokens):
        raise ValueError("every trajectory must contain at least one response token")
    total_tokens = sum(trajectory_tokens)
    advantages = group_relative_advantages([trajectory.reward for trajectory in trajectories])
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")

    model.eval()
    optimizer.zero_grad(set_to_none=True)
    grpo_value = 0.0
    reference_kl_value = 0.0
    opsd_value = 0.0
    group_size = len(trajectories)

    for replay, trajectory_token_count, advantage in zip(
        replays, trajectory_tokens, advantages
    ):
        for turn in replay:
            reference_log_probs = None
            if reference_model is not None and beta > 0:
                with torch.no_grad():
                    reference_logits = _response_logits(
                        reference_model,
                        turn.student_prefix_ids,
                        turn.response_ids,
                        max_sequence_length,
                    )
                    reference_log_probs = torch.log_softmax(
                        reference_logits.float(), dim=-1
                    )
            privileged_log_probs = None
            if turn.privileged_prefix_ids is not None:
                with torch.no_grad():
                    privileged_logits = _response_logits(
                        model,
                        turn.privileged_prefix_ids,
                        turn.response_ids,
                        max_sequence_length,
                    )
                    privileged_log_probs = torch.log_softmax(
                        privileged_logits.float(), dim=-1
                    )

            student_logits = _response_logits(
                model,
                turn.student_prefix_ids,
                turn.response_ids,
                max_sequence_length,
            )
            student_log_probs = torch.log_softmax(student_logits.float(), dim=-1)
            response_ids = torch.tensor(
                turn.response_ids,
                device=student_log_probs.device,
                dtype=torch.long,
            )
            selected_log_probs = student_log_probs.gather(
                -1, response_ids.unsqueeze(-1)
            ).squeeze(-1)
            old_log_probs = _old_log_probs(turn, selected_log_probs)
            ratios = torch.exp(selected_log_probs - old_log_probs)
            clipped = ratios.clamp(1.0 - clip_low, 1.0 + clip_high)
            advantage_tensor = selected_log_probs.new_tensor(advantage)
            token_objective = torch.minimum(
                ratios * advantage_tensor,
                clipped * advantage_tensor,
            )
            token_reference_kl = selected_log_probs.new_zeros(selected_log_probs.shape)
            if reference_log_probs is not None:
                token_reference_kl = (
                    student_log_probs.exp()
                    * (student_log_probs - reference_log_probs.detach())
                ).sum(-1)
            normalization = trajectory_token_count * group_size
            turn_grpo = -(
                token_objective - beta * token_reference_kl
            ).sum() / normalization

            turn_opsd = selected_log_probs.new_zeros(())
            if privileged_log_probs is not None:
                token_kl = (
                    student_log_probs.exp()
                    * (student_log_probs - privileged_log_probs.detach())
                ).sum(-1)
                turn_opsd = token_kl.sum() / total_tokens

            turn_loss = turn_grpo + lambda_opsd * turn_opsd
            if not torch.isfinite(turn_loss):
                raise FloatingPointError("non-finite MAPD loss")
            turn_loss.backward()
            grpo_value += float(turn_grpo.detach().cpu())
            reference_kl_value += float(
                (token_reference_kl.sum() / normalization).detach().cpu()
            )
            opsd_value += float(turn_opsd.detach().cpu())

    parameters = [parameter for _, parameter in trainable]
    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    grad_norm = float(grad_norm_tensor.detach().cpu())
    if not torch.isfinite(grad_norm_tensor):
        raise FloatingPointError("non-finite gradient norm")
    sampled_before = _sample_parameters_at_max_gradient(trainable)
    if not sampled_before:
        raise RuntimeError("the MAPD objective produced no trainable gradients")
    optimizer.step()
    sampled_delta = max(
        abs(float(parameter.detach().reshape(-1)[index].cpu()) - before)
        for parameter, index, before in sampled_before
    )

    loss = LossBreakdown(
        grpo=grpo_value,
        reference_kl=reference_kl_value,
        opsd=opsd_value,
        total=grpo_value + lambda_opsd * opsd_value,
        mean_reward=sum(item.reward for item in trajectories) / group_size,
        distillation_tokens=total_tokens if privileged_information is not None else 0,
    )
    return SingleDeviceStepResult(
        loss=loss,
        grad_norm=grad_norm,
        sampled_parameter_delta=sampled_delta,
        response_tokens=total_tokens,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
    )


def save_smoke_checkpoint(
    output_dir: str | Path,
    model: Any,
    optimizer: Any,
    result: SingleDeviceStepResult,
    *,
    base_model: str,
    step: int,
) -> Path:
    import torch

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    checkpoint_path = destination / "checkpoint.pt"
    torch.save(
        {
            "step": step,
            "base_model": base_model,
            "trainable_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    manifest = {
        "format": "mapd-single-device-smoke-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "step": step,
        "base_model": base_model,
        "result": {**asdict(result), "loss": asdict(result.loss)},
        "checkpoint": str(checkpoint_path),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def load_smoke_checkpoint(checkpoint_path: str | Path, model: Any, optimizer: Any | None = None) -> int:
    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = set(payload["trainable_state"])
    if actual != expected:
        raise ValueError("checkpoint trainable parameter set does not match the model")
    current = model.state_dict()
    current.update(payload["trainable_state"])
    model.load_state_dict(current, strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return int(payload["step"])


def _response_logits(
    model: Any,
    prefix_ids: list[int],
    response_ids: list[int],
    max_sequence_length: int,
):
    import torch

    if max_sequence_length <= len(response_ids):
        raise ValueError("max_sequence_length must leave room for a prompt token")
    prefix = prefix_ids[-(max_sequence_length - len(response_ids)) :]
    if not prefix:
        raise ValueError("the rendered prompt is empty")
    device = next(model.parameters()).device
    # Causal position prefix_len - 1 predicts the first response token. The
    # final response token therefore does not need to be fed back into the
    # model. Qwen's logits_to_keep avoids materializing vocabulary logits for
    # the long prompt, which is essential on a 24 GB smoke-test GPU.
    input_ids = torch.tensor(
        [prefix + response_ids[:-1]], device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids)
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": False,
        "logits_to_keep": len(response_ids),
    }
    try:
        output = model(**kwargs)
    except TypeError:
        kwargs.pop("logits_to_keep")
        output = model(**kwargs)
    return output.logits[0, -len(response_ids) :, :]


def _old_log_probs(turn: TokenReplayTurn, selected_log_probs: Any):
    import torch

    if turn.old_log_probs is None:
        return selected_log_probs.detach()
    if len(turn.old_log_probs) != len(turn.response_ids):
        raise ValueError("rollout log-probabilities do not align with response ids")
    return torch.tensor(
        turn.old_log_probs,
        device=selected_log_probs.device,
        dtype=selected_log_probs.dtype,
    )


def _sample_parameters_at_max_gradient(
    trainable: list[tuple[str, Any]],
) -> list[tuple[Any, int, float]]:
    sampled: list[tuple[Any, int, float]] = []
    for _, parameter in trainable:
        if parameter.grad is None or parameter.numel() == 0:
            continue
        flat_gradient = parameter.grad.detach().reshape(-1)
        index = int(flat_gradient.abs().argmax().item())
        before = float(parameter.detach().reshape(-1)[index].cpu())
        sampled.append((parameter, index, before))
    return sampled
