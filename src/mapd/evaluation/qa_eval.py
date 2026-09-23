from dataclasses import dataclass

from mapd.environment.schema import AgentTrajectory


@dataclass(frozen=True)
class EvaluationSummary:
    count: int
    successes: int
    success_rate: float


def evaluate_trajectories(trajectories: list[AgentTrajectory]) -> EvaluationSummary:
    successes = sum(trajectory.reward == 1.0 for trajectory in trajectories)
    return EvaluationSummary(
        count=len(trajectories),
        successes=successes,
        success_rate=successes / len(trajectories) if trajectories else 0.0,
    )

