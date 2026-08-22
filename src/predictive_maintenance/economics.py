from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import THRESHOLDS


@dataclass(frozen=True)
class CostTuple:
    early_cycle_cost: float
    failure_cost: float
    downtime_cost: float


def first_alert(predicted_rul: np.ndarray, threshold: int) -> int | None:
    indices = np.where(predicted_rul <= threshold)[0]
    return int(indices[0]) if len(indices) else None


def trajectory_cost(
    predicted_rul: np.ndarray,
    actual_remaining: np.ndarray,
    threshold: int,
    cost: CostTuple,
    complete: bool,
) -> float:
    if not complete:
        raise ValueError("policy cost is forbidden on censored trajectories")
    alert = first_alert(predicted_rul, threshold)
    if alert is None:
        return cost.failure_cost + cost.downtime_cost
    return 1.0 + cost.early_cycle_cost * float(actual_remaining[alert])


def choose_threshold(
    trajectories: list[tuple[np.ndarray, np.ndarray, bool]], cost: CostTuple
) -> int:
    values = []
    for threshold in THRESHOLDS:
        values.append(
            (
                sum(trajectory_cost(p, a, threshold, cost, c) for p, a, c in trajectories)
                / len(trajectories),
                threshold,
            )
        )
    return min(values)[1]


def sensitivity(
    validation: list[tuple[np.ndarray, np.ndarray, bool]],
    policy_test: list[tuple[np.ndarray, np.ndarray, bool]],
) -> list[dict[str, float | int]]:
    if not validation or not policy_test:
        raise ValueError("validation and sealed policy-test trajectories are required")
    output = []
    for early in (0.0025, 0.005, 0.01):
        for failure in (5.0, 10.0, 20.0):
            for downtime in (0.0, 1.0, 2.0):
                cost = CostTuple(early, failure, downtime)
                threshold = choose_threshold(validation, cost)
                mean = sum(
                    trajectory_cost(predicted, actual, threshold, cost, complete)
                    for predicted, actual, complete in policy_test
                ) / len(policy_test)
                output.append(
                    {
                        "early_cycle_cost": early,
                        "failure_cost": failure,
                        "downtime_cost": downtime,
                        "threshold": threshold,
                        "mean_cost": mean,
                    }
                )
    return output
