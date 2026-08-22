import numpy as np
import pandas as pd
import pytest

from predictive_maintenance.economics import (
    CostTuple,
    choose_threshold,
    sensitivity,
    trajectory_cost,
)
from predictive_maintenance.protocol import (
    Preprocessor,
    ai4i_split,
    assert_no_leakage,
    engine_split,
    windows,
)


def test_exact_engine_split_and_leakage_guards():
    split = engine_split(list(range(1, 101)))
    assert {k: len(v) for k, v in split.items()} == {
        "train": 70,
        "validation": 15,
        "policy_test": 15,
    }
    with pytest.raises(ValueError):
        assert_no_leakage(
            set(split["train"]),
            set(split["validation"]),
            set(split["policy_test"]),
            set(split["validation"]),
        )


def test_windows_never_cross_engines_and_train_fit():
    rows = [
        {"engine": engine, "cycle": cycle, "s": engine + cycle}
        for engine in (1, 2)
        for cycle in range(1, 36)
    ]
    frame = pd.DataFrame(rows)
    processor = Preprocessor.fit(frame[frame.engine == 1][["s"]].to_numpy(float))
    x, y, engines = windows(frame, ["s"], processor)
    assert x.shape[1] == 30 and set(engines) == {1, 2} and all(v <= 125 for v in y)


def test_policy_only_complete_and_sensitivity():
    actual = np.arange(39, -1, -1, dtype=float)
    trajectories = [(actual + 1, actual, True) for _ in range(3)]
    assert choose_threshold(trajectories, CostTuple(0.005, 10, 1)) in (
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
    )
    assert len(sensitivity(trajectories, trajectories)) == 27
    with pytest.raises(ValueError):
        trajectory_cost(actual, actual, 20, CostTuple(0.005, 10, 1), False)


def test_ai4i_fixed_split():
    y = np.r_[np.zeros(966), np.ones(34)]
    first = ai4i_split(y)
    second = ai4i_split(y)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert {k: len(v) for k, v in first.items()} == {"train": 700, "validation": 150, "test": 150}
