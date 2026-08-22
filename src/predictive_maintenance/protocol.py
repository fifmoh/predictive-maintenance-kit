from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

SEEDS = (17, 23, 42, 71, 101)
RUL_CAP = 125
WINDOW = 30
THRESHOLDS = (5, 10, 15, 20, 25, 30, 40, 50)


def engine_split(engine_ids: list[int]) -> dict[str, tuple[int, ...]]:
    if len(set(engine_ids)) < 100:
        raise ValueError("FD001 split requires 100 unique training engines")
    ordered = sorted(
        set(engine_ids), key=lambda engine: hashlib.sha256(f"fd001\0{engine}".encode()).hexdigest()
    )
    result = {
        "train": tuple(ordered[:70]),
        "validation": tuple(ordered[70:85]),
        "policy_test": tuple(ordered[85:100]),
    }
    assert (
        not set(result["train"]) & set(result["validation"])
        and not set(result["train"]) & set(result["policy_test"])
        and not set(result["validation"]) & set(result["policy_test"])
    )
    return result


@dataclass(frozen=True)
class Preprocessor:
    means: np.ndarray
    scales: np.ndarray
    kept: tuple[int, ...]

    @classmethod
    def fit(cls, x: np.ndarray) -> Preprocessor:
        variance = x.var(axis=0)
        kept = tuple(int(i) for i in np.where(variance > 0)[0])
        selected = x[:, kept]
        scale = selected.std(axis=0)
        scale[scale == 0] = 1
        return cls(selected.mean(axis=0), scale, kept)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray((x[:, self.kept] - self.means) / self.scales, dtype=float)


def windows(
    frame: pd.DataFrame, sensors: list[str], preprocessor: Preprocessor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    engines = []
    for engine, group in frame.sort_values(["engine", "cycle"]).groupby("engine"):
        values = preprocessor.transform(group[sensors].to_numpy(float))
        cycles = group["cycle"].to_numpy(int)
        maximum = int(cycles.max())
        for end in range(WINDOW - 1, len(group)):
            xs.append(values[end - WINDOW + 1 : end + 1])
            ys.append(min(RUL_CAP, maximum - int(cycles[end])))
            engines.append(int(engine))
    return np.asarray(xs), np.asarray(ys), np.asarray(engines)


def assert_no_leakage(
    train: set[int], validation: set[int], policy: set[int], fit_engines: set[int]
) -> None:
    if train & validation or train & policy or validation & policy:
        raise ValueError("engine overlap")
    if fit_engines != train:
        raise ValueError("preprocessing must fit exactly the 70 training engines")


def ai4i_split(target: np.ndarray) -> dict[str, np.ndarray]:
    indices = np.arange(len(target))
    train, rest = train_test_split(indices, test_size=0.30, stratify=target, random_state=42)
    validation, test = train_test_split(
        rest, test_size=0.50, stratify=target[rest], random_state=42
    )
    return {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
