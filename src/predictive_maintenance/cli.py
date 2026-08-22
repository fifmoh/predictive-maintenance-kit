from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from researchops import ArtifactEntry, RunManifest, create_run_manifest, sha256_file

from .economics import CostTuple, choose_threshold, sensitivity
from .protocol import Preprocessor, ai4i_split, assert_no_leakage, engine_split, windows


def root() -> Path:
    return Path(__file__).parents[2]


def fixture_data() -> pd.DataFrame:
    rows = []
    for engine in range(1, 101):
        for cycle in range(1, 41):
            rows.append(
                {
                    "engine": engine,
                    "cycle": cycle,
                    "s1": engine * 0.01 + cycle * 0.1,
                    "s2": cycle * 0.2,
                    "constant": 1.0,
                }
            )
    return pd.DataFrame(rows)


def reproduce() -> dict[str, object]:
    base = root()
    frame = fixture_data()
    identities = engine_split(list(range(1, 101)))
    train = set(identities["train"])
    validation = set(identities["validation"])
    policy = set(identities["policy_test"])
    fit_frame = frame[frame.engine.isin(train)]
    sensors = ["s1", "s2", "constant"]
    processor = Preprocessor.fit(fit_frame[sensors].to_numpy(float))
    assert_no_leakage(train, validation, policy, train)
    features, _, engines = windows(frame[frame.engine.isin(train)], sensors, processor)
    assert set(engines) == train
    validation_trajectories = []
    for _engine in sorted(validation):
        actual = np.arange(39, -1, -1, dtype=float)
        prediction = actual + np.sin(np.arange(40))
        validation_trajectories.append((prediction, actual, True))
    policy_trajectories = []
    for _engine in sorted(policy):
        actual = np.arange(39, -1, -1, dtype=float)
        prediction = actual + np.cos(np.arange(40))
        policy_trajectories.append((prediction, actual, True))
    chosen = choose_threshold(validation_trajectories, CostTuple(0.005, 10, 1))
    surface = sensitivity(validation_trajectories, policy_trajectories)
    dataset_hash = hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode()
    ).hexdigest()
    config = {"seed": 42, "window": 30, "rul_cap": 125, "fixture": True}
    run = create_run_manifest(
        project="predictive-maintenance-kit",
        git_sha="FIXTURE",
        cell_id=None,
        attempt=1,
        config=config,
        lock_sha256=sha256_file(base / "uv.lock"),
        datasets={"synthetic_fd001": dataset_hash},
        models={},
        seeds=[42],
        command=["make", "reproduce"],
    )
    output = base / "artifacts/runs" / run.run_id
    target = output / "result.json"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = RunManifest.model_validate_json(manifest_path.read_text())
        if existing.datasets.get("synthetic_fd001") != dataset_hash or not existing.artifacts:
            raise RuntimeError("immutable maintenance manifest does not match inputs")
        for artifact in existing.artifacts:
            artifact_path = (output / artifact.path).resolve()
            if (
                output.resolve() not in artifact_path.parents
                or not artifact_path.is_file()
                or sha256_file(artifact_path) != artifact.sha256
            ):
                raise RuntimeError(
                    f"immutable maintenance artifact failed verification: {artifact.path}"
                )
        return cast(dict[str, object], json.loads(target.read_text()))
    if output.exists():
        raise RuntimeError(f"immutable run directory is incomplete: {output}")
    output.mkdir(parents=True)
    payload: dict[str, object] = {
        "warning": "SYNTHETIC TRAJECTORY FIXTURE — NOT A MEASURED RESULT",
        "split": {key: list(value) for key, value in identities.items()},
        "window_shape": list(features.shape),
        "chosen_threshold": chosen,
        "threshold_selected_on": "validation",
        "cost_evaluated_on": "sealed_policy_test",
        "sensitivity": surface,
        "ai4i_counts": {
            key: len(value) for key, value in ai4i_split(np.r_[np.zeros(966), np.ones(34)]).items()
        },
    }
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    artifact = ArtifactEntry(
        path="result.json",
        sha256=sha256_file(target),
        media_type="application/json",
        size=target.stat().st_size,
        role="fixture_evidence",
    )
    finished = run.model_copy(
        update={
            "finished_at": datetime.now(UTC),
            "artifacts": [artifact],
            "status": "success",
            "warnings": ["SYNTHETIC FIXTURE — NOT A MEASURED RESULT"],
        }
    )
    manifest_path.write_text(finished.model_dump_json(indent=2) + "\n")
    print(target)
    return payload


def bundle() -> None:
    base = root()
    target = base / "dist/predictive-maintenance-colab.tar.gz"
    target.parent.mkdir(exist_ok=True)
    members = [
        base / name for name in ("pyproject.toml", "uv.lock", "Makefile", "README.md", "LICENSE")
    ]
    for directory in ("src", "config", "notebooks", "docs", "report"):
        members.extend(path for path in (base / directory).rglob("*") if path.is_file())
    if any(path.is_symlink() for path in members):
        raise RuntimeError("Colab bundle refuses symbolic links")
    with tarfile.open(target, "w:gz") as archive:
        for path in sorted(members):
            archive.add(path, arcname=path.relative_to(base), recursive=False)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n")
    print(digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "smoke",
            "docs",
            "demo",
            "data",
            "leakage-check",
            "reproduce",
            "evaluate",
            "report",
            "ui-test",
            "colab-bundle",
        ],
    )
    args = parser.parse_args()
    if args.command == "data":
        if os.getenv("ACK_DATA_LICENSES") != "1":
            raise SystemExit("Set ACK_DATA_LICENSES=1 before official NASA/UCI download")
        print(root() / "config/sources.json")
    elif args.command == "colab-bundle":
        bundle()
    else:
        reproduce()


if __name__ == "__main__":
    main()
