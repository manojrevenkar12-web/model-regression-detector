import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    warning_threshold: float = 0.05
    critical_threshold: float = 0.15
    drift_window_runs: int = 5
    concurrency_limit: int = 5
    classifier_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o"


def load_config(path: str | Path = "config.yaml") -> Config:
    data: dict = {}
    p = Path(path)
    if p.exists():
        with p.open() as f:
            data = yaml.safe_load(f) or {}

    return Config(
        warning_threshold=float(
            os.environ.get("WARNING_THRESHOLD", data.get("warning_threshold", 0.05))
        ),
        critical_threshold=float(
            os.environ.get("CRITICAL_THRESHOLD", data.get("critical_threshold", 0.15))
        ),
        drift_window_runs=int(
            os.environ.get("DRIFT_WINDOW_RUNS", data.get("drift_window_runs", 5))
        ),
        concurrency_limit=int(
            os.environ.get("CONCURRENCY_LIMIT", data.get("concurrency_limit", 5))
        ),
        classifier_model=os.environ.get(
            "CLASSIFIER_MODEL", data.get("classifier_model", "openai/gpt-4o-mini")
        ),
        judge_model=os.environ.get(
            "JUDGE_MODEL", data.get("judge_model", "openai/gpt-4o")
        ),
    )
