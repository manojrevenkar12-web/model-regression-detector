import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    warning_threshold: float = 0.05
    critical_threshold: float = 0.15
    drift_window_runs: int = 7
    drift_threshold: float = 0.05
    report_trend_runs: int = 20
    concurrency_limit: int = 5
    classifier_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o"
    classifier_max_tokens: int = 256
    judge_max_tokens: int = 16


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
            os.environ.get("DRIFT_WINDOW_RUNS", data.get("drift_window_runs", 7))
        ),
        drift_threshold=float(
            os.environ.get("DRIFT_THRESHOLD", data.get("drift_threshold", 0.05))
        ),
        report_trend_runs=int(
            os.environ.get("REPORT_TREND_RUNS", data.get("report_trend_runs", 20))
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
        classifier_max_tokens=int(
            os.environ.get("CLASSIFIER_MAX_TOKENS", data.get("classifier_max_tokens", 256))
        ),
        judge_max_tokens=int(
            os.environ.get("JUDGE_MAX_TOKENS", data.get("judge_max_tokens", 16))
        ),
    )
