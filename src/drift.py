from statistics import mean

from src.config import Config
from src.models import AlertLevel, DriftResult
from src.storage import list_runs


def detect_drift(config: Config) -> DriftResult:
    """
    Detect gradual pass-rate degradation over a rolling window of runs.

    Uses summary rows from SQLite (pass_rate, mean_judge_score) — no full
    RunResult JSON is loaded. The oldest run in the window is the anchor;
    drop = anchor.pass_rate - mean(window.pass_rates). Fires when drop
    >= drift_threshold even if no individual run crossed warning_threshold.
    """
    rows = list_runs(limit=config.drift_window_runs)
    # list_runs returns newest-first; preserve that order for the result.
    run_ids = [r["run_id"] for r in rows]

    if len(rows) < 2:
        return DriftResult(
            window_run_ids=run_ids,
            window_size=len(rows),
            insufficient_data=True,
            reference_pass_rate=None,
            rolling_avg_pass_rate=None,
            rolling_avg_judge_score=None,
            pass_rate_drop=None,
            drift_detected=False,
            alert_level=AlertLevel.ok,
        )

    pass_rates = [r["pass_rate"] for r in rows]
    rolling_avg = mean(pass_rates)

    # Oldest run = last in the newest-first list = the window's anchor.
    reference = rows[-1]["pass_rate"]
    drop = round(reference - rolling_avg, 4)

    judge_scores = [r["mean_judge_score"] for r in rows if r["mean_judge_score"] is not None]
    rolling_avg_judge = mean(judge_scores) if judge_scores else None

    drift_detected = drop >= config.drift_threshold
    alert_level = AlertLevel.warning if drift_detected else AlertLevel.ok

    return DriftResult(
        window_run_ids=run_ids,
        window_size=len(rows),
        insufficient_data=False,
        reference_pass_rate=round(reference, 4),
        rolling_avg_pass_rate=round(rolling_avg, 4),
        rolling_avg_judge_score=round(rolling_avg_judge, 4) if rolling_avg_judge is not None else None,
        pass_rate_drop=drop,
        drift_detected=drift_detected,
        alert_level=alert_level,
    )
