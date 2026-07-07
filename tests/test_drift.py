"""Unit tests for src/drift.py — rolling average, drop detection, edge cases."""
from unittest.mock import patch

import pytest

from src.config import Config
from src.models import AlertLevel, DriftResult
from src.drift import detect_drift

_CFG = Config(drift_window_runs=5, drift_threshold=0.05)


def _rows(*pass_rates, judge_scores=None):
    """Build a fake list_runs result (newest first)."""
    js = judge_scores or [None] * len(pass_rates)
    return [
        {"run_id": f"run-{i}", "pass_rate": pr, "mean_judge_score": jsc}
        for i, (pr, jsc) in enumerate(zip(pass_rates, js))
    ]


def _run(rows) -> DriftResult:
    with patch("src.drift.list_runs", return_value=rows):
        return detect_drift(_CFG)


# ── insufficient data ──────────────────────────────────────────────────────


def test_zero_runs_insufficient():
    result = _run([])
    assert result.insufficient_data is True
    assert result.drift_detected is False
    assert result.alert_level == AlertLevel.ok


def test_one_run_insufficient():
    result = _run(_rows(0.90))
    assert result.insufficient_data is True
    assert result.drift_detected is False
    assert result.window_size == 1


def test_two_runs_sufficient():
    result = _run(_rows(0.90, 0.90))
    assert result.insufficient_data is False


# ── rolling average ────────────────────────────────────────────────────────


def test_flat_window_zero_drop():
    result = _run(_rows(0.90, 0.90, 0.90, 0.90, 0.90))
    assert result.pass_rate_drop == pytest.approx(0.0)
    assert result.drift_detected is False
    assert result.alert_level == AlertLevel.ok


def test_rolling_avg_computed_correctly():
    # newest-first: [1.0, 0.8, 0.6]; avg = 0.8; anchor (oldest) = 0.6; drop = -0.2
    result = _run(_rows(1.0, 0.8, 0.6))
    assert result.rolling_avg_pass_rate == pytest.approx(0.8)
    assert result.reference_pass_rate == pytest.approx(0.6)
    # Improving trend → negative drop → no drift
    assert result.pass_rate_drop == pytest.approx(-0.2, abs=1e-4)
    assert result.drift_detected is False


def test_reference_is_oldest_run_in_window():
    # newest-first: 0.80, 0.85, 0.90 → oldest = 0.90
    result = _run(_rows(0.80, 0.85, 0.90))
    assert result.reference_pass_rate == pytest.approx(0.90)


def test_window_size_reflects_actual_rows():
    result = _run(_rows(0.90, 0.85))
    assert result.window_size == 2


# ── drift detection ────────────────────────────────────────────────────────


def test_drop_exactly_at_threshold_fires():
    # 2 runs: newest=0.80, oldest(anchor)=0.90 → avg=0.85, drop=0.05 >= 0.05
    result = _run(_rows(0.80, 0.90))
    assert result.pass_rate_drop == pytest.approx(0.05, abs=1e-4)
    assert result.drift_detected is True
    assert result.alert_level == AlertLevel.warning


def test_drop_just_below_threshold_no_fire():
    # anchor = 0.90, runs = [0.854, 0.854, 0.90] → avg ≈ 0.869, drop ≈ 0.031
    result = _run(_rows(0.86, 0.86, 0.90))
    assert result.drift_detected is False
    assert result.alert_level == AlertLevel.ok


def test_gradual_decline_fires_drift():
    # Each step drops 1pp; no single step crosses warning_threshold (0.05).
    # Newest-first: 0.86, 0.87, 0.88, 0.89, 0.90
    # anchor = 0.90, avg = 0.88, drop = 0.02 — below threshold → no fire yet.
    # Steeper: 0.82, 0.84, 0.86, 0.88, 0.90 → avg=0.86, drop=0.04 — still ok.
    # 0.80, 0.83, 0.86, 0.88, 0.90 → avg=0.854, drop=0.046 — still ok.
    # 0.79, 0.82, 0.85, 0.88, 0.90 → avg=0.848, drop=0.052 → fires.
    result = _run(_rows(0.79, 0.82, 0.85, 0.88, 0.90))
    assert result.drift_detected is True
    assert result.pass_rate_drop == pytest.approx(0.052, abs=1e-3)


def test_improvement_trend_no_drift():
    # Rising: oldest=0.80, newest=0.95 → drop is negative
    result = _run(_rows(0.95, 0.90, 0.85, 0.82, 0.80))
    assert result.drift_detected is False
    assert result.pass_rate_drop < 0


# ── judge score rolling average ────────────────────────────────────────────


def test_judge_avg_computed_from_non_none_scores():
    result = _run(_rows(0.90, 0.90, 0.90, judge_scores=[4.0, 5.0, 3.0]))
    assert result.rolling_avg_judge_score == pytest.approx(4.0)


def test_judge_avg_none_when_all_scores_none():
    result = _run(_rows(0.90, 0.90))
    assert result.rolling_avg_judge_score is None


def test_judge_avg_skips_none_values():
    result = _run(_rows(0.90, 0.90, 0.90, judge_scores=[None, 4.0, None]))
    assert result.rolling_avg_judge_score == pytest.approx(4.0)


# ── window_run_ids ─────────────────────────────────────────────────────────


def test_run_ids_preserved_newest_first():
    rows = _rows(0.90, 0.85, 0.80)
    result = _run(rows)
    assert result.window_run_ids == ["run-0", "run-1", "run-2"]
