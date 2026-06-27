"""Unit tests for src/diff.py — regressions, improvements, alert boundaries."""
from datetime import datetime, timezone

import pytest

from src.config import Config
from src.diff import compare_runs
from src.models import AlertLevel, CaseResult, RunComparison, RunResult

_CFG = Config(warning_threshold=0.05, critical_threshold=0.15)
_NOW = datetime(2026, 6, 27, tzinfo=timezone.utc)


# ── helpers ────────────────────────────────────────────────────────────────


def _case(
    case_id: str,
    expected: str,
    predicted: str | None,
    status: str = "ok",
) -> CaseResult:
    labeled = bool(expected)
    match: bool | None = None
    if status == "ok" and labeled and predicted is not None:
        match = predicted == expected
    return CaseResult(
        case_id=case_id,
        status=status,
        predicted_category=predicted,
        predicted_summary="s" if status == "ok" else None,
        expected_category=expected,
        expected_summary="e" if labeled else "",
        category_match=match,
        error=None if status == "ok" else "err",
    )


def _run(
    run_id: str,
    cases: list[CaseResult],
    pass_rate: float | None = None,
) -> RunResult:
    if pass_rate is None:
        labeled_ok = [c for c in cases if c.expected_category and c.status == "ok"]
        pass_rate = (
            sum(1 for c in labeled_ok if c.category_match) / len(labeled_ok)
            if labeled_ok
            else 0.0
        )
    return RunResult(
        run_id=run_id,
        run_at=_NOW,
        dataset_version="1.0.0",
        prompt_version="1.0.0",
        classifier_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o",
        cases=cases,
        pass_rate=pass_rate,
        mean_judge_score=None,
        mean_latency_ms=None,
    )


def _cmp(base_rate: float, cur_rate: float) -> RunComparison:
    return compare_runs(_run("b", [], base_rate), _run("c", [], cur_rate), _CFG)


# ── alert level boundaries ─────────────────────────────────────────────────


class TestAlertLevel:
    def test_zero_delta_is_ok(self):
        assert _cmp(0.80, 0.80).alert_level == AlertLevel.ok

    def test_small_delta_below_warning_is_ok(self):
        # abs(delta) = 0.04 < 0.05
        assert _cmp(1.0, 0.96).alert_level == AlertLevel.ok

    def test_at_warning_boundary_is_warning(self):
        # abs(delta) = 0.05 — exactly at threshold
        assert _cmp(1.0, 0.95).alert_level == AlertLevel.warning

    def test_between_warning_and_critical_is_warning(self):
        # abs(delta) = 0.10
        assert _cmp(1.0, 0.90).alert_level == AlertLevel.warning

    def test_at_critical_boundary_is_critical(self):
        # abs(delta) = 0.15 — exactly at threshold
        assert _cmp(1.0, 0.85).alert_level == AlertLevel.critical

    def test_above_critical_is_critical(self):
        # abs(delta) = 0.20
        assert _cmp(1.0, 0.80).alert_level == AlertLevel.critical

    def test_positive_delta_at_warning_triggers_warning(self):
        # Surprising improvement also alertable (uses abs)
        assert _cmp(0.80, 0.90).alert_level == AlertLevel.warning

    def test_positive_delta_at_critical_triggers_critical(self):
        assert _cmp(0.70, 0.90).alert_level == AlertLevel.critical

    def test_pass_rate_delta_sign_preserved_negative(self):
        assert _cmp(0.80, 0.70).pass_rate_delta == pytest.approx(-0.10)

    def test_pass_rate_delta_sign_preserved_positive(self):
        assert _cmp(0.70, 0.80).pass_rate_delta == pytest.approx(0.10)


# ── regression / improvement detection ────────────────────────────────────


class TestRegressionsAndImprovements:
    def test_correct_to_wrong_is_regression(self):
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [_case("c1", "billing", "technical")])
        cmp  = compare_runs(base, cur, _CFG)
        assert "c1" in cmp.regressions
        assert "c1" not in cmp.improvements

    def test_wrong_to_correct_is_improvement(self):
        base = _run("b", [_case("c1", "billing", "technical")])
        cur  = _run("c", [_case("c1", "billing", "billing")])
        cmp  = compare_runs(base, cur, _CFG)
        assert "c1" in cmp.improvements
        assert "c1" not in cmp.regressions

    def test_stable_correct_not_flagged(self):
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [_case("c1", "billing", "billing")])
        cmp  = compare_runs(base, cur, _CFG)
        assert cmp.regressions == []
        assert cmp.improvements == []

    def test_stable_wrong_not_flagged(self):
        base = _run("b", [_case("c1", "billing", "general")])
        cur  = _run("c", [_case("c1", "billing", "general")])
        cmp  = compare_runs(base, cur, _CFG)
        assert cmp.regressions == []
        assert cmp.improvements == []

    def test_correct_to_error_is_regression(self):
        # LLM error in current counts the same as a wrong answer
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [_case("c1", "billing", None, status="error")])
        cmp  = compare_runs(base, cur, _CFG)
        assert "c1" in cmp.regressions

    def test_error_to_correct_is_improvement(self):
        base = _run("b", [_case("c1", "billing", None, status="error")])
        cur  = _run("c", [_case("c1", "billing", "billing")])
        cmp  = compare_runs(base, cur, _CFG)
        assert "c1" in cmp.improvements

    def test_error_in_both_not_flagged(self):
        base = _run("b", [_case("c1", "billing", None, status="error")])
        cur  = _run("c", [_case("c1", "billing", None, status="error")])
        cmp  = compare_runs(base, cur, _CFG)
        assert cmp.regressions == []
        assert cmp.improvements == []

    def test_unlabeled_case_not_flagged(self):
        # category_match is None for unlabeled → neither regression nor improvement
        base = _run("b", [_case("c1", "", "billing")])
        cur  = _run("c", [_case("c1", "", "technical")])
        cmp  = compare_runs(base, cur, _CFG)
        assert cmp.regressions == []
        assert cmp.improvements == []

    def test_new_case_only_in_current_ignored(self):
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [
            _case("c1", "billing", "billing"),
            _case("c2", "billing", "billing"),  # no baseline to compare against
        ])
        cmp = compare_runs(base, cur, _CFG)
        assert cmp.regressions == []
        assert cmp.improvements == []

    def test_multiple_mixed_changes(self):
        base_cases = [
            _case("c1", "billing", "billing"),    # correct
            _case("c2", "technical", "billing"),   # wrong
            _case("c3", "account", "account"),     # correct, stable
        ]
        cur_cases = [
            _case("c1", "billing", "technical"),   # regression
            _case("c2", "technical", "technical"),  # improvement
            _case("c3", "account", "account"),      # unchanged
        ]
        cmp = compare_runs(_run("b", base_cases), _run("c", cur_cases), _CFG)
        assert set(cmp.regressions) == {"c1"}
        assert set(cmp.improvements) == {"c2"}


# ── per-category delta ─────────────────────────────────────────────────────


class TestPerCategoryDelta:
    def test_delta_computed_correctly(self):
        # billing: baseline 1/2=0.5, current 2/2=1.0 → delta=+0.5
        base = _run("b", [
            _case("c1", "billing", "billing"),
            _case("c2", "billing", "technical"),
        ])
        cur = _run("c", [
            _case("c1", "billing", "billing"),
            _case("c2", "billing", "billing"),
        ])
        cmp = compare_runs(base, cur, _CFG)
        assert "billing" in cmp.per_category_delta
        assert cmp.per_category_delta["billing"] == pytest.approx(0.5)

    def test_category_absent_from_both_omitted(self):
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [_case("c1", "billing", "billing")])
        cmp  = compare_runs(base, cur, _CFG)
        for cat in ("technical", "account", "general"):
            assert cat not in cmp.per_category_delta

    def test_category_absent_from_baseline_omitted(self):
        # "technical" only in current — not enough to compute a delta
        base = _run("b", [_case("c1", "billing", "billing")])
        cur  = _run("c", [
            _case("c1", "billing", "billing"),
            _case("c2", "technical", "technical"),
        ])
        cmp = compare_runs(base, cur, _CFG)
        assert "technical" not in cmp.per_category_delta

    def test_negative_category_delta(self):
        # billing: baseline 2/2=1.0, current 1/2=0.5 → delta=-0.5
        base = _run("b", [
            _case("c1", "billing", "billing"),
            _case("c2", "billing", "billing"),
        ])
        cur = _run("c", [
            _case("c1", "billing", "billing"),
            _case("c2", "billing", "technical"),
        ])
        cmp = compare_runs(base, cur, _CFG)
        assert cmp.per_category_delta["billing"] == pytest.approx(-0.5)
