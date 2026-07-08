"""Unit tests for src/report.py — HTML section content, regression rows, trend chart."""
import base64
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.config import Config
from src.models import AlertLevel, CaseResult, RunComparison, RunResult
from src.report import generate_report

_CFG = Config(report_trend_runs=10)
_NOW = datetime(2026, 6, 27, tzinfo=timezone.utc)


def _case(
    case_id: str,
    expected_category: str,
    predicted_category: str | None,
    predicted_summary: str | None = None,
    status: str = "ok",
) -> CaseResult:
    match = (predicted_category == expected_category) if status == "ok" else None
    return CaseResult(
        case_id=case_id,
        status=status,
        predicted_category=predicted_category,
        predicted_summary=predicted_summary,
        expected_category=expected_category,
        expected_summary="ref summary",
        category_match=match,
        error=None if status == "ok" else "boom",
    )


def _run(run_id: str, cases: list[CaseResult], pass_rate: float) -> RunResult:
    return RunResult(
        run_id=run_id,
        run_at=_NOW,
        dataset_version="1.0.0",
        prompt_version="v1",
        classifier_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o",
        cases=cases,
        pass_rate=pass_rate,
        mean_judge_score=None,
        mean_latency_ms=None,
    )


_BASELINE = _run(
    "baseline-1",
    [
        _case("c1", "billing", "billing", "billing question"),
        _case("c2", "technical", "technical", "tech question"),
    ],
    pass_rate=1.0,
)
_CURRENT = _run(
    "current-1",
    [
        _case("c1", "billing", "technical", "wrong summary"),  # regression
        _case("c2", "technical", "technical", "tech question"),
    ],
    pass_rate=0.5,
)
_COMPARISON = RunComparison(
    baseline_run_id="baseline-1",
    current_run_id="current-1",
    pass_rate_delta=-0.5,
    per_category_delta={"billing": -1.0},
    regressions=["c1"],
    improvements=[],
    alert_level=AlertLevel.critical,
)


@pytest.fixture(autouse=True)
def _reports_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("src.report._REPORTS_DIR", tmp_path)
    return tmp_path


def _generate(list_runs_return):
    with patch("src.report.list_runs", return_value=list_runs_return):
        return generate_report(_COMPARISON, _BASELINE, _CURRENT, _CFG)


def test_writes_html_to_reports_dir(_reports_dir):
    path = _generate([])
    assert path == _reports_dir / "current-1.html"
    assert path.exists()


def test_contains_metadata_section():
    html = _generate([]).read_text(encoding="utf-8")
    assert "Run metadata" in html
    assert "baseline-1" in html
    assert "current-1" in html
    assert "openai/gpt-4o-mini" in html
    assert "v1" in html


def test_contains_scorecard_section():
    html = _generate([]).read_text(encoding="utf-8")
    assert "Scorecard" in html
    assert "CRITICAL" in html
    assert "-50.0pp" in html  # pass_rate_delta as a percentage-point figure


def test_contains_regressed_case_row():
    html = _generate([]).read_text(encoding="utf-8")
    assert "Regressed cases" in html
    assert "c1" in html
    assert "technical" in html  # new (wrong) category shown
    assert "wrong summary" in html


def test_no_regressions_shows_placeholder_message():
    empty_cmp = RunComparison(
        baseline_run_id="baseline-1",
        current_run_id="current-1",
        pass_rate_delta=0.0,
        per_category_delta={},
        regressions=[],
        improvements=[],
        alert_level=AlertLevel.ok,
    )
    with patch("src.report.list_runs", return_value=[]):
        path = generate_report(empty_cmp, _BASELINE, _CURRENT, _CFG)
    html = path.read_text(encoding="utf-8")
    assert "No regressions in this run." in html


def test_trend_section_placeholder_when_insufficient_runs():
    html = _generate([{"run_id": "current-1", "pass_rate": 0.5, "mean_judge_score": None}]).read_text(
        encoding="utf-8"
    )
    assert "Pass rate trend" in html
    assert "Not enough saved runs" in html
    assert "<img" not in html


def test_trend_section_embeds_base64_png_when_enough_runs():
    rows = [
        {"run_id": "run-3", "pass_rate": 0.9, "mean_judge_score": None},
        {"run_id": "run-2", "pass_rate": 0.85, "mean_judge_score": None},
        {"run_id": "run-1", "pass_rate": 0.8, "mean_judge_score": None},
    ]
    html = _generate(rows).read_text(encoding="utf-8")
    assert "<img" in html
    assert 'src="data:image/png;base64,' in html

    b64_payload = html.split("base64,", 1)[1].split('"', 1)[0]
    png_bytes = base64.b64decode(b64_payload)
    assert png_bytes.startswith(b"\x89PNG")


def test_html_escapes_case_content():
    """Summaries containing HTML-special characters must not break out of the table."""
    xss_case = _case("c3", "billing", "billing", "<script>alert(1)</script>")
    baseline = _run("baseline-2", [xss_case], pass_rate=1.0)
    current = _run(
        "current-2",
        [_case("c3", "billing", "technical", "<script>alert(1)</script>")],
        pass_rate=0.0,
    )
    cmp = RunComparison(
        baseline_run_id="baseline-2",
        current_run_id="current-2",
        pass_rate_delta=-1.0,
        per_category_delta={},
        regressions=["c3"],
        improvements=[],
        alert_level=AlertLevel.critical,
    )
    with patch("src.report.list_runs", return_value=[]):
        path = generate_report(cmp, baseline, current, _CFG)
    html = path.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
