"""Unit tests for src/alerts.py — payload shape, webhook POST vs console fallback."""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.alerts import build_payload, send_alert
from src.models import AlertLevel, CaseResult, RunComparison, RunResult

_NOW = datetime(2026, 6, 27, tzinfo=timezone.utc)


def _run(run_id: str, pass_rate: float) -> RunResult:
    return RunResult(
        run_id=run_id,
        run_at=_NOW,
        dataset_version="1.0.0",
        prompt_version="v1",
        classifier_model="openai/gpt-4o-mini",
        judge_model="openai/gpt-4o",
        cases=[],
        pass_rate=pass_rate,
        mean_judge_score=None,
        mean_latency_ms=None,
    )


def _cmp(alert_level: AlertLevel, delta: float = -0.1, regressions=None) -> RunComparison:
    return RunComparison(
        baseline_run_id="baseline-1",
        current_run_id="current-1",
        pass_rate_delta=delta,
        per_category_delta={},
        regressions=regressions or ["c1"],
        improvements=[],
        alert_level=alert_level,
    )


_BASELINE = _run("baseline-1", 0.9)
_CURRENT = _run("current-1", 0.8)


# ── build_payload ────────────────────────────────────────────────────────────


class TestBuildPayload:
    def test_headline_reports_regression_count_and_pass_rates(self):
        payload = build_payload(_cmp(AlertLevel.warning), _BASELINE, _CURRENT)
        text = payload["text"]
        assert "1 regressions" in text
        assert "90.0%" in text
        assert "80.0%" in text

    def test_ok_level_uses_pass_badge(self):
        payload = build_payload(_cmp(AlertLevel.ok), _BASELINE, _CURRENT)
        assert "PASS" in payload["text"]

    def test_warning_level_uses_warning_badge(self):
        payload = build_payload(_cmp(AlertLevel.warning), _BASELINE, _CURRENT)
        assert "WARNING" in payload["text"]

    def test_critical_level_uses_critical_badge(self):
        payload = build_payload(_cmp(AlertLevel.critical), _BASELINE, _CURRENT)
        assert "CRITICAL" in payload["text"]

    def test_drift_status_included_when_provided(self):
        payload = build_payload(_cmp(AlertLevel.ok), _BASELINE, _CURRENT, drift_status="no drift")
        assert "Drift: no drift" in payload["text"]

    def test_drift_status_omitted_when_none(self):
        payload = build_payload(_cmp(AlertLevel.ok), _BASELINE, _CURRENT, drift_status=None)
        assert "Drift:" not in payload["text"]

    def test_payload_has_slack_blocks_shape(self):
        payload = build_payload(_cmp(AlertLevel.ok), _BASELINE, _CURRENT)
        assert payload["blocks"][0]["type"] == "section"
        assert payload["blocks"][0]["text"]["type"] == "mrkdwn"


# ── send_alert ────────────────────────────────────────────────────────────


class TestSendAlert:
    def test_posts_to_webhook_when_url_configured(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
        mock_response = MagicMock()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False

        with patch("src.alerts.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            send_alert(_cmp(AlertLevel.critical), _BASELINE, _CURRENT, drift_status="no drift")

        assert mock_urlopen.called
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://hooks.slack.com/services/T/B/X"
        sent_payload = json.loads(request.data.decode("utf-8"))
        assert "CRITICAL" in sent_payload["text"]

    def test_falls_back_to_console_when_url_unset(self, monkeypatch, capsys):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

        with patch("src.alerts.urllib.request.urlopen") as mock_urlopen:
            send_alert(_cmp(AlertLevel.ok), _BASELINE, _CURRENT)

        mock_urlopen.assert_not_called()
        printed = capsys.readouterr().out
        payload = json.loads(printed)
        assert "PASS" in payload["text"]

    def test_falls_back_to_console_when_url_is_env_example_placeholder(self, monkeypatch, capsys):
        """A fresh clone that copies .env.example to .env without editing it
        must not 404 against the placeholder — treat it as unconfigured."""
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/YOUR/WEBHOOK/URL")

        with patch("src.alerts.urllib.request.urlopen") as mock_urlopen:
            send_alert(_cmp(AlertLevel.ok), _BASELINE, _CURRENT)

        mock_urlopen.assert_not_called()
        printed = capsys.readouterr().out
        payload = json.loads(printed)
        assert "PASS" in payload["text"]

    def test_webhook_failure_raises_runtime_error(self, monkeypatch):
        import urllib.error

        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/T/B/X")
        with patch(
            "src.alerts.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Slack webhook POST failed"):
                send_alert(_cmp(AlertLevel.ok), _BASELINE, _CURRENT)
