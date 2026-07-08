"""Slack alerts for run comparisons. Posts to the incoming webhook at
SLACK_WEBHOOK_URL (.env); if unset, prints the same payload to console so the
pipeline is demoable without a real webhook."""
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

from src.models import AlertLevel, RunComparison, RunResult

load_dotenv()

_BADGE_TEXT = {
    AlertLevel.ok: ":white_check_mark: PASS",
    AlertLevel.warning: ":warning: WARNING",
    AlertLevel.critical: ":rotating_light: CRITICAL",
}

# The literal placeholder shipped in .env.example — a fresh clone that copies
# .env.example to .env without editing it should fall back to console, not
# 404 against a fake URL.
_PLACEHOLDER_WEBHOOK_URL = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"


def build_payload(
    comparison: RunComparison,
    baseline: RunResult,
    current: RunResult,
    drift_status: str | None = None,
) -> dict:
    """Build the Slack message payload for a run comparison."""
    headline = (
        f"{len(comparison.regressions)} regressions, "
        f"pass rate {baseline.pass_rate:.1%} -> {current.pass_rate:.1%} "
        f"({comparison.pass_rate_delta:+.1%})"
    )

    lines = [
        f"*{_BADGE_TEXT[comparison.alert_level]}*",
        headline,
        f"Run: `{comparison.baseline_run_id}` -> `{comparison.current_run_id}`",
    ]
    if drift_status:
        lines.append(f"Drift: {drift_status}")

    text = "\n".join(lines)
    return {
        "text": text,
        "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": text}}],
    }


def send_alert(
    comparison: RunComparison,
    baseline: RunResult,
    current: RunResult,
    drift_status: str | None = None,
) -> None:
    """Post the comparison result to Slack, or print it if no webhook is configured."""
    payload = build_payload(comparison, baseline, current, drift_status)
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    if not webhook_url or webhook_url == _PLACEHOLDER_WEBHOOK_URL:
        print(json.dumps(payload, indent=2))
        return

    _post(webhook_url, payload)


def _post(webhook_url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack webhook POST failed: {exc}") from exc
