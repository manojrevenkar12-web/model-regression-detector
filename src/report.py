"""Self-contained HTML diff report: run metadata, scorecard, regressed-case
table, and a matplotlib pass-rate trend chart embedded as a base64 PNG so the
file opens standalone with no network access. Written to reports/<run_id>.html
(gitignored)."""
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from jinja2 import BaseLoader, Environment, select_autoescape

from src.config import Config
from src.models import CaseResult, RunComparison, RunResult
from src.storage import list_runs

_REPORTS_DIR = Path("reports")

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Run report {{ current.run_id }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.1rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }
  table { border-collapse: collapse; width: 100%; margin-top: .75rem; font-size: .9rem; }
  th, td { border: 1px solid #ddd; padding: .4rem .6rem; text-align: left; vertical-align: top; }
  th { background: #f5f5f5; }
  .meta-table td:first-child { font-weight: 600; width: 12rem; }
  .badge { display: inline-block; padding: .25rem .7rem; border-radius: .3rem;
           font-weight: 600; color: #fff; font-size: .85rem; }
  .badge-ok { background: #2e7d32; }
  .badge-warning { background: #e69100; }
  .badge-critical { background: #c62828; }
  .delta-pos { color: #2e7d32; }
  .delta-neg { color: #c62828; }
  .muted { color: #666; }
  .scorecard { display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 1rem; }
  .scorecard .stat { min-width: 9rem; }
  .scorecard .stat .value { font-size: 1.4rem; font-weight: 700; }
  .scorecard .stat .label { font-size: .8rem; color: #666; }
  img.trend { max-width: 100%; margin-top: .75rem; }
</style>
</head>
<body>

<h1>Regression report — {{ current.run_id }}</h1>
<p><span class="badge badge-{{ comparison.alert_level.value }}">{{ comparison.alert_level.value|upper }}</span></p>

<h2 id="metadata">Run metadata</h2>
<table class="meta-table">
  <tr><th></th><th>Baseline</th><th>Current</th></tr>
  <tr><td>Run ID</td><td>{{ baseline.run_id }}</td><td>{{ current.run_id }}</td></tr>
  <tr><td>Run at (UTC)</td><td>{{ baseline.run_at }}</td><td>{{ current.run_at }}</td></tr>
  <tr><td>Dataset version</td><td>{{ baseline.dataset_version }}</td><td>{{ current.dataset_version }}</td></tr>
  <tr><td>Prompt version</td><td>{{ baseline.prompt_version }}</td><td>{{ current.prompt_version }}</td></tr>
  <tr><td>Classifier model</td><td>{{ baseline.classifier_model }}</td><td>{{ current.classifier_model }}</td></tr>
  <tr><td>Judge model</td><td>{{ baseline.judge_model }}</td><td>{{ current.judge_model }}</td></tr>
</table>

<h2 id="scorecard">Scorecard</h2>
<div class="scorecard">
  <div class="stat">
    <div class="value">{{ "%.1f"|format(baseline.pass_rate * 100) }}% &rarr; {{ "%.1f"|format(current.pass_rate * 100) }}%</div>
    <div class="label">pass rate</div>
  </div>
  <div class="stat">
    <div class="value {{ 'delta-pos' if comparison.pass_rate_delta >= 0 else 'delta-neg' }}">{{ "%+.1f"|format(comparison.pass_rate_delta * 100) }}pp</div>
    <div class="label">pass rate delta</div>
  </div>
  <div class="stat">
    <div class="value">{{ comparison.regressions|length }}</div>
    <div class="label">regressions</div>
  </div>
  <div class="stat">
    <div class="value">{{ comparison.improvements|length }}</div>
    <div class="label">improvements</div>
  </div>
</div>

{% if comparison.per_category_delta %}
<table>
  <tr><th>Category</th><th>Delta</th></tr>
  {% for cat, delta in comparison.per_category_delta.items() %}
  <tr><td>{{ cat }}</td><td class="{{ 'delta-pos' if delta >= 0 else 'delta-neg' }}">{{ "%+.1f"|format(delta * 100) }}pp</td></tr>
  {% endfor %}
</table>
{% endif %}

<h2 id="regressed-cases">Regressed cases</h2>
{% if regressed_cases %}
<table>
  <tr>
    <th>Case ID</th>
    <th>Expected category</th>
    <th>Old category</th>
    <th>New category</th>
    <th>Old summary</th>
    <th>New summary</th>
  </tr>
  {% for row in regressed_cases %}
  <tr>
    <td>{{ row.case_id }}</td>
    <td>{{ row.expected_category }}</td>
    <td>{{ row.old_category }}</td>
    <td>{{ row.new_category }}</td>
    <td>{{ row.old_summary }}</td>
    <td>{{ row.new_summary }}</td>
  </tr>
  {% endfor %}
</table>
{% else %}
<p class="muted">No regressions in this run.</p>
{% endif %}

<h2 id="trend">Pass rate trend</h2>
{% if chart_b64 %}
<img class="trend" alt="Pass rate trend" src="data:image/png;base64,{{ chart_b64 }}">
{% else %}
<p class="muted">Not enough saved runs to plot a trend yet.</p>
{% endif %}

</body>
</html>
"""


def _regressed_case_rows(
    comparison: RunComparison, baseline: RunResult, current: RunResult
) -> list[dict[str, str]]:
    baseline_by_id: dict[str, CaseResult] = {c.case_id: c for c in baseline.cases}
    current_by_id: dict[str, CaseResult] = {c.case_id: c for c in current.cases}

    rows = []
    for case_id in comparison.regressions:
        base = baseline_by_id.get(case_id)
        cur = current_by_id.get(case_id)
        if base is None or cur is None:
            continue
        rows.append(
            {
                "case_id": case_id,
                "expected_category": cur.expected_category,
                "old_category": base.predicted_category or "ERROR",
                "new_category": cur.predicted_category or "ERROR",
                "old_summary": base.predicted_summary or "",
                "new_summary": cur.predicted_summary or "",
            }
        )
    return rows


def _render_trend_chart(config: Config) -> str:
    """Render pass_rate over the last N runs to a base64-encoded PNG.

    Returns "" when fewer than 2 runs exist — nothing meaningful to plot.
    """
    rows = list_runs(limit=config.report_trend_runs)
    rows = list(reversed(rows))  # oldest -> newest, left to right
    if len(rows) < 2:
        return ""

    x = list(range(len(rows)))
    y = [r["pass_rate"] for r in rows]
    labels = [r["run_id"][:8] for r in rows]

    fig, ax = plt.subplots(figsize=(6.5, 2.5))
    ax.plot(x, y, marker="o", color="#1f6feb")
    ax.set_ylim(0, 1)
    ax.set_ylabel("pass rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Pass rate — last {len(rows)} runs")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def generate_report(
    comparison: RunComparison,
    baseline: RunResult,
    current: RunResult,
    config: Config,
) -> Path:
    """Render a self-contained HTML diff report to reports/<current.run_id>.html."""
    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
    template = env.from_string(_TEMPLATE)
    html = template.render(
        comparison=comparison,
        baseline=baseline,
        current=current,
        regressed_cases=_regressed_case_rows(comparison, baseline, current),
        chart_b64=_render_trend_chart(config),
    )

    _REPORTS_DIR.mkdir(exist_ok=True)
    out_path = _REPORTS_DIR / f"{current.run_id}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
