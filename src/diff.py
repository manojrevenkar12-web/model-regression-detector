from src.config import Config
from src.models import AlertLevel, RunComparison, RunResult

_CATEGORIES = ("billing", "technical", "account", "general")


def compare_runs(
    baseline: RunResult, current: RunResult, config: Config
) -> RunComparison:
    pass_rate_delta = current.pass_rate - baseline.pass_rate

    baseline_by_id = {c.case_id: c for c in baseline.cases}
    current_by_id = {c.case_id: c for c in current.cases}

    regressions: list[str] = []
    improvements: list[str] = []
    for case_id, cur in current_by_id.items():
        base = baseline_by_id.get(case_id)
        if base is None:
            continue
        was_correct = base.status == "ok" and base.category_match is True
        now_correct = cur.status == "ok" and cur.category_match is True
        if was_correct and not now_correct:
            regressions.append(case_id)
        elif not was_correct and now_correct:
            improvements.append(case_id)

    per_category_delta: dict[str, float] = {}
    for cat in _CATEGORIES:
        base_labeled = [c for c in baseline.cases if c.expected_category == cat]
        cur_labeled = [c for c in current.cases if c.expected_category == cat]
        if not base_labeled or not cur_labeled:
            continue
        base_rate = sum(1 for c in base_labeled if c.category_match) / len(base_labeled)
        cur_rate = sum(1 for c in cur_labeled if c.category_match) / len(cur_labeled)
        per_category_delta[cat] = round(cur_rate - base_rate, 4)

    delta = abs(pass_rate_delta)
    if delta >= config.critical_threshold:
        alert_level = AlertLevel.critical
    elif delta >= config.warning_threshold:
        alert_level = AlertLevel.warning
    else:
        alert_level = AlertLevel.ok

    return RunComparison(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        pass_rate_delta=round(pass_rate_delta, 4),
        per_category_delta=per_category_delta,
        regressions=regressions,
        improvements=improvements,
        alert_level=alert_level,
    )
