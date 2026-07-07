"""Entry point: python cli.py run | compare"""
import argparse
import asyncio

from src.config import load_config
from src.diff import compare_runs
from src.runner import run_eval
from src.storage import list_runs, load_run, save_run


def _print_run(run) -> None:
    print(f"run_id       : {run.run_id}")
    print(f"pass_rate    : {run.pass_rate:.1%}")
    print(f"mean_judge   : {run.mean_judge_score}")
    print(f"mean_latency : {run.mean_latency_ms:.0f}ms" if run.mean_latency_ms else "mean_latency : n/a")
    errors = [c for c in run.cases if c.status == "error"]
    if errors:
        print(f"errors       : {[c.case_id for c in errors]}")

    print("\n  id        expected    predicted   match  judge")
    print("  " + "-" * 52)
    for c in run.cases:
        predicted = c.predicted_category or ("ERROR" if c.status == "error" else "?")
        match_str = ("YES" if c.category_match else "NO ") if c.category_match is not None else "n/a"
        judge_str = str(c.judge_score) if c.judge_score is not None else "n/a"
        flag = "  <-- MISS" if c.category_match is False else ""
        print(f"  {c.case_id:8}  {c.expected_category:10}  {predicted:10}  {match_str:5}  {judge_str}{flag}")


def _print_compare(cmp, baseline, current) -> None:
    print(f"baseline  : {cmp.baseline_run_id}  pass_rate={baseline.pass_rate:.1%}")
    print(f"current   : {cmp.current_run_id}  pass_rate={current.pass_rate:.1%}")
    print(f"delta     : {cmp.pass_rate_delta:+.1%}  [{cmp.alert_level.value.upper()}]")
    print(f"regressions  : {cmp.regressions or 'none'}")
    print(f"improvements : {cmp.improvements or 'none'}")

    print("\nper-category delta:")
    for cat, delta in cmp.per_category_delta.items():
        print(f"  {cat:10s}: {delta:+.0%}")

    by1 = {c.case_id: c for c in baseline.cases}
    by2 = {c.case_id: c for c in current.cases}
    print("\nper-case judge delta:")
    print("  id        r1_judge  r2_judge  delta")
    print("  " + "-" * 38)
    for case_id in sorted(by1):
        c1, c2 = by1[case_id], by2[case_id]
        j1, j2 = c1.judge_score, c2.judge_score
        jd = (j2 - j1) if (j1 is not None and j2 is not None) else None
        flag = f"  <-- {jd:+d}" if jd else ""
        print(f"  {case_id:8}  {str(j1):8}  {str(j2):8}  {(f'{jd:+d}' if jd is not None else 'n/a'):5}{flag}")


async def cmd_run(args) -> None:
    config = load_config(args.config)
    print(f"Running eval: dataset={args.dataset}  prompt={args.prompt}\n")
    run = await run_eval(
        dataset_path=args.dataset,
        prompt_path=args.prompt,
        config=config,
    )
    saved = save_run(run)
    print(f"Saved: {saved}\n")
    _print_run(run)


def cmd_compare(args) -> None:
    config = load_config(args.config)
    rows = list_runs(limit=2)
    if len(rows) < 2:
        raise SystemExit("Need at least 2 saved runs to compare.")
    current = load_run(rows[0]["run_id"])
    baseline = load_run(rows[1]["run_id"])
    cmp = compare_runs(baseline=baseline, current=current, config=config)
    _print_compare(cmp, baseline, current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model regression detector CLI")
    parser.add_argument("--config", default="config.yaml", metavar="PATH")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Evaluate the golden dataset and save the run")
    run_p.add_argument("--dataset", default="data/golden_v1.json", metavar="PATH")
    run_p.add_argument("--prompt", default="prompts/email_classifier_v1.yaml", metavar="PATH")

    sub.add_parser("compare", help="Diff the two most recent saved runs")

    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
