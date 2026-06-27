import asyncio
import time
import uuid
from datetime import datetime, timezone
from statistics import mean

from src.classifier import _build_messages, _parse_output
from src.config import Config, load_config
from src.dataset import load_dataset
from src.llm import call_llm_full
from src.models import CaseResult, GoldenCase, RunResult
from src.prompts import load_prompt
from src.scoring import judge_summary, score_case


async def _run_case(
    case: GoldenCase,
    prompt_config,
    config: Config,
    sem: asyncio.Semaphore,
) -> CaseResult:
    async with sem:
        try:
            messages = _build_messages(case.text, prompt_config)
            t0 = time.perf_counter()
            llm_resp = await call_llm_full(
                messages, model=config.classifier_model, temperature=0.0
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            output = _parse_output(llm_resp.content)

            judge_score = None
            j_pt = j_ct = 0
            if case.expected_summary:
                judge_score, j_pt, j_ct = await judge_summary(
                    output.summary, case.expected_summary, config.judge_model
                )

            return score_case(
                case=case,
                output=output,
                latency_ms=latency_ms,
                prompt_tokens=llm_resp.prompt_tokens,
                completion_tokens=llm_resp.completion_tokens,
                judge_score=judge_score,
                judge_prompt_tokens=j_pt,
                judge_completion_tokens=j_ct,
            )
        except Exception as exc:
            return CaseResult(
                case_id=case.id,
                status="error",
                expected_category=case.expected_category,
                expected_summary=case.expected_summary,
                error=str(exc),
            )


async def run_eval(
    dataset_path: str = "data/golden_v1.json",
    prompt_path: str = "prompts/email_classifier_v1.yaml",
    config: Config | None = None,
) -> RunResult:
    if config is None:
        config = load_config()

    dataset = load_dataset(dataset_path)
    prompt_config = load_prompt(prompt_path)

    sem = asyncio.Semaphore(config.concurrency_limit)
    tasks = [_run_case(case, prompt_config, config, sem) for case in dataset.cases]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    case_results: list[CaseResult] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, BaseException):
            # _run_case catches internally; this is a last-resort safety net
            case_results.append(
                CaseResult(
                    case_id=dataset.cases[i].id,
                    status="error",
                    expected_category=dataset.cases[i].expected_category,
                    expected_summary=dataset.cases[i].expected_summary,
                    error=str(r),
                )
            )
        else:
            case_results.append(r)

    labeled_ok = [
        c for c in case_results if c.expected_category and c.status == "ok"
    ]
    pass_rate = (
        sum(1 for c in labeled_ok if c.category_match) / len(labeled_ok)
        if labeled_ok
        else 0.0
    )

    scored = [c for c in case_results if c.judge_score is not None]
    mean_judge: float | None = mean(c.judge_score for c in scored) if scored else None  # type: ignore[arg-type]

    timed = [c for c in case_results if c.latency_ms is not None]
    mean_latency: float | None = mean(c.latency_ms for c in timed) if timed else None  # type: ignore[arg-type]

    return RunResult(
        run_id=str(uuid.uuid4()),
        run_at=datetime.now(tz=timezone.utc),
        dataset_version=dataset.version,
        prompt_version=prompt_config.version,
        classifier_model=config.classifier_model,
        judge_model=config.judge_model,
        cases=case_results,
        pass_rate=pass_rate,
        mean_judge_score=mean_judge,
        mean_latency_ms=mean_latency,
    )
