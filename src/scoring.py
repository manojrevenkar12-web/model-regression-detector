import re

from src.llm import call_llm_full
from src.models import CaseResult, ClassifierOutput, GoldenCase

_JUDGE_SYSTEM = """\
You are an expert evaluator scoring how well an AI-generated summary matches \
the reference summary for a customer support email.
Respond with a single integer from 1 to 5 using this rubric:
5 - Captures all key points; same meaning with only minor wording differences.
4 - Covers the main issue but misses a minor detail.
3 - Partially correct; key point present but notable gaps or imprecision.
2 - Mostly wrong or misses the main point; only superficial overlap.
1 - Completely wrong or unrelated.
Respond ONLY with the integer. No explanation."""


async def judge_summary(
    predicted_summary: str,
    expected_summary: str,
    judge_model: str,
    max_tokens: int = 16,
) -> tuple[int, int, int]:
    """Return (score 1-5, prompt_tokens, completion_tokens)."""
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Reference summary: {expected_summary}\n"
                f"Generated summary: {predicted_summary}"
            ),
        },
    ]
    resp = await call_llm_full(messages, model=judge_model, temperature=0.0, max_tokens=max_tokens)
    score = _parse_judge_score(resp.content)
    return score, resp.prompt_tokens, resp.completion_tokens


def _parse_judge_score(raw: str) -> int:
    match = re.search(r"[1-5]", raw.strip())
    if not match:
        raise ValueError(f"Judge returned unparseable score: {raw!r}")
    return int(match.group())


def score_case(
    case: GoldenCase,
    output: ClassifierOutput,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    judge_score: int | None,
    judge_prompt_tokens: int,
    judge_completion_tokens: int,
) -> CaseResult:
    labeled = bool(case.expected_category)
    category_match = (output.category == case.expected_category) if labeled else None
    return CaseResult(
        case_id=case.id,
        status="ok",
        predicted_category=output.category,
        predicted_summary=output.summary,
        expected_category=case.expected_category,
        expected_summary=case.expected_summary,
        category_match=category_match,
        judge_score=judge_score,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens + judge_prompt_tokens,
        completion_tokens=completion_tokens + judge_completion_tokens,
        error=None,
    )
