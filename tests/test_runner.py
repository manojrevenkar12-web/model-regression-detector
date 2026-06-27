"""Unit tests for src/runner.py — error handling, pass_rate exclusion."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.config import Config
from src.llm import LLMResponse
from src.models import GoldenCase, GoldenDataset, PromptConfig
from src.runner import run_eval

_CFG = Config(
    concurrency_limit=5,
    classifier_model="openai/gpt-4o-mini",
    judge_model="openai/gpt-4o",
)

# Three cases: two labeled correctly, one that will trigger an LLM error.
# expected_summary is empty so no judge calls are made.
_DATASET = GoldenDataset(
    version="1.0.0",
    cases=[
        GoldenCase(
            id="c1", text="billing email",
            expected_category="billing", expected_summary="",
            expected_difficulty="easy",
        ),
        GoldenCase(
            id="c2", text="technical email",
            expected_category="technical", expected_summary="",
            expected_difficulty="easy",
        ),
        GoldenCase(
            id="c3", text="bad case",
            expected_category="account", expected_summary="",
            expected_difficulty="easy",
        ),
    ],
)

_PROMPT = PromptConfig(
    version="test-1.0.0",
    created_at=datetime(2026, 6, 27, tzinfo=timezone.utc),
    model="openai/gpt-4o-mini",
    system_prompt="Classify the email.",
    few_shot_examples=[],
)


def _last_user(messages: list[dict]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


async def _fake_one_error(messages, model, temperature=0.0, **kwargs):
    text = _last_user(messages)
    if text == "bad case":
        raise RuntimeError("LLM timeout")
    if text == "billing email":
        return LLMResponse(
            content='{"category":"billing","summary":"billing issue"}',
            prompt_tokens=10, completion_tokens=5,
        )
    return LLMResponse(
        content='{"category":"technical","summary":"tech issue"}',
        prompt_tokens=10, completion_tokens=5,
    )


def _run(fake_llm):
    """Context manager triple for a run_eval call with mocked I/O."""
    import contextlib

    @contextlib.asynccontextmanager
    async def _ctx():
        with (
            patch("src.runner.load_dataset", return_value=_DATASET),
            patch("src.runner.load_prompt", return_value=_PROMPT),
            patch("src.runner.call_llm_full", side_effect=fake_llm),
        ):
            yield await run_eval(config=_CFG)

    return _ctx()


# ── tests ──────────────────────────────────────────────────────────────────


async def test_error_case_status_and_message():
    """LLM exception → CaseResult.status='error' with the error message."""
    async with _run(_fake_one_error) as result:
        err = next(c for c in result.cases if c.case_id == "c3")
    assert err.status == "error"
    assert err.error is not None
    assert "LLM timeout" in err.error


async def test_error_case_has_no_predictions():
    """Error case must not expose a predicted category or category_match."""
    async with _run(_fake_one_error) as result:
        err = next(c for c in result.cases if c.case_id == "c3")
    assert err.category_match is None
    assert err.predicted_category is None


async def test_run_completes_despite_partial_errors():
    """run_eval must return a full RunResult even when some cases error."""
    async with _run(_fake_one_error) as result:
        pass
    assert result is not None
    assert len(result.cases) == 3
    assert sum(1 for c in result.cases if c.status == "ok") == 2


async def test_error_case_excluded_from_pass_rate():
    """
    c1: billing→billing (correct), c2: technical→technical (correct), c3: error.
    labeled_ok denominator = 2 (c3 excluded), both correct → pass_rate = 1.0.
    """
    async with _run(_fake_one_error) as result:
        pass
    assert result.pass_rate == pytest.approx(1.0)


async def test_all_errors_gives_zero_pass_rate():
    async def always_raise(messages, model, temperature=0.0, **kwargs):
        raise RuntimeError("all bad")

    async with _run(always_raise) as result:
        pass
    assert all(c.status == "error" for c in result.cases)
    assert result.pass_rate == pytest.approx(0.0)


async def test_partial_correct_gives_fractional_pass_rate():
    """c1 correct, c2 returns wrong category (billing), c3 errors → pass_rate = 0.5."""

    async def fake_mixed(messages, model, temperature=0.0, **kwargs):
        text = _last_user(messages)
        if text == "bad case":
            raise RuntimeError("err")
        if text == "billing email":
            return LLMResponse(
                content='{"category":"billing","summary":"ok"}',
                prompt_tokens=10, completion_tokens=5,
            )
        # technical email → wrong answer
        return LLMResponse(
            content='{"category":"billing","summary":"wrong"}',
            prompt_tokens=10, completion_tokens=5,
        )

    async with _run(fake_mixed) as result:
        pass

    # labeled_ok = c1 (correct) + c2 (wrong) = 2; correct = 1 → 0.5
    assert result.pass_rate == pytest.approx(0.5)
