from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Category = Literal["billing", "technical", "account", "general"]
Difficulty = Literal["easy", "medium", "hard"]

_VALID_CATEGORIES: frozenset[str] = frozenset({"billing", "technical", "account", "general"})


class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    version: str
    created_at: datetime
    model: str
    system_prompt: str
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)


class EmailInput(BaseModel):
    id: str
    text: str


class ClassifierOutput(BaseModel):
    category: Category
    summary: str


class GoldenCase(BaseModel):
    id: str
    text: str
    expected_category: str = ""   # empty until human-labeled
    expected_summary: str = ""    # empty until human-labeled
    expected_difficulty: Difficulty
    notes: str = ""

    @field_validator("expected_category")
    @classmethod
    def category_must_be_valid_or_empty(cls, v: str) -> str:
        if v and v not in _VALID_CATEGORIES:
            raise ValueError(
                f"expected_category must be one of {sorted(_VALID_CATEGORIES)} or "
                f"an empty string, got {v!r}"
            )
        return v


class GoldenDataset(BaseModel):
    version: str
    cases: list[GoldenCase]


class CaseResult(BaseModel):
    case_id: str
    status: Literal["ok", "error"] = "ok"
    predicted_category: str | None = None
    predicted_summary: str | None = None
    expected_category: str
    expected_summary: str
    category_match: bool | None = None
    judge_score: int | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class RunResult(BaseModel):
    run_id: str
    run_at: datetime
    dataset_version: str
    prompt_version: str
    classifier_model: str
    judge_model: str
    cases: list[CaseResult]
    pass_rate: float
    mean_judge_score: float | None
    mean_latency_ms: float | None


class AlertLevel(str, Enum):
    ok = "ok"
    warning = "warning"
    critical = "critical"


class RunComparison(BaseModel):
    baseline_run_id: str
    current_run_id: str
    pass_rate_delta: float
    per_category_delta: dict[str, float]
    regressions: list[str]
    improvements: list[str]
    alert_level: AlertLevel
