from datetime import datetime
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
