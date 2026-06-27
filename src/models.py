from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


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
