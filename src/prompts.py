from pathlib import Path

import yaml
from pydantic import ValidationError

from src.models import PromptConfig


def load_prompt(path: str | Path) -> PromptConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    try:
        return PromptConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid prompt file {path}: {exc}") from exc
