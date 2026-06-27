import json
from pathlib import Path

from pydantic import ValidationError

from src.models import GoldenDataset


def load_dataset(path: str | Path) -> GoldenDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    try:
        return GoldenDataset.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid dataset {path}:\n{exc}") from exc
