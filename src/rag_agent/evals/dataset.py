from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str
    query: str
    expected_sources: list[str] = Field(default_factory=list)
    expected_contains: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvalDataset(BaseModel):
    name: str
    description: str = ""
    corpus_dir: str
    cases: list[EvalCase]


def load_dataset(path: Path | str) -> EvalDataset:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)
