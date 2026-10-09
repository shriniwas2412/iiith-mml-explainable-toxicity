"""Shared constants and config loading."""

from __future__ import annotations

from pathlib import Path

import yaml

# Label order used everywhere: CSV columns, model outputs, reports.
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

ID_COL = "id"
TEXT_COL = "comment_text"

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML config. Falls back to configs/default.yaml."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
