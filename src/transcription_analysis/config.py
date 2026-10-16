from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = [
        "list_ids",
        "list_directory_pattern",
        "column_aliases",
        "normalization",
        "questionnaire",
    ]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config is missing required keys: {', '.join(missing)}")
    return config

