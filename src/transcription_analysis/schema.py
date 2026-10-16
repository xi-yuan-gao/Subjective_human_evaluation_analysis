from __future__ import annotations

from collections import defaultdict
from typing import Any


REQUIRED_MAIN_FIELDS = [
    "participant_id",
    "session_id",
    "utterance_id",
    "condition",
    "speaker_id",
    "reference_text",
    "transcription",
]


def resolve_columns(rows: list[dict[str, Any]], aliases: dict[str, list[str]]) -> dict[str, str]:
    present: set[str] = set()
    for row in rows:
        present.update(row.keys())
    lowered = {name.lower(): name for name in present}
    resolved: dict[str, str] = {}
    for canonical, options in aliases.items():
        for option in options:
            if option.lower() in lowered:
                resolved[canonical] = lowered[option.lower()]
                break
    return resolved


def canonicalize(rows: list[dict[str, Any]], mapping: dict[str, str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    reverse = {source: canonical for canonical, source in mapping.items()}
    for row in rows:
        canonical = dict(row)
        for source, target in reverse.items():
            canonical[target] = row.get(source, "")
        output.append(canonical)
    return output


def duplicate_groups(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, group in groups.items():
        if len(group) <= 1:
            continue
        base = {key: value for key, value in zip(keys, key_values)}
        for row in group:
            output.append(
                {
                    **base,
                    "duplicate_count": len(group),
                    "source_file": row.get("source_file", ""),
                    "source_row": row.get("source_row", ""),
                }
            )
    return output

