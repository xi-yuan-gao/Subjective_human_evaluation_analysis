from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({str(key).strip(): (value or "").strip() for key, value in row.items()})
        return rows


def read_folder(folder: Path, list_id: int, source_kind: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    inventory: list[dict[str, Any]] = []
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    for path in files:
        file_rows = read_csv(path)
        for row_number, row in enumerate(file_rows, start=2):
            enriched = dict(row)
            if "list_id" in row:
                enriched["reported_list_id"] = row["list_id"]
            enriched["list_id"] = str(list_id)
            enriched["source_file"] = str(path)
            enriched["source_row"] = str(row_number)
            rows.append(enriched)
        inventory.append(
            {
                "list_id": list_id,
                "source_kind": source_kind,
                "source_file": str(path),
                "row_count": len(file_rows),
                "columns": " | ".join(file_rows[0].keys()) if file_rows else "",
            }
        )
    if not files:
        inventory.append(
            {
                "list_id": list_id,
                "source_kind": source_kind,
                "source_file": "",
                "row_count": 0,
                "columns": "",
            }
        )
    return rows, inventory


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in materialized:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "t"}:
        return True
    if normalized in {"false", "0", "no", "n", "f"}:
        return False
    return None


def participant_session_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("participant_id", "")).strip(), str(row.get("session_id", "")).strip())

