from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def summarize_questionnaire(
    rows: list[dict[str, Any]],
    categorical_columns: list[str],
    free_text_columns: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    comments: list[dict[str, Any]] = []
    for question in categorical_columns:
        for scope, scoped_rows in [("overall", rows), *[(f"list_{list_id}", [row for row in rows if str(row.get("list_id")) == str(list_id)]) for list_id in sorted({row.get("list_id") for row in rows})]]:
            values = [str(row.get(question, "")).strip() for row in scoped_rows if str(row.get(question, "")).strip()]
            counts = Counter(values)
            for response, count in sorted(counts.items()):
                summaries.append(
                    {
                        "scope": scope,
                        "question": question,
                        "response": response,
                        "count": count,
                        "valid_responses": len(values),
                        "percent": count / len(values) if values else "",
                    }
                )
    for row in rows:
        for column in free_text_columns:
            value = str(row.get(column, "")).strip()
            if value:
                comments.append(
                    {
                        "list_id": row.get("list_id", ""),
                        "participant_id": row.get("participant_id", ""),
                        "session_id": row.get("session_id", ""),
                        "question": column,
                        "response": value,
                        "source_file": row.get("source_file", ""),
                    }
                )
    return summaries, comments

