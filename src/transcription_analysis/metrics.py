from __future__ import annotations

from typing import Any, Sequence


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> dict[str, int | float | None]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    distance = [[0] * cols for _ in range(rows)]
    operation = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        distance[i][0] = i
        operation[i][0] = "D"
    for j in range(1, cols):
        distance[0][j] = j
        operation[0][j] = "I"
    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                operation[i][j] = "E"
                continue
            candidates = [
                (distance[i - 1][j - 1] + 1, "S"),
                (distance[i - 1][j] + 1, "D"),
                (distance[i][j - 1] + 1, "I"),
            ]
            distance[i][j], operation[i][j] = min(candidates, key=lambda item: item[0])
    substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        op = operation[i][j]
        if op == "E":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:
            break
    errors = substitutions + deletions + insertions
    denominator = len(reference)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "reference_length": denominator,
        "error_rate": (errors / denominator) if denominator else None,
    }


def edit_counts_with_wildcards(
    reference: Sequence[str],
    hypothesis: Sequence[str],
    wildcard: str,
) -> dict[str, Any]:
    """Calculate edit counts while wildcard reference spans absorb any words.

    A wildcard may match zero or more hypothesis tokens at no cost. Returned
    evaluated sequences omit the wildcard and all hypothesis tokens it absorbs;
    these sequences can also be used for name-excluded character scoring.
    """
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    infinity = len(reference) + len(hypothesis) + 1
    # The second tuple value is negative so equal-error paths prefer absorbing
    # more hypothesis tokens inside the excluded name span.
    distance = [[(infinity, 0)] * cols for _ in range(rows)]
    predecessor: list[list[tuple[int, int, str] | None]] = [
        [None] * cols for _ in range(rows)
    ]
    distance[0][0] = (0, 0)

    def relax(
        source_i: int,
        source_j: int,
        target_i: int,
        target_j: int,
        cost: int,
        operation: str,
    ) -> None:
        source_errors, source_absorbed = distance[source_i][source_j]
        candidate = (
            source_errors + cost,
            source_absorbed - (1 if operation == "W1" else 0),
        )
        if candidate < distance[target_i][target_j]:
            distance[target_i][target_j] = candidate
            predecessor[target_i][target_j] = (source_i, source_j, operation)

    for i in range(rows):
        for j in range(cols):
            if distance[i][j][0] == infinity:
                continue
            if i < len(reference) and reference[i] == wildcard:
                relax(i, j, i + 1, j, 0, "W0")
                if j < len(hypothesis):
                    relax(i, j, i, j + 1, 0, "W1")
                continue
            if i < len(reference) and j < len(hypothesis):
                operation = "E" if reference[i] == hypothesis[j] else "S"
                relax(i, j, i + 1, j + 1, 0 if operation == "E" else 1, operation)
            if i < len(reference):
                relax(i, j, i + 1, j, 1, "D")
            if j < len(hypothesis):
                relax(i, j, i, j + 1, 1, "I")

    substitutions = deletions = insertions = 0
    evaluated_reference: list[str] = []
    evaluated_hypothesis: list[str] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        previous = predecessor[i][j]
        if previous is None:
            raise RuntimeError("Wildcard edit-distance backtrace is incomplete.")
        previous_i, previous_j, operation = previous
        if operation in {"E", "S"}:
            evaluated_reference.append(reference[previous_i])
            evaluated_hypothesis.append(hypothesis[previous_j])
            if operation == "S":
                substitutions += 1
        elif operation == "D":
            evaluated_reference.append(reference[previous_i])
            deletions += 1
        elif operation == "I":
            evaluated_hypothesis.append(hypothesis[previous_j])
            insertions += 1
        i, j = previous_i, previous_j

    evaluated_reference.reverse()
    evaluated_hypothesis.reverse()
    errors = substitutions + deletions + insertions
    denominator = len(evaluated_reference)
    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "errors": errors,
        "reference_length": denominator,
        "error_rate": (errors / denominator) if denominator else None,
        "evaluated_reference": evaluated_reference,
        "evaluated_hypothesis": evaluated_hypothesis,
    }

