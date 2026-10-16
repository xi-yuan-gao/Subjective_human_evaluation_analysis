from __future__ import annotations

from collections import defaultdict
from statistics import mean, median, pstdev
from typing import Any


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_scores(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    two_play_threshold: float = 2.0,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_values, group in sorted(groups.items()):
        wer = [value for value in (_number(row.get("wer")) for row in group) if value is not None]
        cer = [value for value in (_number(row.get("cer")) for row in group) if value is not None]
        response_times = [value for value in (_number(row.get("response_time_sec")) for row in group) if value is not None]
        plays = [value for value in (_number(row.get("plays_used")) for row in group) if value is not None]
        total_ref_words = sum(int(row.get("reference_words", 0) or 0) for row in group)
        total_ref_chars = sum(int(row.get("reference_characters", 0) or 0) for row in group)
        substitutions = sum(int(row.get("word_substitutions", 0) or 0) for row in group)
        deletions = sum(int(row.get("word_deletions", 0) or 0) for row in group)
        insertions = sum(int(row.get("word_insertions", 0) or 0) for row in group)
        result: dict[str, Any] = {key: value for key, value in zip(group_keys, group_values)}
        result.update(
            {
                "n_trials": len(group),
                "n_participants": len({row.get("participant_id", "") for row in group}),
                "n_utterances": len({row.get("utterance_id", "") for row in group}),
                "mean_wer": mean(wer) if wer else "",
                "median_wer": median(wer) if wer else "",
                "sd_wer": pstdev(wer) if len(wer) > 1 else 0 if wer else "",
                "mean_cer": mean(cer) if cer else "",
                "median_cer": median(cer) if cer else "",
                "micro_wer": ((substitutions + deletions + insertions) / total_ref_words) if total_ref_words else "",
                "substitution_rate": substitutions / total_ref_words if total_ref_words else "",
                "deletion_rate": deletions / total_ref_words if total_ref_words else "",
                "insertion_rate": insertions / total_ref_words if total_ref_words else "",
                "xxx_trial_rate": sum(int(row.get("contains_xxx", 0)) for row in group) / len(group),
                "whole_utterance_xxx_rate": sum(int(row.get("whole_utterance_xxx", 0)) for row in group) / len(group),
                "perfect_word_transcription_rate": sum(float(row.get("wer", 1)) == 0 for row in group) / len(group),
                "perfect_character_transcription_rate": sum(float(row.get("cer", 1)) == 0 for row in group) / len(group),
                "median_response_time_sec": median(response_times) if response_times else "",
                "mean_response_time_sec": mean(response_times) if response_times else "",
                "median_plays_used": median(plays) if plays else "",
                "mean_plays_used": mean(plays) if plays else "",
                "two_or_more_plays_rate": sum(value >= two_play_threshold for value in plays) / len(plays) if plays else "",
                "timing_observation_count": len(response_times),
                "plays_observation_count": len(plays),
                "total_reference_words": total_ref_words,
                "total_reference_characters": total_ref_chars,
            }
        )
        output.append(result)
    return output


def summarize_xxx(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scopes = {
        "condition": ["condition"],
        "speaker": ["speaker_id"],
        "speaker_condition": ["speaker_id", "condition"],
        "listener": ["participant_id", "list_id"],
        "listener_condition": ["participant_id", "list_id", "condition"],
    }
    output: list[dict[str, Any]] = []
    for scope, keys in scopes.items():
        grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)
        for values, group in sorted(grouped.items()):
            result: dict[str, Any] = {
                "scope": scope,
                "n_trials": len(group),
                "trials_with_xxx": sum(int(row.get("contains_xxx", 0)) for row in group),
                "whole_utterance_xxx_trials": sum(int(row.get("whole_utterance_xxx", 0)) for row in group),
                "total_xxx_tokens": sum(int(row.get("xxx_token_count", 0)) for row in group),
            }
            result.update({key: value for key, value in zip(keys, values)})
            result["xxx_trial_rate"] = result["trials_with_xxx"] / len(group)
            result["whole_utterance_xxx_rate"] = result["whole_utterance_xxx_trials"] / len(group)
            result["mean_xxx_tokens_per_trial"] = result["total_xxx_tokens"] / len(group)
            output.append(result)
    return output


def summarize_plays_wer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        plays = _number(row.get("plays_used"))
        if plays is None:
            continue
        groups[(str(row.get("condition", "")), str(int(plays)) if plays.is_integer() else str(plays))].append(row)
    output: list[dict[str, Any]] = []
    for (condition, plays_used), group in sorted(groups.items()):
        wer = [value for value in (_number(row.get("wer")) for row in group) if value is not None]
        cer = [value for value in (_number(row.get("cer")) for row in group) if value is not None]
        response_times = [value for value in (_number(row.get("response_time_sec")) for row in group) if value is not None]
        output.append(
            {
                "condition": condition,
                "plays_used": plays_used,
                "n_trials": len(group),
                "n_participants": len({row.get("participant_id", "") for row in group}),
                "mean_wer": mean(wer) if wer else "",
                "median_wer": median(wer) if wer else "",
                "mean_cer": mean(cer) if cer else "",
                "median_cer": median(cer) if cer else "",
                "median_response_time_sec": median(response_times) if response_times else "",
            }
        )
    return output

