from __future__ import annotations

import re
import string
import unicodedata
from typing import Any


APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
EXCLUDED_NAME_TOKEN = "<excluded_name>"


def normalize_text(text: str, settings: dict[str, Any]) -> str:
    value = str(text or "")
    if settings.get("unicode_nfkc", True):
        value = unicodedata.normalize("NFKC", value)
    if settings.get("standardize_apostrophes", True):
        value = value.translate(APOSTROPHES)
    if settings.get("lowercase", True):
        value = value.lower()
    if settings.get("remove_apostrophes", False):
        value = value.replace("'", "")
    if settings.get("remove_punctuation", True):
        punctuation = string.punctuation.replace("'", "")
        value = value.translate(str.maketrans({character: " " for character in punctuation}))
        value = "".join(" " if unicodedata.category(character).startswith("P") and character != "'" else character for character in value)
    if settings.get("collapse_whitespace", True):
        value = re.sub(r"\s+", " ", value).strip()
    word_map = {str(k).lower(): str(v).lower() for k, v in settings.get("word_map", {}).items()}
    if word_map:
        value = " ".join(word_map.get(token, token) for token in value.split())
    return value


def collapse_excluded_name_phrases(
    normalized_text: str,
    phrases: list[str],
) -> tuple[list[str], list[str], int]:
    """Collapse configured personal-name spans into wildcard tokens.

    Phrases are matched only in the normalized reference. Longest phrases take
    precedence so a full name such as ``jim shooter`` becomes one wildcard.
    """
    tokens = normalized_text.split()
    candidates = sorted(
        ((phrase, phrase.split()) for phrase in phrases if phrase.strip()),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    collapsed: list[str] = []
    found: list[str] = []
    excluded_token_count = 0
    index = 0
    while index < len(tokens):
        match = next(
            (
                (phrase, phrase_tokens)
                for phrase, phrase_tokens in candidates
                if tokens[index : index + len(phrase_tokens)] == phrase_tokens
            ),
            None,
        )
        if match is None:
            collapsed.append(tokens[index])
            index += 1
            continue
        phrase, phrase_tokens = match
        collapsed.append(EXCLUDED_NAME_TOKEN)
        found.append(phrase)
        excluded_token_count += len(phrase_tokens)
        index += len(phrase_tokens)
    return collapsed, found, excluded_token_count


def character_sequence(text: str, remove_spaces: bool = True) -> list[str]:
    if remove_spaces:
        text = text.replace(" ", "")
    return list(text)

