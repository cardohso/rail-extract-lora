"""Number reading and the number-fidelity check (docs/PREREGISTRATION.md section 6)."""

from __future__ import annotations

import re

from rail_extract.schema import Extraction

_SPACES = re.compile(r"[   ]")
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-"})
# "1 000", "0,5", "3.47": a space between groups of three digits is a thousands separator.
_NUMBER = re.compile(r"(?<![\w.,])\d{1,3}(?: \d{3})+(?:[.,]\d+)?|(?<![\w.,])\d+(?:[.,]\d+)?")
_WORDS = {
    w: float(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split()
    )
}
_WORD = re.compile(r"\b(" + "|".join(_WORDS) + r")\b")
_UNIT_ALIASES = {"m/s2": "m/s²", "m2": "m²", "mm2": "mm²", "km/hr": "km/h", "°c": "°c"}


def norm_text(text: str) -> str:
    text = _SPACES.sub(" ", text.translate(_DASHES))
    return re.sub(r"\s+", " ", text).strip()


def numbers_in(text: str) -> set[float]:
    """Numbers written in the text. "1 000" is 1000, "0,5" is 0.5, "two" is 2."""
    text = norm_text(text)
    found = {float(t.replace(" ", "").replace(",", ".")) for t in _NUMBER.findall(text)}
    return found | {_WORDS[w] for w in _WORD.findall(text.lower())}


def norm_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = norm_text(unit).lower().replace(" ", "")
    return _UNIT_ALIASES.get(u, u)


def fidelity_errors(extraction: Extraction, passage: str) -> list[str]:
    """Output values or units that the passage does not contain. Empty means faithful."""
    stated = numbers_in(passage)
    text = norm_text(passage).lower()
    for alias, canonical in _UNIT_ALIASES.items():
        text = text.replace(alias, canonical)
    errors = []
    for r in extraction.requirements:
        for v in (r.value, r.value_max):
            if v is not None and v not in stated:
                errors.append(f"value {v:g}")
        unit = norm_unit(r.unit)
        if unit and not re.search(rf"(?<![^\W\d_]){re.escape(unit)}(?![^\W\d_])", text):
            errors.append(f"unit {r.unit}")
    return errors
