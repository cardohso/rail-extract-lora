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
_UNIT_ALIASES = {"m/s2": "m/s²", "m2": "m²", "cm2": "cm²", "mm2": "mm²", "km/hr": "km/h"}


def norm_text(text: str) -> str:
    text = _SPACES.sub(" ", text.translate(_DASHES))
    return re.sub(r"\s+", " ", text).strip()


def _is_sign(text: str, start: int) -> bool:
    """Is the "-" just before a number at `start` a minus sign, not a range dash ("15 - 21")?"""
    before = text[:start].rstrip()
    if not before.endswith("-"):
        return False
    prior = before[:-1].rstrip()
    return not prior or not (prior[-1].isdigit() or prior[-1] in "%°)")


def numbers_in(text: str) -> set[float]:
    """Numbers written in the text. "1 000" is 1000, "0,5" is 0.5, "two" is 2, "– 25" is -25.

    Lines and table cells ("|") are read separately, so the last number of one row never merges
    with the first of the next. A number after a minus sign is stored both ways, so a positive
    reading never fails."""
    found: set[float] = set()
    for segment in re.split(r"\n|\|", text):
        seg = norm_text(segment)
        for m in _NUMBER.finditer(seg):
            value = float(m.group(0).replace(" ", "").replace(",", "."))
            found.add(value)
            if _is_sign(seg, m.start()):
                found.add(-value)
        found |= {_WORDS[w] for w in _WORD.findall(seg.lower())}
    return found


def norm_unit(unit: str | None) -> str | None:
    """Lower case, single spaces, no spaces around "/"; "m2" style aliases mapped."""
    if unit is None:
        return None
    u = re.sub(r"\s*/\s*", "/", norm_text(unit).lower())
    return _UNIT_ALIASES.get(u, u)


def fidelity_errors(extraction: Extraction, passage: str) -> list[str]:
    """Output values or units that the passage does not contain. Empty means faithful."""
    stated = numbers_in(passage)
    text = re.sub(r"\s*/\s*", "/", norm_text(passage).lower())
    for alias, canonical in _UNIT_ALIASES.items():  # whole tokens only: "m2" is not in "cm2"
        text = re.sub(rf"(?<![^\W\d_]){re.escape(alias)}(?![^\W\d_])", canonical, text)
    errors = []
    for r in extraction.requirements:
        for v in (r.value, r.value_max):
            if v is not None and v not in stated:
                errors.append(f"value {v:g}")
        unit = norm_unit(r.unit)
        if unit and not re.search(rf"(?<![^\W\d_]){re.escape(unit)}(?![^\W\d_])", text):
            errors.append(f"unit {r.unit}")
    return errors
