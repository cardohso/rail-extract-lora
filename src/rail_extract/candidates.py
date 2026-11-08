"""Sampling strata. A heuristic, used to choose which passages to label, never as a label."""

from __future__ import annotations

import re

_UNIT = (
    r"(?:mm2|mm²|mm|cm|km/h|km|m/s²|m/s2|m/s|m²|m2|dB\s?\(?A?\)?|kV|V|kN|N|kg|t|Hz|kHz|MHz|GHz"
    r"|lux|lx|%|°C|°|ms|s|min|hours?|seconds?|minutes?|h|m|A|kW|MW|W|Pa|kPa|bar|l|‰)"
)
NUMBER_WITH_UNIT = re.compile(r"(?<![\w.,])\d[\d ,.]*\s?" + _UNIT + r"(?![\w/²])")
NORMATIVE = re.compile(
    r"\b(minimum|maximum|at least|exceed|no more than|shall|less than|greater than|more than"
    r"|between|up to|longer than|above|below|lower than|higher than|not be|range)\b",
    re.IGNORECASE,
)
ANY_DIGIT = re.compile(r"\d")

MATCH, NUMERIC, NONE = "match", "numeric", "none"
STRATA = (MATCH, NUMERIC, NONE)


def stratum(text: str) -> str:
    if NUMBER_WITH_UNIT.search(text) and NORMATIVE.search(text):
        return MATCH
    return NUMERIC if ANY_DIGIT.search(text) else NONE
