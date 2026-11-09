"""R0: regular expressions and rules, written from the train set only (PREREGISTRATION.md s.4).

Finds a number followed by a unit in a sentence with normative wording, reads the comparator from
the words around it, copies the words before "shall/must/..." as the parameter and the sentence
as the quote. It cannot read applicability conditions, table rows without units or counts.
Nothing in it was tuned on validation or test passages.
"""

from __future__ import annotations

import json
import re

from rail_extract.schema import Extraction, Requirement

# Units that occur in the train labels, longest first so "km/h" wins over "km".
UNITS = [
    "m/s2", "m/s²", "m/s", "km/h", "kN/mm", "mm/second", "g/l", "km", "mm", "cm", "kV", "kHz", "Hz",
    "kPa", "Pa", "kN", "kg", "dB(A)", "dB", "mA", "kA", "ppm", "Ohm", "ohm", "°C", "%", "lux",
    "minutes", "minute", "seconds", "second", "hours", "hour", "tons", "ton", "meters", "metres",
    "m2", "m²", "ms", "kW", "W", "V", "A", "N", "t", "l", "m", "s", "h",
]  # fmt: skip
_NUM = r"\d{1,3}(?: \d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_UNIT = "|".join(re.escape(u) for u in UNITS)
_AMOUNT = re.compile(rf"(?<![\w.,])(?P<num>{_NUM})\s?(?P<unit>{_UNIT})(?![\w²])")
_RANGE = re.compile(
    rf"(?<![\w.,])(?P<a>{_NUM})\s?(?:(?P<u1>{_UNIT})\s?)?(?:and|to|-|–)\s?"
    rf"(?P<b>{_NUM})\s?(?P<unit>{_UNIT})(?![\w²])"
)
_NORMATIVE = re.compile(
    r"\b(shall|must|should|minimum|maximum|at least|at most|not exceed|no more than|no less than|"
    r"not less than|not more than|not greater than|higher than|lower than|between)\b|[≥≤]",
    re.I,
)
_MODAL = re.compile(r"\b(shall|must|should|is|are|may|will|needs?)\b", re.I)
_MIN = re.compile(
    r"≥|\bat least\b|\bminimum\b|\bnot less than\b|\bno less than\b|\bnot lower than\b|"
    r"\bor more\b|\bor greater\b|\bor higher\b|\bgreater than or equal\b",
    re.I,
)
_MAX = re.compile(
    r"≤|\bat most\b|\bmaximum\b|\bnot exceed|\bno more than\b|\bnot more than\b|"
    r"\bnot greater than\b|\bnot higher than\b|\bor less\b|\bor lower\b|\bup to\b|"
    r"\bless than or equal\b|\bno greater than\b|\bno higher than\b",
    re.I,
)
_GT = re.compile(
    r"\bmore than\b|\bgreater than\b|\bhigher than\b|\bexceed|\babove\b|\bover\b", re.I
)
_LT = re.compile(r"\bless than\b|\blower than\b|\bbelow\b|\bunder\b|\bshorter than\b", re.I)
_GAUGE_AFTER = re.compile(r"^\s*(track\s+)?(gauge|system|network)", re.I)
_SCOPE_BEFORE = re.compile(
    r"\b(designed for|intended (?:for|to be operated)|operated on|apply|applies|applicable|"
    r"applying)\b",
    re.I,
)
_LEAD = re.compile(r"^(\(\w{1,3}\)\s*)+")
_CLAUSE = re.compile(r"^(for|in the case|where|if|when|in case)\b[^,]*,\s*", re.I)
_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])|\n+")


def to_float(text: str) -> float:
    return float(text.replace(" ", "").replace(",", "."))


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def comparator_for(window: str, tail: str) -> str:
    """Pick the comparator from the words before the number (`window`) and just after it."""
    near = window[-45:]
    for pattern, comp in ((_MAX, "<="), (_MIN, ">=")):
        if pattern.search(near) or pattern.search(tail):
            return comp
    if _GT.search(near):
        return ">"
    if _LT.search(near):
        return "<"
    for pattern, comp in ((_MAX, "<="), (_MIN, ">=")):  # wording earlier in the sentence
        if pattern.search(window):
            return comp
    return "=="


_CUE_WORD = re.compile(r"\b(minimum|maximum|min\.|max\.)\s+(?!of\b)", re.I)
_GENERIC = {"it", "they", "these", "this", "those", "the following", "there", "such"}


def parameter_for(sentence: str, start: int) -> str:
    """A short contiguous piece of the sentence, so it is verbatim: "maximum" or "minimum" with
    the next words if the sentence has them, else the subject before the first modal verb, else
    the words before the number."""
    lead = _LEAD.match(sentence)
    offset = lead.end() if lead else 0
    clause = _CLAUSE.match(sentence[offset:])
    offset += clause.end() if clause else 0
    body = sentence[offset:]
    start -= offset
    cue = _CUE_WORD.search(body)
    if cue:
        return " ".join(body[cue.start() :].split()[:5]).strip(" ,:;()")
    m = _MODAL.search(body)
    head = body[: m.start()] if m and 0 < m.start() < start else body[: max(start, 0)]
    words = head.split()[-6:]
    first = " ".join(words[:2]).lower()
    if not words or words[0].lower() in _GENERIC or first in _GENERIC:
        words = body[: max(start, 0)].split()[-5:]
    return " ".join(words).strip(" ,:;()") or sentence[:40].strip()


def extract(heading: str, text: str) -> Extraction:
    found: list[Requirement] = []
    seen: set[tuple] = set()
    for sent in sentences(text):
        if not _NORMATIVE.search(sent) or "|" in sent:
            continue
        spans = []
        for m in _RANGE.finditer(sent):
            spans.append(
                (m.start(), m.end(), "between", m.group("a"), m.group("b"), m.group("unit"))
            )
        covered = [(s, e) for s, e, *_ in spans]
        for m in _AMOUNT.finditer(sent):
            if any(s <= m.start() < e for s, e in covered):
                continue
            window = sent[: m.start()]
            tail = sent[m.end() : m.end() + 14]
            comp = comparator_for(window, tail)
            if comp in (">", "<") or _GAUGE_AFTER.match(sent[m.end() :]):
                continue  # strict bounds and gauges only scope other rules in the train labels
            if comp == "==" and _SCOPE_BEFORE.search(window[-60:]):
                continue
            spans.append((m.start(), m.end(), comp, m.group("num"), None, m.group("unit")))
        for start, _end, comp, a, b, unit in sorted(spans):
            lo, hi = to_float(a), (to_float(b) if b else None)
            if comp == "between" and hi is not None and hi < lo:
                continue
            key = (comp, lo, hi, unit)
            if key in seen:
                continue
            seen.add(key)
            quote = sent if len(sent) <= 400 else sent[max(0, start - 150) : start + 150]
            found.append(
                Requirement(
                    parameter=parameter_for(sent, start),
                    comparator=comp,
                    value=lo,
                    value_max=hi,
                    unit=unit,
                    conditions=[],
                    quote=quote,
                )
            )
    return Extraction(requirements=found)


def run(passage: dict) -> str:
    """Raw output in the same JSON form an LLM would return."""
    return json.dumps(extract(passage["heading"], passage["text"]).model_dump(), ensure_ascii=False)
