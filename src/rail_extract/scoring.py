"""Scoring exactly as docs/PREREGISTRATION.md section 6. Pure functions, no models.

A prediction file holds one record per passage:
    {"id", "system", "raw", "json_valid", "schema_valid", "requirements": [...]}
`requirements` is empty when the output was not schema-valid (section 5: counts as empty).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from rail_extract.numbers import fidelity_errors, norm_text, norm_unit, numbers_in
from rail_extract.schema import Extraction, Requirement

FIELDS = ("parameter", "comparator", "value", "unit", "conditions")
JACCARD_MIN = 0.5
# General English function words. A fixed list that uses no corpus split, so it cannot leak.
STOP = frozenset(
    "a an and any as at be by each for from in is it its of on or shall should that the this "
    "to which with".split()
)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

Counts = dict[str, list[int]]  # field -> [tp, fp, fn]


def tokens(text: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[^\W_]+", norm_text(text).lower()) if t not in STOP)


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _close(x: float | None, y: float | None) -> bool:
    if x is None or y is None:
        return x is None and y is None
    return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def same_value(g: Requirement, p: Requirement) -> bool:
    return _close(g.value, p.value) and _close(g.value_max, p.value_max)


def same_unit(g: Requirement, p: Requirement) -> bool:
    return norm_unit(g.unit) == norm_unit(p.unit)


def eligible(g: Requirement, p: Requirement) -> bool:
    """Section 6 step 2: value and unit match, or parameter Jaccard of 0.5 or more."""
    return (same_value(g, p) and same_unit(g, p)) or jaccard(
        g.parameter, p.parameter
    ) >= JACCARD_MIN


def _zero() -> Counts:
    return {f: [0, 0, 0] for f in FIELDS}


def _add(into: Counts, other: Counts) -> None:
    for f in FIELDS:
        for i in range(3):
            into[f][i] += other[f][i]


def condition_counts(gold: list[str], pred: list[str]) -> list[int]:
    """Each element is matched at most once, best Jaccard first (greedy, ties by position)."""
    pairs = sorted(
        (
            (-jaccard(g, p), i, j)
            for i, g in enumerate(gold)
            for j, p in enumerate(pred)
            if jaccard(g, p) >= JACCARD_MIN
        )
    )
    used_g: set[int] = set()
    used_p: set[int] = set()
    for _, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
    tp = len(used_g)
    return [tp, len(pred) - tp, len(gold) - tp]


def pair_counts(g: Requirement, p: Requirement) -> Counts:
    c = _zero()
    checks = {
        "parameter": jaccard(g.parameter, p.parameter) >= JACCARD_MIN,
        "comparator": g.comparator == p.comparator,
        "value": same_value(g, p),
    }
    for f, ok in checks.items():
        c[f] = [1, 0, 0] if ok else [0, 1, 1]
    gu, pu = norm_unit(g.unit), norm_unit(p.unit)
    if gu is None and pu is None:
        pass
    elif gu is None:
        c["unit"] = [0, 1, 0]
    elif pu is None:
        c["unit"] = [0, 0, 1]
    else:
        c["unit"] = [1, 0, 0] if gu == pu else [0, 1, 1]
    c["conditions"] = condition_counts(g.conditions, p.conditions)
    return c


def _only(r: Requirement, slot: int) -> Counts:
    """Counts of a requirement with no partner: all its non-null fields are FN (gold) or FP."""
    c = _zero()
    for f in ("parameter", "comparator", "value"):
        c[f][slot] = 1
    if r.unit is not None:
        c["unit"][slot] = 1
    c["conditions"][slot] = len(r.conditions)
    return c


def _weight(c: Counts) -> int:
    return sum(v[0] for v in c.values())


def align(gold: list[Requirement], pred: list[Requirement]) -> list[tuple[int, int]]:
    """Pairs (gold index, pred index) maximising matched fields among eligible pairs.
    Exact (bitmask DP) while the smaller side has at most 12 items, greedy beyond that."""
    w = {
        (i, j): _weight(pair_counts(g, p))
        for i, g in enumerate(gold)
        for j, p in enumerate(pred)
        if eligible(g, p)
    }
    if not w:
        return []
    swap = len(pred) > len(gold)  # bitmask over the smaller side
    rows = len(pred) if swap else len(gold)
    cols = len(gold) if swap else len(pred)
    weight = {((j, i) if swap else (i, j)): v for (i, j), v in w.items()}
    if cols > 12:  # greedy fallback; far beyond any passage in this corpus
        out, ur, uc = [], set(), set()
        for (r, c), _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0])):
            if r not in ur and c not in uc:
                ur.add(r)
                uc.add(c)
                out.append((r, c))
        return [(c, r) if swap else (r, c) for r, c in out]
    best: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}

    def solve(r: int, mask: int) -> tuple[int, list[tuple[int, int]]]:
        if r == rows:
            return 0, []
        key = (r, mask)
        if key in best:
            return best[key]
        top = solve(r + 1, mask)  # leave this row unmatched
        for c in range(cols):
            if not mask & (1 << c) and (r, c) in weight:
                sub, pairs = solve(r + 1, mask | (1 << c))
                if sub + weight[(r, c)] > top[0]:
                    top = (sub + weight[(r, c)], [(r, c)] + pairs)
        best[key] = top
        return top

    pairs = solve(0, 0)[1]
    return [(c, r) if swap else (r, c) for r, c in pairs]


def passage_counts(gold: list[Requirement], pred: list[Requirement]) -> tuple[Counts, list]:
    total = _zero()
    pairs = align(gold, pred)
    for i, j in pairs:
        _add(total, pair_counts(gold[i], pred[j]))
    matched_g = {i for i, _ in pairs}
    matched_p = {j for _, j in pairs}
    for i, g in enumerate(gold):
        if i not in matched_g:
            _add(total, _only(g, 2))
    for j, p in enumerate(pred):
        if j not in matched_p:
            _add(total, _only(p, 1))
    return total, pairs


def coarse(comparator: str) -> str:
    return {">=": "min", ">": "min", "<=": "max", "<": "max", "==": "exact", "between": "range"}[
        comparator
    ]


def verified(requirements: list[Requirement]) -> list[Requirement]:
    """Deterministic verifier: drop requirements whose numbers are absent from their quote."""
    out = []
    for r in requirements:
        stated = numbers_in(r.quote)
        if all(v is None or v in stated for v in (r.value, r.value_max)):
            out.append(r)
    return out


def quote_is_verbatim(quote: str, passage: dict) -> bool:
    body = norm_text(passage["text"]).lower()
    parts = [p for p in re.split(r"\.\.\.|…", quote) if norm_text(p)]
    return bool(parts) and all(norm_text(p).lower() in body for p in parts)


def prf(c: list[int]) -> dict[str, float]:
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------- predictions


def extract_json(raw: str) -> str:
    """The raw text, with one enclosing markdown code fence removed. Nothing else is repaired."""
    m = _FENCE.match(raw)
    return (m.group(1) if m else raw).strip()


def make_prediction(pid: str, system: str, raw: str) -> dict:
    text = extract_json(raw)
    try:
        json.loads(text)
        json_valid = True
    except (json.JSONDecodeError, ValueError):
        json_valid = False
    try:
        ext = Extraction.model_validate_json(text)
    except ValueError:  # pydantic.ValidationError is a ValueError
        ext = None
    return {
        "id": pid,
        "system": system,
        "raw": raw,
        "json_valid": json_valid,
        "schema_valid": ext is not None,
        "requirements": [r.model_dump() for r in ext.requirements] if ext else [],
    }


# -------------------------------------------------------------------- scoring


@dataclass
class PassageScore:
    id: str
    counts: Counts
    counts_verified: Counts
    exact: bool
    n_gold: int
    n_pred: int
    json_valid: bool
    schema_valid: bool
    fidelity_items: int
    fidelity_errors: int
    quotes: int
    quotes_verbatim: int
    coarse_total: int
    coarse_correct: int
    stratum: str = ""

    @property
    def fidelity_error(self) -> bool:
        return self.fidelity_errors > 0

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {
            "fidelity_error": self.fidelity_error
        }


def score_passage(
    passage: dict, gold: list[Requirement], pred_record: dict, stratum: str = ""
) -> PassageScore:
    pred = [Requirement.model_validate(r) for r in pred_record["requirements"]]
    counts, pairs = passage_counts(gold, pred)
    counts_v, _ = passage_counts(gold, verified(pred))
    text = passage["heading"] + " " + passage["text"]
    items = sum(
        (r.value is not None) + (r.value_max is not None) + (r.unit is not None) for r in pred
    )
    errors = len(fidelity_errors(Extraction(requirements=pred), text)) if pred else 0
    return PassageScore(
        id=passage["id"],
        counts=counts,
        counts_verified=counts_v,
        exact=all(v[1] == 0 and v[2] == 0 for v in counts.values()),
        n_gold=len(gold),
        n_pred=len(pred),
        json_valid=bool(pred_record["json_valid"]),
        schema_valid=bool(pred_record["schema_valid"]),
        fidelity_items=items,
        fidelity_errors=errors,
        quotes=len(pred),
        quotes_verbatim=sum(quote_is_verbatim(r.quote, passage) for r in pred),
        coarse_total=len(pairs),
        coarse_correct=sum(
            coarse(gold[i].comparator) == coarse(pred[j].comparator) for i, j in pairs
        ),
        stratum=stratum,
    )


def _sum_counts(scores: list[PassageScore], attr: str = "counts") -> Counts:
    total = _zero()
    for s in scores:
        _add(total, getattr(s, attr))
    return total


def _overall(counts: Counts) -> list[int]:
    return [sum(counts[f][i] for f in FIELDS) for i in range(3)]


def _rate(num: float, den: float) -> float | None:
    return num / den if den else None


@dataclass
class Summary:
    n: int
    f1: float
    precision: float
    recall: float
    per_field: dict[str, dict[str, float]]
    exact_match: float
    fidelity_error_passages: float
    fidelity_error_values: float | None
    json_valid: float
    schema_valid: float
    false_extraction: float | None
    false_extraction_by_stratum: dict[str, float | None] = field(default_factory=dict)
    verbatim_quote: float | None = None
    coarse_comparator: float | None = None
    f1_verified: float = 0.0
    empty_passages: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def summarise(scores: list[PassageScore]) -> Summary:
    counts = _sum_counts(scores)
    overall = prf(_overall(counts))
    empty = [s for s in scores if s.n_gold == 0]
    strata = sorted({s.stratum for s in empty})
    n = len(scores)
    return Summary(
        n=n,
        f1=overall["f1"],
        precision=overall["precision"],
        recall=overall["recall"],
        per_field={f: prf(counts[f]) for f in FIELDS},
        exact_match=sum(s.exact for s in scores) / n,
        fidelity_error_passages=sum(s.fidelity_error for s in scores) / n,
        fidelity_error_values=_rate(
            sum(s.fidelity_errors for s in scores), sum(s.fidelity_items for s in scores)
        ),
        json_valid=sum(s.json_valid for s in scores) / n,
        schema_valid=sum(s.schema_valid for s in scores) / n,
        false_extraction=_rate(sum(s.n_pred > 0 for s in empty), len(empty)),
        false_extraction_by_stratum={
            st: _rate(
                sum(s.n_pred > 0 for s in empty if s.stratum == st),
                sum(1 for s in empty if s.stratum == st),
            )
            for st in strata
        },
        verbatim_quote=_rate(sum(s.quotes_verbatim for s in scores), sum(s.quotes for s in scores)),
        coarse_comparator=_rate(
            sum(s.coarse_correct for s in scores), sum(s.coarse_total for s in scores)
        ),
        f1_verified=prf(_overall(_sum_counts(scores, "counts_verified")))["f1"],
        empty_passages=len(empty),
    )
