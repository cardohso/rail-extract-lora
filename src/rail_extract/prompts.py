"""Prompts for the prompting systems (B1, B2, B3, A1) and the training format of the LoRA systems.

One instruction text per variant. Few-shot examples come from the train split only, in a fixed
order, so k=3, 5, 8 are nested. Variants are tuned on validation only (PREREGISTRATION.md section 4;
at most 8 for the small model, 3 for A1).
"""

from __future__ import annotations

import hashlib
import json

from rail_extract.schema import flat_json_schema

INSTRUCTION_V1 = """You extract measurable requirements from passages of EU railway technical \
specifications (TSI). A requirement is a limit or permission that a design, test result or \
measurement can be checked against: a parameter, a comparator, one or two numbers and \
usually a unit.

Answer with JSON only, matching this schema:
{schema}

If the passage states no measurable requirement, answer {{"requirements": []}}.

Fields:
- parameter: short phrase (3 to 8 words) copied from the passage that names what is limited.
- comparator: ">=" (at least, minimum, not less than), "<=" (at most, maximum, shall not exceed, \
no more than, "up to"), ">" (more than, greater than), "<" (less than, lower than), "==" (exactly \
or equal to), "between" (a range; value is the lower and value_max the upper number). Use the \
boundary exactly as written: "lower than 250 km/h" is "<", never "<=".
- value, value_max: the number as stated ("1 000" is 1000, "0,5" is 0.5). Do not convert units or \
round. value_max is null unless the comparator is "between".
- unit: as written in the passage; null only for plain counts ("at least two spaces").
- conditions: phrases copied from the passage that say when the rule applies (a case, a \
country, a speed range, "if X is provided"). Empty list if none.
- quote: the shortest stretch of the passage, copied word for word, that states the rule and \
contains its number(s).

Rules:
- One requirement per number-bearing rule: "500 mm wide and 700 mm long" is two. A range \
"between 800 mm and 1 000 mm" is one. A permission ("may be raised up to 89 dB") is its own \
requirement, with the case that allows it in conditions. A table row that pairs a case with a \
limit is one requirement, with the case in conditions. Alternative values are one "==" \
requirement each.
- Limits without "shall" count ("does not exceed 0,05 Ohm"). Counts of things the design must \
have count (unit null).
- These are NOT requirements: scope sentences ("applies to tunnels of more than 1 km"), \
definitions, dates, years, versions, article, point and index numbers, reference or calculation \
data, thresholds that only govern a calculation method or paperwork, bare references to other \
specifications, exemptions with no value.
- Never invent a number or unit that is not in the passage."""

PROMPTS = {"v1": INSTRUCTION_V1}


def system_prompt(variant: str = "v1") -> str:
    return PROMPTS[variant].format(schema=json.dumps(flat_json_schema(), separators=(",", ":")))


def user_message(passage: dict) -> str:
    return f"Heading: {passage['heading']}\n\nPassage:\n{passage['text']}"


def answer(requirements: list) -> str:
    """The expected answer for a labelled passage (also the training target in step 6)."""
    return json.dumps(
        {"requirements": [r.model_dump() if hasattr(r, "model_dump") else r for r in requirements]},
        ensure_ascii=False,
    )


def _order(pid: str) -> str:
    return hashlib.sha1(("shots:" + pid).encode()).hexdigest()


def select_examples(train_rows: list[dict], max_words: int = 140) -> list[dict]:
    """Fixed, nested order of few-shot examples drawn from train rows (with `gold`).

    The first three are one empty case, one with conditions and one with several requirements, so
    every k >= 3 contains all three kinds (PREREGISTRATION.md section 4). The rest follow in hash
    order, alternating positive and empty."""
    short = sorted(
        (r for r in train_rows if len(r["text"].split()) <= max_words),
        key=lambda r: _order(r["id"]),
    )
    empty = [r for r in short if not r["gold"]]
    cond = [r for r in short if any(g.conditions for g in r["gold"]) and len(r["gold"]) <= 3]
    multi = [r for r in short if len(r["gold"]) >= 2 and r not in cond]
    chosen = [empty[0], cond[0], multi[0]]
    ids = {r["id"] for r in chosen}
    pos = [r for r in short if r["gold"] and r["id"] not in ids]
    neg = [r for r in empty[1:] if r["id"] not in ids]
    for p, n in zip(pos, neg, strict=False):
        for r in (p, n):
            if r["id"] not in ids:
                chosen.append(r)
                ids.add(r["id"])
    return chosen


def build_messages(passage: dict, variant: str = "v1", examples: list[dict] | None = None) -> list:
    msgs = [{"role": "system", "content": system_prompt(variant)}]
    for ex in examples or []:
        msgs.append({"role": "user", "content": user_message(ex)})
        msgs.append({"role": "assistant", "content": answer(ex["gold"])})
    msgs.append({"role": "user", "content": user_message(passage)})
    return msgs


def prompt_hash(variant: str, examples: list[dict] | None) -> str:
    blob = system_prompt(variant) + "".join(r["id"] for r in examples or [])
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
