"""Compare two drafters and decide what the author must review (train and validation).

    uv run python -m rail_extract.agree val       # writes auto-agreed labels and the queue

Protocol, fixed before any second-drafter output existed:
- Two drafts agree when they have the same requirements in the same (comparator, value,
  value_max, unit) after unit normalisation, in any order. Two empty drafts agree.
- A passage goes to the author if the drafts do not agree, or if BOTH drafters marked it unsure.
- Agreed passages are saved as labels with source "auto-agreed" (drafter a's text), except for a
  random 10 % audit sample, which stays in the author's queue. The audit measures how often
  agreement hides an error.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from rail_extract import labels as L
from rail_extract.numbers import norm_unit

AUDIT_RATE = 0.10
SEED = 20261108


def core(req: dict) -> tuple:
    return (req["comparator"], req["value"], req["value_max"], norm_unit(req["unit"]))


def agree(a: dict, b: dict) -> bool:
    return Counter(map(core, a["requirements"])) == Counter(map(core, b["requirements"]))


def needs_review(a: dict, b: dict) -> bool:
    return not agree(a, b) or (bool(a["unsure"]) and bool(b["unsure"]))


def decide(split: str, root: Path = L.ROOT) -> dict:
    da, db = L.read_drafts(split, "a", root), L.read_drafts(split, "b", root)
    ids = sorted(set(da) & set(db))
    rng = random.Random(f"{SEED}:{split}")
    labels = L.read_labels(split, root)
    review, auto, audit = [], [], []
    for pid in ids:
        if needs_review(da[pid], db[pid]):
            review.append(pid)
        elif rng.random() < AUDIT_RATE:
            audit.append(pid)
        else:
            auto.append(pid)
    for pid in auto:
        if pid not in labels:
            labels[pid] = L.Label(
                id=pid,
                split=split,
                requirements=da[pid]["requirements"],
                unsure=False,
                source="auto-agreed",
            )
    L.write_labels(split, labels, root)
    out = {
        "split": split,
        "compared": len(ids),
        "only_one_draft": sorted((set(da) | set(db)) - set(ids)),
        "review": review,
        "audit": audit,
        "auto_agreed": auto,
        "digest": hashlib.sha256(json.dumps([review, audit, auto]).encode()).hexdigest()[:12],
    }
    path = root / "data/review_queue" / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = decide(sys.argv[1])
    print(
        f"{r['split']}: compared {r['compared']}, review {len(r['review'])}, "
        f"audit {len(r['audit'])}, auto-agreed {len(r['auto_agreed'])}, "
        f"missing a draft {len(r['only_one_draft'])}"
    )
