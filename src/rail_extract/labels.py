"""Label files, the labelling queue and label checks. No model is involved.

Labels live in data/labels/<split>.jsonl (one line per passage). Drafts live in
data/drafts/<split>.jsonl and are only suggestions shown in the labelling tool.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rail_extract.candidates import STRATA
from rail_extract.numbers import fidelity_errors, norm_text
from rail_extract.schema import Extraction, Requirement
from rail_extract.splits import QUOTAS

ROOT = Path(__file__).resolve().parents[2]
SPLITS = tuple(QUOTAS)
MAX_SECONDS_PER_PASSAGE = 600  # a passage left open overnight must not inflate the labelling time


class Label(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    split: str
    requirements: list[Requirement] = Field(default_factory=list)
    unsure: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    seconds: float = 0.0


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def read_labels(split: str, root: Path = ROOT) -> dict[str, Label]:
    return {
        r["id"]: Label.model_validate(r)
        for r in read_jsonl(root / "data/labels" / f"{split}.jsonl")
    }


def write_labels(split: str, labels: dict[str, Label], root: Path = ROOT) -> None:
    """Rewrite the file atomically, so a crash never leaves half a line."""
    path = root / "data/labels" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "".join(labels[k].model_dump_json() + "\n" for k in sorted(labels)), encoding="utf-8"
    )
    os.replace(tmp, path)


def read_drafts(split: str, root: Path = ROOT) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(root / "data/drafts" / f"{split}.jsonl")}


def _order_key(pid: str) -> str:
    return hashlib.sha1(pid.encode()).hexdigest()


def active_ids(manifest: list[dict], split: str, labels: dict[str, Label]) -> list[str]:
    """The passages that make up the split's sample right now: per stratum, the first `quota`
    passages in rank order that were not skipped. A skipped passage is replaced by the next
    reserve of its stratum."""
    out: list[str] = []
    for s in STRATA:
        rows = sorted(
            (m for m in manifest if m["split"] == split and m["stratum"] == s),
            key=lambda m: m["rank"],
        )
        kept = 0
        for m in rows:
            if kept >= QUOTAS[split][s]:
                break
            lab = labels.get(m["id"])
            if lab is not None and lab.skipped:
                continue
            out.append(m["id"])
            kept += 1
    return out


def next_pending(manifest: list[dict], split: str, labels: dict[str, Label]) -> str | None:
    """Unlabelled active passages come in a fixed pseudo-random order, strata mixed."""
    pending = [i for i in active_ids(manifest, split, labels) if i not in labels]
    return min(pending, key=_order_key) if pending else None


def progress(manifest: list[dict], split: str, labels: dict[str, Label]) -> dict:
    active = active_ids(manifest, split, labels)
    own = [labels[i] for i in labels if i in set(active)]
    return {
        "target": len(active),
        "done": len(own),
        "unsure": sum(x.unsure for x in own),
        "skipped": sum(x.skipped for x in labels.values()),
        "seconds": round(sum(x.seconds for x in labels.values())),
    }


def check_requirements(passage: dict, requirements: list[Requirement]) -> list[str]:
    """Warnings only; the author decides. Quotes, parameters and conditions must be verbatim, and
    every number and unit must be in the passage (docs/LABELLING.md section 3)."""
    text = norm_text(passage["heading"] + " " + passage["text"]).lower()
    body = norm_text(passage["text"]).lower()
    warnings = []
    for n, r in enumerate(requirements, 1):
        parts = [p for p in re.split(r"\.\.\.|…", r.quote) if norm_text(p)]
        if not parts or not all(norm_text(p).lower() in body for p in parts):
            warnings.append(f"#{n} quote is not verbatim in the passage")
        if norm_text(r.parameter).lower() not in text:
            warnings.append(f"#{n} parameter '{r.parameter}' is not copied from the passage")
        for c in r.conditions:
            if norm_text(c).lower() not in text:
                warnings.append(f"#{n} condition '{c}' is not copied from the passage")
        for err in fidelity_errors(Extraction(requirements=[r]), passage["text"]):
            warnings.append(f"#{n} {err} is not in the passage")
    return warnings
