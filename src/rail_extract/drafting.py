"""Helpers for drafting labels with an LLM session. Train and validation only.

    uv run python -m rail_extract.drafting show val 0 12     # print 12 passages still to draft
    uv run python -m rail_extract.drafting save val FILE     # check and store drafts

FILE is {"<passage id>": {"requirements": [...], "unsure": false, "note": "..."}}. A draft is
saved only if it parses as the schema; verbatim and number warnings are printed so the drafter can
fix them first. Drafts are suggestions: the author reviews every one in the labelling tool.
The test and held-out splits are refused, so a drafting session never loads their text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import ValidationError

from rail_extract import labels as L
from rail_extract.schema import Extraction

DRAFTABLE = ("train", "val")
LONG = 1600  # characters; longer passages are shown cut (long tables) with a marker


def _load(split: str, root: Path):
    if split not in DRAFTABLE:
        raise SystemExit(f"refusing '{split}': only {DRAFTABLE} may be drafted by an LLM session")
    passages = {r["id"]: r for r in L.read_jsonl(root / "data/passages.jsonl")}
    manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
    return passages, manifest


def todo(split: str, root: Path = L.ROOT) -> list[str]:
    _, manifest = _load(split, root)
    drafts = L.read_drafts(split, root)
    return [i for i in L.active_ids(manifest, split, {}) if i not in drafts]


def show(split: str, start: int, n: int, root: Path = L.ROOT) -> None:
    passages, _ = _load(split, root)
    ids = todo(split, root)
    print(f"# {len(ids)} passages still to draft in {split}")
    for pid in ids[start : start + n]:
        p = passages[pid]
        text = p["text"] if len(p["text"]) <= LONG else p["text"][:LONG] + " [...CUT...]"
        print(f"\n=== {pid}\nHEADING: {p['heading']}\n{text}")


def save(split: str, file: Path, root: Path = L.ROOT) -> int:
    passages, _ = _load(split, root)
    new = json.loads(file.read_text(encoding="utf-8"))
    drafts = L.read_drafts(split, root)
    problems = 0
    for pid, d in new.items():
        if pid not in passages:
            print(f"UNKNOWN id {pid}")
            problems += 1
            continue
        try:
            ext = Extraction.model_validate({"requirements": d["requirements"]})
        except ValidationError as exc:
            print(f"INVALID {pid}: {exc.errors()[0]['loc']} {exc.errors()[0]['msg']}")
            problems += 1
            continue
        for w in L.check_requirements(passages[pid], ext.requirements):
            print(f"WARN {pid}: {w}")
            problems += 1
        drafts[pid] = {
            "id": pid,
            "requirements": [r.model_dump() for r in ext.requirements],
            "unsure": bool(d.get("unsure")),
            "note": d.get("note"),
        }
    path = root / "data/drafts" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(drafts[k], ensure_ascii=False) + "\n" for k in sorted(drafts))
    )
    print(f"saved {len(new)} drafts ({len(drafts)} total in {split}); {problems} problems")
    return problems


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2]
    if cmd == "show":
        show(split, int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "save":
        save(split, Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)
