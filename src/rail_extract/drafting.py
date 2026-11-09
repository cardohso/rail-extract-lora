"""Helpers for drafting labels with an LLM session.

    uv run python -m rail_extract.drafting show val  --drafter b --slice 0/5 --start 0 --n 12
    uv run python -m rail_extract.drafting save val FILE --drafter b --slice 0

Each drafter writes its own files, data/drafts/<split>.<drafter>.s<slice>.jsonl, so drafters and
slices can run in parallel without touching each other's files. A drafter must not read another
drafter's files. FILE is {"<passage id>": {"requirements": [...], "unsure": false, "note": "..."}}.
A draft is saved only if it parses as the schema; verbatim and number warnings are printed so the
drafter can fix them first. Drafts are suggestions, never labels.

train and val may be drafted by any session. test and held-out are refused unless the environment
variable RAIL_EXTRACT_ISOLATED_DRAFTER=1 is set, which only an isolated drafting session that
reports nothing but counts back may do (docs/LABELLING.md section 5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from pydantic import ValidationError

from rail_extract import labels as L
from rail_extract.schema import Extraction

OPEN_SPLITS = ("train", "val")
LONG = 1300  # characters; longer passages are shown cut (long tables) with a marker


def _load(split: str, root: Path):
    if split not in OPEN_SPLITS and os.environ.get("RAIL_EXTRACT_ISOLATED_DRAFTER") != "1":
        raise SystemExit(
            f"refusing '{split}': only {OPEN_SPLITS} may be drafted in an open session"
        )
    passages = {r["id"]: r for r in L.read_jsonl(root / "data/passages.jsonl")}
    manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
    return passages, manifest


def slice_ids(split: str, slice_spec: str, root: Path = L.ROOT) -> list[str]:
    """The sample of a split in a fixed pseudo-random order, then every n-th passage."""
    _, manifest = _load(split, root)
    i, n = (int(x) for x in slice_spec.split("/"))
    ids = sorted(
        L.active_ids(manifest, split, {}), key=lambda p: hashlib.sha1(p.encode()).hexdigest()
    )
    return ids[i::n]


def todo(split: str, drafter: str, slice_spec: str = "0/1", root: Path = L.ROOT) -> list[str]:
    done = L.read_drafts(split, drafter, root)
    return [p for p in slice_ids(split, slice_spec, root) if p not in done]


def show(
    split: str, drafter: str, slice_spec: str, start: int, n: int, root: Path = L.ROOT
) -> None:
    passages, _ = _load(split, root)
    ids = todo(split, drafter, slice_spec, root)
    print(f"# {len(ids)} passages still to draft ({split}, drafter {drafter}, slice {slice_spec})")
    for pid in ids[start : start + n]:
        p = passages[pid]
        text = p["text"] if len(p["text"]) <= LONG else p["text"][:LONG] + " [...CUT...]"
        print(f"\n=== {pid}\nHEADING: {p['heading']}\n{text}")


def save(split: str, file: Path, drafter: str, slice_no: int, root: Path = L.ROOT) -> int:
    passages, _ = _load(split, root)
    new = json.loads(file.read_text(encoding="utf-8"))
    path = root / "data/drafts" / f"{split}.{drafter}.s{slice_no}.jsonl"
    drafts = {r["id"]: r for r in L.read_jsonl(path)}
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(drafts[k], ensure_ascii=False) + "\n" for k in sorted(drafts))
    )
    print(f"saved {len(new)} drafts ({len(drafts)} in this file); {problems} problems")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["show", "save"])
    ap.add_argument("split", choices=L.SPLITS)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    if a.cmd == "show":
        show(a.split, a.drafter, a.slice, a.start, a.n)
    else:
        save(a.split, Path(a.file), a.drafter, int(a.slice.split("/")[0]))


if __name__ == "__main__":
    main()
