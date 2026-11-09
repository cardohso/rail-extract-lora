"""Settle disagreements between the two drafters without a human (decision of 2026-11-09).

    uv run python -m rail_extract.adjudicate show val --slice 0/2 --n 8
    uv run python -m rail_extract.adjudicate save val FILE --slice 0
    uv run python -m rail_extract.adjudicate apply val

Protocol, fixed before any adjudication ran:
- A passage needs adjudication when `agree.needs_review` says so (the drafts differ in comparator,
  value, value_max, unit or number of requirements, or both drafters were unsure).
- The adjudicator sees the passage and both drafts and picks "a" or "b". It never edits a draft.
- `apply` writes one label per passage that both drafters drafted: the chosen draft
  (source "adjudicated") or, if the drafts agree, drafter a's draft (source "auto-agreed").
  There is no human audit sample any more.
- test and held-out need RAIL_EXTRACT_ISOLATED_DRAFTER=1, as for drafting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rail_extract import labels as L
from rail_extract.agree import needs_review

OPEN_SPLITS = ("train", "val")
CHOICES = ("a", "b")


def _guard(split: str) -> None:
    if split not in OPEN_SPLITS and os.environ.get("RAIL_EXTRACT_ISOLATED_DRAFTER") != "1":
        raise SystemExit(f"refusing '{split}': only {OPEN_SPLITS} may be read in an open session")


def read_adjudications(split: str, root: Path = L.ROOT) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((root / "data/adjudication").glob(f"{split}.s*.jsonl")):
        for r in L.read_jsonl(path):
            out[r["id"]] = r
    return out


def to_adjudicate(split: str, root: Path = L.ROOT) -> list[str]:
    """Every passage that needs adjudication, in a fixed order that does not shrink."""
    da, db = L.read_drafts(split, "a", root), L.read_drafts(split, "b", root)
    return [p for p in sorted(set(da) & set(db)) if needs_review(da[p], db[p])]


def pending(split: str, root: Path = L.ROOT, slice_spec: str = "0/1") -> list[str]:
    """Not yet adjudicated. Slices are cut from the fixed list first, so they never overlap."""
    done = read_adjudications(split, root)
    return [p for p in slice_of(to_adjudicate(split, root), slice_spec) if p not in done]


def slice_of(ids: list[str], spec: str) -> list[str]:
    i, n = (int(x) for x in spec.split("/"))
    return ids[i::n]


def _fmt(draft: dict) -> str:
    return json.dumps(
        {"requirements": draft["requirements"], "unsure": draft["unsure"], "note": draft["note"]},
        ensure_ascii=False,
        indent=1,
    )


def show(split: str, slice_spec: str, n: int, root: Path = L.ROOT) -> None:
    _guard(split)
    passages = {r["id"]: r for r in L.read_jsonl(root / "data/passages.jsonl")}
    da, db = L.read_drafts(split, "a", root), L.read_drafts(split, "b", root)
    ids = pending(split, root, slice_spec)
    print(f"# {len(ids)} passages still to adjudicate ({split}, slice {slice_spec})")
    for pid in ids[:n]:
        p = passages[pid]
        print(f"\n=== {pid}\nHEADING: {p['heading']}\n{p['text']}")
        print(f"--- DRAFT A\n{_fmt(da[pid])}\n--- DRAFT B\n{_fmt(db[pid])}")


def save(split: str, file: Path, slice_no: int, root: Path = L.ROOT) -> int:
    _guard(split)
    new = json.loads(file.read_text(encoding="utf-8"))
    ok = set(to_adjudicate(split, root))
    path = root / "data/adjudication" / f"{split}.s{slice_no}.jsonl"
    rows = {r["id"]: r for r in L.read_jsonl(path)}
    problems = 0
    for pid, d in new.items():
        if pid not in ok and pid not in rows:
            print(f"NOT PENDING {pid}")
            problems += 1
        elif d.get("choice") not in CHOICES:
            print(f"BAD CHOICE {pid}: {d.get('choice')!r}")
            problems += 1
        else:
            rows[pid] = {
                "id": pid,
                "choice": d["choice"],
                "unsure": bool(d.get("unsure")),
                "reason": d.get("reason"),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(rows[k], ensure_ascii=False) + "\n" for k in sorted(rows)))
    print(f"saved {len(new)} choices ({len(rows)} in this file); {problems} problems")
    return problems


def apply(split: str, root: Path = L.ROOT) -> dict:
    """Write the labels. Passages that still lack an adjudication get no label."""
    da, db = L.read_drafts(split, "a", root), L.read_drafts(split, "b", root)
    adj = read_adjudications(split, root)
    labels: dict[str, L.Label] = {}
    missing, counts = [], {"auto-agreed": 0, "adjudicated": 0, "chose_a": 0, "chose_b": 0}
    for pid in sorted(set(da) & set(db)):
        if needs_review(da[pid], db[pid]):
            if pid not in adj:
                missing.append(pid)
                continue
            choice = adj[pid]["choice"]
            draft = da[pid] if choice == "a" else db[pid]
            source, unsure = "adjudicated", bool(adj[pid]["unsure"])
            counts["adjudicated"] += 1
            counts[f"chose_{choice}"] += 1
        else:
            draft, source, unsure = da[pid], "auto-agreed", False
            counts["auto-agreed"] += 1
        labels[pid] = L.Label(
            id=pid, split=split, requirements=draft["requirements"], unsure=unsure, source=source
        )
    L.write_labels(split, labels, root)
    return {"split": split, "labels": len(labels), "missing": missing, **counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["show", "save", "apply"])
    ap.add_argument("split", choices=L.SPLITS)
    ap.add_argument("file", nargs="?")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "show":
        show(a.split, a.slice, a.n)
    elif a.cmd == "save":
        save(a.split, Path(a.file), int(a.slice.split("/")[0]))
    else:
        print(apply(a.split))


if __name__ == "__main__":
    main()
