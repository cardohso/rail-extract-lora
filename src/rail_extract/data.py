"""Loading a split for prediction or scoring. test and held-out need an explicit final=True
(docs/PREREGISTRATION.md section 3: code reads the test files only with --final)."""

from __future__ import annotations

from pathlib import Path

from rail_extract import labels as L

FINAL_SPLITS = ("test", "heldout")


def load_split(split: str, final: bool = False, root: Path = L.ROOT) -> list[dict]:
    """The split's passages in manifest order, each with its sampling stratum and gold label."""
    if split in FINAL_SPLITS and not final:
        raise SystemExit(f"refusing '{split}' without --final (the test files are frozen)")
    passages = {r["id"]: r for r in L.read_jsonl(root / "data/passages.jsonl")}
    manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
    labels = L.read_labels(split, root)
    stratum = {m["id"]: m["stratum"] for m in manifest}
    ids = [i for i in L.active_ids(manifest, split, labels) if i in labels]
    return [
        passages[i] | {"stratum": stratum[i], "gold": labels[i].requirements, "split": split}
        for i in sorted(ids)
    ]
