"""Leakage-safe splits (docs/PREREGISTRATION.md section 3). Uses passage text only, no labels.

    uv run python -m rail_extract.splits    # writes data/splits/manifest.jsonl

Every group of passages goes entirely into one split. Groups are the section prefix (four numeric
levels), all specific-case sections of a document as one group, then merged when two passages are
near duplicates (5-word shingles, Jaccard >= 0.5), across documents too. Any group that contains an
INF passage leaves the training pool. Within each split, passages of each stratum get a random
rank; the first N of a stratum are the sample and the rest are reserves for replacements.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from rail_extract.candidates import MATCH, NONE, NUMERIC, STRATA, stratum

SEED = 20261108
HELDOUT_DOC = "tsi-inf"
# split -> {stratum: target count}. Validation and train keep the main test's proportions.
QUOTAS: dict[str, dict[str, int]] = {
    "heldout": {MATCH: 24, NUMERIC: 5, NONE: 1},
    "test": {MATCH: 54, NUMERIC: 12, NONE: 4},
    "val": {MATCH: 39, NUMERIC: 8, NONE: 3},
    "train": {MATCH: 193, NUMERIC: 43, NONE: 14},
}
POOL_SPLITS = ("test", "val", "train")
_SPECIFIC = re.compile(r"specific case|P cases|\(‘[A-Z]’\)|‘[A-Z]’\)", re.IGNORECASE)
_PREFIX = re.compile(r"^\d+(?:\.\d+){0,3}")
SHINGLE = 5
JACCARD = 0.5


def group_key(passage: dict) -> str:
    section = passage["section"]
    doc = passage["doc_id"]
    if _SPECIFIC.search(section):
        return f"{doc}:specific-cases"
    container, _, rest = section.partition("§")
    m = _PREFIX.match(rest.strip())
    return f"{doc}:{container.strip()}:{m.group(0) if m else rest.strip() or container.strip()}"


def shingles(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def merged_groups(passages: list[dict]) -> dict[str, str]:
    """passage id -> final group id, after merging groups linked by near-duplicate passages."""
    uf = _UnionFind()
    keys = {p["id"]: group_key(p) for p in passages}
    for k in keys.values():
        uf.find(k)
    sets = {p["id"]: shingles(p["text"]) for p in passages}
    index: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for pid, sh in sets.items():
        for s in sh:
            index[s].append(pid)
    for pid, sh in sets.items():
        shared: Counter[str] = Counter()
        for s in sh:
            for other in index[s]:
                if other > pid:
                    shared[other] += 1
        for other, n in shared.items():
            union = len(sh) + len(sets[other]) - n
            if union and n / union >= JACCARD and keys[pid] != keys[other]:
                uf.union(keys[pid], keys[other])
    return {pid: uf.find(k) for pid, k in keys.items()}


def assign_groups(group_sizes: dict[str, dict[str, int]], rng: random.Random) -> dict[str, str]:
    """Randomised greedy: each group goes to the pool split with the most unmet MATCH need."""
    need = {s: QUOTAS[s][MATCH] for s in POOL_SPLITS}
    total = dict(need)
    out: dict[str, str] = {}
    groups = sorted(group_sizes)
    rng.shuffle(groups)
    for g in groups:
        weight = group_sizes[g].get(MATCH, 0)
        best = max(POOL_SPLITS, key=lambda s: (need[s] / total[s], rng.random()))
        out[g] = best
        need[best] -= weight
    return out


def build_manifest(passages: list[dict], seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    group_of = merged_groups(passages)
    strat = {p["id"]: stratum(p["text"]) for p in passages}
    doc_of = {p["id"]: p["doc_id"] for p in passages}
    members: dict[str, list[str]] = defaultdict(list)
    for pid, g in group_of.items():
        members[g].append(pid)
    inf_linked = {g for g, ids in members.items() if any(doc_of[i] == HELDOUT_DOC for i in ids)}

    pool_groups = {
        g: Counter(strat[i] for i in ids) for g, ids in members.items() if g not in inf_linked
    }
    assignment = assign_groups(pool_groups, rng)

    split_of: dict[str, str] = {}
    for g, ids in members.items():
        for i in ids:
            if g in inf_linked:
                if doc_of[i] == HELDOUT_DOC:
                    split_of[i] = "heldout"
            else:
                split_of[i] = assignment[g]

    manifest = []
    for split, quota in QUOTAS.items():
        for s in STRATA:
            ids = sorted(i for i, sp in split_of.items() if sp == split and strat[i] == s)
            rng.shuffle(ids)
            for rank, pid in enumerate(ids):
                manifest.append(
                    {
                        "id": pid,
                        "split": split,
                        "group": group_of[pid],
                        "stratum": s,
                        "rank": rank,
                        "in_sample": rank < quota[s],
                    }
                )
    return manifest


def shortfalls(manifest: list[dict]) -> dict[str, dict[str, int]]:
    have: Counter[tuple[str, str]] = Counter()
    for m in manifest:
        have[(m["split"], m["stratum"])] += 1
    return {
        sp: {s: QUOTAS[sp][s] - have[(sp, s)] for s in STRATA if have[(sp, s)] < QUOTAS[sp][s]}
        for sp in QUOTAS
        if any(have[(sp, s)] < QUOTAS[sp][s] for s in STRATA)
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    passages = [
        json.loads(line)
        for line in (root / "data" / "passages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = build_manifest(passages)
    out = root / "data" / "splits" / "manifest.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(m) + "\n" for m in manifest), encoding="utf-8")
    pool = Counter((m["split"], m["stratum"]) for m in manifest)
    for sp, quota in QUOTAS.items():
        print(
            sp,
            {
                s: f"{min(quota[s], pool[(sp, s)])}/{quota[s]} (pool {pool[(sp, s)]})"
                for s in STRATA
            },
        )
    print("shortfalls:", shortfalls(manifest) or "none")


if __name__ == "__main__":
    main()
