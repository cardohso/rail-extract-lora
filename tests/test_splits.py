import json
from collections import defaultdict
from pathlib import Path

import pytest

from rail_extract import splits
from rail_extract.candidates import MATCH, NONE, NUMERIC, stratum
from rail_extract.splits import build_manifest, group_key, merged_groups, shortfalls

ROOT = Path(__file__).resolve().parents[1]


def p(i, doc, section, text):
    return {"id": f"{doc}:{section}#{i}", "doc_id": doc, "section": section, "text": text}


def test_strata():
    assert stratum("The width shall be at least 800 mm.") == MATCH
    assert stratum("Published in 2019 under Article 5.") == NUMERIC
    assert stratum("The infrastructure manager shall inform the agency.") == NONE


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("Annex §4.2.2.9.1.3", "d:Annex:4.2.2.9"),
        ("Annex §4.2.2.9", "d:Annex:4.2.2.9"),
        ("Article 5", "d:Article 5:Article 5"),
        ("Annex §Specific case Spain ‘P’", "d:specific-cases"),
        ("Annex §P cases (47)", "d:specific-cases"),
    ],
)
def test_group_key(section, expected):
    assert group_key(p(1, "d", section, "x")) == expected


def test_near_duplicates_merge_groups_across_documents():
    text = "the usable surface shall be between 800 mm and 1 000 mm above floor level of the table"
    ps = [p(1, "a", "Annex §4.1", text), p(1, "b", "Annex §9.9", text + " here")]
    ps.append(
        p(1, "c", "Annex §1.1", "completely different words about something else entirely now")
    )
    g = merged_groups(ps)
    assert g[ps[0]["id"]] == g[ps[1]["id"]] != g[ps[2]["id"]]


def _corpus():
    ps = []
    for d in ("a", "b", "tsi-inf"):
        for s in range(40):
            text = f"{d} section {s} word{s} the width shall be at least {s + 1} mm for every unit"
            ps.append(
                p(1, d, f"Annex §{s}.1.1", text + f" unique{d}{s} filler{s} extra{s} more{s}")
            )
    return ps


def test_no_group_in_two_splits_and_heldout_only_inf(monkeypatch):
    monkeypatch.setattr(
        splits,
        "QUOTAS",
        {
            "heldout": {MATCH: 5, NUMERIC: 0, NONE: 0},
            "test": {MATCH: 5, NUMERIC: 0, NONE: 0},
            "val": {MATCH: 5, NUMERIC: 0, NONE: 0},
            "train": {MATCH: 20, NUMERIC: 0, NONE: 0},
        },
    )
    ps = _corpus()
    m = build_manifest(ps)
    by_group = defaultdict(set)
    for row in m:
        by_group[row["group"]].add(row["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert {r["id"].split(":")[0] for r in m if r["split"] == "heldout"} == {"tsi-inf"}
    assert not any(r["id"].startswith("tsi-inf") for r in m if r["split"] != "heldout")


def test_inf_linked_group_never_reaches_training(monkeypatch):
    monkeypatch.setattr(
        splits, "QUOTAS", {s: {MATCH: 1, NUMERIC: 0, NONE: 0} for s in splits.QUOTAS}
    )
    shared = "the gauge shall be at least 1 435 mm on every line of the network as written here"
    ps = [p(1, "tsi-ene", "Article 2", shared), p(1, "tsi-inf", "Article 2", shared)]
    ps += _corpus()
    m = {r["id"]: r for r in build_manifest(ps)}
    assert "tsi-ene:Article 2#1" not in m
    assert m["tsi-inf:Article 2#1"]["split"] == "heldout"


def test_deterministic():
    ps = _corpus()
    assert build_manifest(ps, seed=1) == build_manifest(ps, seed=1)
    assert build_manifest(ps, seed=1) != build_manifest(ps, seed=2)


def test_shortfalls_reported(monkeypatch):
    monkeypatch.setattr(splits, "QUOTAS", {"train": {MATCH: 99, NUMERIC: 0, NONE: 0}})
    assert shortfalls([{"split": "train", "stratum": MATCH}]) == {"train": {MATCH: 98}}


@pytest.mark.skipif(
    not (ROOT / "data" / "splits" / "manifest.jsonl").exists(), reason="manifest not built"
)
def test_committed_manifest_has_no_leakage():
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "splits" / "manifest.jsonl").read_text().splitlines()
    ]
    by_group = defaultdict(set)
    for r in rows:
        by_group[r["group"]].add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(r["id"].startswith("tsi-inf") for r in rows if r["split"] == "heldout")
    assert not any(r["id"].startswith("tsi-inf") for r in rows if r["split"] != "heldout")
