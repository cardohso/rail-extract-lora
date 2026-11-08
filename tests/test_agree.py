import json

from rail_extract import agree, drafting
from rail_extract import labels as L

REQ = {"comparator": ">=", "value": 800.0, "value_max": None, "unit": "mm"}


def d(reqs, unsure=False):
    return {"requirements": reqs, "unsure": unsure}


def test_agree_ignores_text_and_order():
    r1 = dict(REQ, parameter="a", conditions=[], quote="q")
    r2 = dict(REQ, value=900.0, parameter="b", conditions=["x"], quote="q2")
    assert agree.agree(d([r1, r2]), d([dict(r2, parameter="zzz"), r1]))
    assert agree.agree(d([]), d([]))


def test_disagree_on_value_unit_comparator_or_count():
    base = d([dict(REQ)])
    assert not agree.agree(base, d([dict(REQ, value=801.0)]))
    assert not agree.agree(base, d([dict(REQ, unit="cm")]))
    assert not agree.agree(base, d([dict(REQ, comparator="<=")]))
    assert not agree.agree(base, d([]))
    assert agree.agree(d([dict(REQ, unit="m2")]), d([dict(REQ, unit="m²")]))


def test_review_rule_needs_both_unsure_when_they_agree():
    assert not agree.needs_review(d([], True), d([], False))
    assert agree.needs_review(d([], True), d([], True))
    assert agree.needs_review(d([dict(REQ)]), d([]))


def test_decide_writes_auto_labels_and_keeps_audit_in_queue(tmp_path):
    (tmp_path / "data/drafts").mkdir(parents=True)
    ids = [f"p{i}" for i in range(60)]
    full = {"parameter": "p", "conditions": [], "quote": "q", **REQ}
    rows = [{"id": i, "requirements": [full], "unsure": False, "note": None} for i in ids]
    rows[0]["requirements"] = []  # a disagreement
    for drafter in ("a", "b"):
        data = [dict(r) for r in rows]
        if drafter == "b":
            data[0] = dict(data[0], requirements=[full])
        (tmp_path / f"data/drafts/val.{drafter}.s0.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in data)
        )
    out = agree.decide("val", tmp_path)
    assert out["review"] == ["p0"]
    assert out["audit"] and not set(out["audit"]) & set(out["auto_agreed"])
    labels = L.read_labels("val", tmp_path)
    assert set(labels) == set(out["auto_agreed"])
    assert all(x.source == "auto-agreed" for x in labels.values())


def test_isolated_splits_are_refused_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("RAIL_EXTRACT_ISOLATED_DRAFTER", raising=False)
    for split in ("test", "heldout"):
        try:
            drafting._load(split, tmp_path)
        except SystemExit as e:
            assert "refusing" in str(e)
        else:
            raise AssertionError("should refuse")
