import json

import pytest

from rail_extract import adjudicate as J
from rail_extract import labels as L

REQ = {
    "parameter": "p",
    "comparator": ">=",
    "value": 800.0,
    "value_max": None,
    "unit": "mm",
    "conditions": [],
    "quote": "q",
}


def _draft(pid, reqs, unsure=False):
    return {"id": pid, "requirements": reqs, "unsure": unsure, "note": None}


@pytest.fixture
def root(tmp_path):
    (tmp_path / "data/drafts").mkdir(parents=True)
    a = [_draft("p0", [REQ]), _draft("p1", [REQ]), _draft("p2", [], True)]
    b = [_draft("p0", []), _draft("p1", [REQ]), _draft("p2", [], True)]
    for name, rows in (("a", a), ("b", b)):
        (tmp_path / f"data/drafts/val.{name}.s0.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return tmp_path


def test_pending_lists_disagreements_and_both_unsure(root):
    assert J.pending("val", root) == ["p0", "p2"]


def test_save_rejects_bad_choice_and_non_pending(root, tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"p0": {"choice": "c"}, "p1": {"choice": "a"}}))
    assert J.save("val", f, 0, root) == 2
    assert J.read_adjudications("val", root) == {}


def test_apply_uses_choice_and_agreement(root, tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"p0": {"choice": "b", "unsure": True}, "p2": {"choice": "a"}}))
    assert J.save("val", f, 0, root) == 0
    out = J.apply("val", root)
    assert out["missing"] == [] and out["adjudicated"] == 2 and out["chose_b"] == 1
    labels = L.read_labels("val", root)
    assert labels["p0"].requirements == [] and labels["p0"].unsure
    assert labels["p0"].source == "adjudicated"
    assert labels["p1"].source == "auto-agreed" and len(labels["p1"].requirements) == 1


def test_apply_leaves_unadjudicated_without_label(root):
    out = J.apply("val", root)
    assert out["missing"] == ["p0", "p2"]
    assert set(L.read_labels("val", root)) == {"p1"}


def test_isolated_splits_refused(monkeypatch):
    monkeypatch.delenv("RAIL_EXTRACT_ISOLATED_DRAFTER", raising=False)
    with pytest.raises(SystemExit):
        J.show("test", "0/1", 1)
