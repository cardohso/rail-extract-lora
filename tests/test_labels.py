import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from rail_extract import labels as L
from rail_extract import splits
from rail_extract.candidates import MATCH, NONE, NUMERIC
from rail_extract.label_tool import State, make_handler
from rail_extract.schema import Requirement

REQ = {
    "parameter": "usable surface",
    "comparator": "between",
    "value": 800,
    "value_max": 1000,
    "unit": "mm",
    "conditions": ["in the lowered position"],
    "quote": "its usable surface shall be between 800 mm and 1 000 mm above floor level",
}
TEXT = (
    "If a baby nappy changing table is provided, in the lowered position its usable surface "
    "shall be between 800 mm and 1 000 mm above floor level."
)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(splits, "QUOTAS", {"val": {MATCH: 2, NUMERIC: 1, NONE: 0}})
    monkeypatch.setattr(L, "QUOTAS", splits.QUOTAS)
    monkeypatch.setattr(L, "SPLITS", ("val",))
    (tmp_path / "data/splits").mkdir(parents=True)
    ps = [
        {"id": f"d:s{i}#1", "doc_id": "d", "section": f"s{i}", "heading": f"H{i}", "text": TEXT}
        for i in range(5)
    ]
    (tmp_path / "data/passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ps))
    strata = [MATCH, MATCH, MATCH, NUMERIC, NUMERIC]
    manifest = [
        {
            "id": p["id"],
            "split": "val",
            "group": p["id"],
            "stratum": s,
            "rank": i % 3,
            "in_sample": True,
        }
        for i, (p, s) in enumerate(zip(ps, strata, strict=True))
    ]
    (tmp_path / "data/splits/manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest)
    )
    return tmp_path


def lab(pid, **kw):
    return L.Label(id=pid, split="val", **kw)


def test_active_set_size_and_skip_replacement(root):
    manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
    assert len(L.active_ids(manifest, "val", {})) == 3  # 2 match + 1 numeric
    first = L.active_ids(manifest, "val", {})
    skipped = {first[0]: lab(first[0], skipped=True, skip_reason="table")}
    after = L.active_ids(manifest, "val", skipped)
    assert len(after) == 3 and first[0] not in after


def test_next_pending_is_stable_and_finishes(root):
    manifest = L.read_jsonl(root / "data/splits/manifest.jsonl")
    labels: dict = {}
    seen = []
    while (pid := L.next_pending(manifest, "val", labels)) is not None:
        assert pid == L.next_pending(manifest, "val", labels)
        seen.append(pid)
        labels[pid] = lab(pid)
    assert len(seen) == len(set(seen)) == 3


def test_write_read_roundtrip(root):
    r = Requirement.model_validate(REQ)
    L.write_labels("val", {"d:s0#1": lab("d:s0#1", requirements=[r], seconds=12.5)}, root)
    back = L.read_labels("val", root)
    assert back["d:s0#1"].requirements[0].value_max == 1000
    assert not list((root / "data/labels").glob("*.tmp"))


def test_check_accepts_a_faithful_label_and_flags_invented_content():
    p = {"heading": "H", "text": TEXT}
    assert L.check_requirements(p, [Requirement.model_validate(REQ)]) == []
    bad = dict(REQ, value=900, unit="cm", quote="it must be about 900 cm", parameter="height")
    w = L.check_requirements(p, [Requirement.model_validate(bad)])
    assert any("quote" in x for x in w) and any("value 900" in x for x in w)
    assert any("unit cm" in x for x in w) and any("parameter" in x for x in w)


def _serve(root):
    state = State("val", root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), method="POST")
    return json.load(urllib.request.urlopen(req))


def test_http_flow_and_time_accumulates(root):
    server, base = _serve(root)
    try:
        assert b"Label requirements" in urllib.request.urlopen(base + "/").read()
        item = json.load(urllib.request.urlopen(base + "/api/next"))
        assert item["progress"]["target"] == 3 and item["text"] == TEXT
        chk = _post(base + "/api/check", {"id": item["id"], "requirements": [REQ]})
        assert chk == {"valid": True, "errors": [], "warnings": []}
        bad = _post(base + "/api/check", {"id": item["id"], "requirements": [{"parameter": ""}]})
        assert bad["valid"] is False
        _post(base + "/api/save", {"id": item["id"], "requirements": [REQ], "seconds": 30})
        out = _post(base + "/api/save", {"id": item["id"], "requirements": [], "seconds": 99999})
        assert out["progress"]["done"] == 1
        saved = L.read_labels("val", root)[item["id"]]
        assert saved.seconds == 30 + L.MAX_SECONDS_PER_PASSAGE  # capped, then added
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(base + "/api/save", {"id": "nope", "requirements": []})
        assert err.value.code == 400
    finally:
        server.shutdown()
