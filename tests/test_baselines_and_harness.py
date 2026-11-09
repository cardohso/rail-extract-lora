import json

import httpx
import pytest

from rail_extract import backends, evaluate, predict, prompts, regex_baseline
from rail_extract import labels as L
from rail_extract import scoring as S
from rail_extract.data import load_split
from rail_extract.schema import Requirement, flat_json_schema

# ----------------------------------------------------------------------------- R0


def extract(text):
    return regex_baseline.extract("H", text).requirements


def test_r0_finds_maximum_with_unit():
    (r,) = extract("The axle load shall not exceed 22,5 t.")
    assert (r.comparator, r.value, r.unit) == ("<=", 22.5, "t")
    assert r.quote == "The axle load shall not exceed 22,5 t."
    assert r.parameter == "The axle load"


def test_r0_minimum_and_range():
    (r,) = extract("The wheel diameter shall be at least 330 mm.")
    assert (r.comparator, r.value) == (">=", 330.0)
    (r,) = extract("The rung spacing shall be between 250 mm and 300 mm.")
    assert (r.comparator, r.value, r.value_max, r.unit) == ("between", 250.0, 300.0, "mm")


def test_r0_ignores_scope_gauge_and_sentences_without_unit_or_wording():
    assert extract("The TSI shall apply to networks with 1 435 mm track gauge.") == []
    assert extract("Units designed for a speed higher than 60 km/h shall have a brake.") == []
    assert extract("The section describes the process in 2019.") == []
    assert extract("The gauge is 1 435 mm.") == []  # no normative wording


def test_r0_output_is_valid_and_verbatim():
    text = "The height shall not exceed 1 500 mm above rail level. The depth shall be at least 2 m."
    passage = {"id": "p", "heading": "H", "text": text}
    rec = S.make_prediction("p", "R0", regex_baseline.run(passage))
    assert rec["schema_valid"] and len(rec["requirements"]) == 2
    assert all(S.quote_is_verbatim(r["quote"], passage) for r in rec["requirements"])


# ------------------------------------------------------------------------- schema


def test_flat_schema_has_no_refs_or_titles():
    flat = json.dumps(flat_json_schema())
    assert "$ref" not in flat and "$defs" not in flat and '"title"' not in flat
    assert "minLength" in flat and "between" in flat


# ------------------------------------------------------------------------ prompts


def row(pid, text, gold):
    return {"id": pid, "heading": "Heading", "text": text, "gold": gold}


def req(conditions=(), value=1.0):
    return Requirement(
        parameter="p",
        comparator=">=",
        value=value,
        value_max=None,
        unit="mm",
        conditions=list(conditions),
        quote="q",
    )


def train_rows():
    rows = [row(f"e{i}", "no limits here", []) for i in range(6)]
    rows += [row(f"c{i}", "limit", [req(["when x"])]) for i in range(4)]
    rows += [row(f"m{i}", "limits", [req(), req(value=2.0)]) for i in range(4)]
    rows += [row(f"s{i}", "limit", [req()]) for i in range(4)]
    rows.append(row("long", "word " * 400, [req()]))
    return rows


def test_examples_cover_the_three_kinds_and_are_nested():
    ex = prompts.select_examples(train_rows())
    first3 = ex[:3]
    assert not first3[0]["gold"]
    assert any(g.conditions for g in first3[1]["gold"])
    assert len(first3[2]["gold"]) >= 2
    assert "long" not in {e["id"] for e in ex}
    assert [e["id"] for e in prompts.select_examples(train_rows())] == [e["id"] for e in ex]
    assert len({e["id"] for e in ex}) == len(ex)


def test_build_messages_layout():
    ex = prompts.select_examples(train_rows())[:3]
    msgs = prompts.build_messages(row("x", "Some text", []), "v1", ex)
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert "Passage:\nSome text" in msgs[-1]["content"]
    assert json.loads(msgs[2]["content"]) == {"requirements": []}
    assert '"requirements"' in msgs[0]["content"]
    assert prompts.prompt_hash("v1", ex) != prompts.prompt_hash("v1", None)


# ----------------------------------------------------------------------- backends


def test_mock_backend_and_predict_with_examples():
    seen = []

    def fn(messages):
        seen.append(messages)
        return '{"requirements": []}'

    rows = [row("a", "t", []), row("b", "t", [])]
    recs = predict.predict("B1", rows, backends.MockBackend(fn))
    assert [r["id"] for r in recs] == ["a", "b"] and all(r["schema_valid"] for r in recs)
    assert len(seen) == 2


def make_http(handler, cls=backends.GeminiBackend, model="m"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return cls(model, client=client, sleep=lambda s: None)


def test_gemini_request_shape_cache_and_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={})
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": '{"requirements": []}'}]}}]}
        )

    be = make_http(handler)
    msgs = prompts.build_messages(row("x", "T", []), "v1", None)
    assert be.generate([msgs]) == ['{"requirements": []}']
    assert len(calls) == 2  # one retry after the 429
    body = json.loads(calls[-1].content)
    assert body["generationConfig"]["temperature"] == 0
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert "responseJsonSchema" in body["generationConfig"]
    assert calls[-1].headers["x-goog-api-key"] == "test-key"
    assert be.generate([msgs]) == ['{"requirements": []}']
    assert len(calls) == 2  # second call served from the cache


def test_groq_request_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    be = make_http(handler, backends.GroqBackend, "some/model")
    be.generate([[{"role": "user", "content": "hi"}]])
    assert seen["body"]["temperature"] == 0
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True


def test_missing_key_stops_without_request(monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    be = make_http(lambda r: pytest.fail("no request expected"))
    with pytest.raises(SystemExit):
        be.generate([[{"role": "user", "content": "hi"}]])


# ----------------------------------------------------- split loading and evaluation


@pytest.fixture
def root(tmp_path):
    (tmp_path / "data/splits").mkdir(parents=True)
    (tmp_path / "data/labels").mkdir(parents=True)
    passages = [
        {
            "id": "p1",
            "heading": "H",
            "section": "s",
            "text": "The axle load shall not exceed 22 t.",
        },
        {"id": "p2", "heading": "H", "section": "s", "text": "Nothing measurable is said here."},
    ]
    (tmp_path / "data/passages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in passages))
    manifest = [
        {
            "id": "p1",
            "split": "val",
            "group": "g",
            "stratum": "match",
            "rank": 0,
            "in_sample": True,
        },
        {"id": "p2", "split": "val", "group": "g", "stratum": "none", "rank": 0, "in_sample": True},
        {
            "id": "p1",
            "split": "test",
            "group": "g",
            "stratum": "match",
            "rank": 0,
            "in_sample": True,
        },
    ]
    (tmp_path / "data/splits/manifest.jsonl").write_text(
        "".join(json.dumps(m) + "\n" for m in manifest)
    )
    gold = req(value=22.0)
    gold = gold.model_copy(update={"comparator": "<=", "parameter": "axle load", "quote": "x"})
    labels = {
        "p1": L.Label(id="p1", split="val", requirements=[gold]),
        "p2": L.Label(id="p2", split="val"),
    }
    L.write_labels("val", labels, tmp_path)
    return tmp_path


def test_test_splits_need_final(root):
    with pytest.raises(SystemExit):
        load_split("test", False, root)
    with pytest.raises(SystemExit):
        load_split("heldout", False, root)


def test_load_split_adds_stratum_and_gold(root):
    rows = load_split("val", False, root)
    assert [r["id"] for r in rows] == ["p1", "p2"]
    assert rows[0]["stratum"] == "match" and len(rows[0]["gold"]) == 1


def test_r0_end_to_end_and_missing_predictions(root):
    rows = load_split("val", False, root)
    recs = predict.predict("R0", rows, None)
    predict.save("R0", "val", recs, {"system": "R0"}, root)
    out = evaluate.evaluate("R0", "val", False, root)
    assert out["summary"]["n"] == 2 and out["summary"]["false_extraction"] == 0.0
    assert 0 < out["summary"]["f1"] <= 1
    assert "field micro-F1" in evaluate.format_summary(out)
    predict.save("R0", "val", recs[:1], {}, root)
    with pytest.raises(SystemExit):
        evaluate.evaluate("R0", "val", False, root)
