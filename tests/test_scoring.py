import json

from rail_extract import scoring as S
from rail_extract.schema import Requirement


def R(
    parameter="gauge",
    comparator=">=",
    value=800.0,
    value_max=None,
    unit="mm",
    conditions=(),
    quote="q",
):
    return Requirement(
        parameter=parameter,
        comparator=comparator,
        value=value,
        value_max=value_max,
        unit=unit,
        conditions=list(conditions),
        quote=quote,
    )


def totals(counts):
    return {f: tuple(v) for f, v in counts.items()}


def test_perfect_match_has_no_errors():
    c, pairs = S.passage_counts([R()], [R()])
    assert pairs == [(0, 0)]
    assert totals(c)["parameter"] == (1, 0, 0) and totals(c)["unit"] == (1, 0, 0)
    assert all(v[1] == 0 and v[2] == 0 for v in c.values())


def test_wrong_value_is_one_fp_and_one_fn():
    c, _ = S.passage_counts([R(value=800.0)], [R(value=900.0)])
    assert totals(c)["value"] == (0, 1, 1)
    assert totals(c)["parameter"] == (1, 0, 0)


def test_unit_null_rules():
    both_null = S.pair_counts(R(unit=None), R(unit=None))
    assert both_null["unit"] == [0, 0, 0]
    assert S.pair_counts(R(unit=None), R(unit="mm"))["unit"] == [0, 1, 0]
    assert S.pair_counts(R(unit="mm"), R(unit=None))["unit"] == [0, 0, 1]
    assert S.pair_counts(R(unit="m2"), R(unit="m²"))["unit"] == [1, 0, 0]


def test_unmatched_gold_and_pred():
    c, pairs = S.passage_counts([R()], [])
    assert pairs == [] and totals(c)["value"] == (0, 0, 1) and totals(c)["unit"] == (0, 0, 1)
    c, _ = S.passage_counts([], [R(conditions=["when x"])])
    assert totals(c)["value"] == (0, 1, 0) and totals(c)["conditions"] == (0, 1, 0)


def test_alignment_prefers_the_better_pair():
    gold = [R(parameter="axle load", value=22.5, unit="t"), R(parameter="gauge", value=1435.0)]
    pred = [R(parameter="gauge", value=1435.0), R(parameter="axle load", value=22.5, unit="t")]
    assert sorted(S.align(gold, pred)) == [(0, 1), (1, 0)]
    c, _ = S.passage_counts(gold, pred)
    assert all(v[1] == 0 and v[2] == 0 for v in c.values())


def test_eligible_by_value_and_unit_even_when_parameter_differs():
    assert S.eligible(R(parameter="a b"), R(parameter="x y"))
    assert not S.eligible(R(parameter="a b"), R(parameter="x y", value=1.0))
    assert S.eligible(R(parameter="wheel diameter"), R(parameter="wheel diameter", value=1.0))


def test_conditions_use_jaccard_and_match_once():
    assert S.condition_counts(["on curves of radius 250 m"], ["curves of radius 250 m"]) == [
        1,
        0,
        0,
    ]
    assert S.condition_counts(["a b"], ["a b", "a b"]) == [1, 1, 0]
    assert S.condition_counts(["alpha"], ["beta"]) == [0, 1, 1]


def test_stop_words_and_punctuation_ignored_in_jaccard():
    assert S.jaccard("The gauge of the track.", "gauge track") == 1.0


def test_prf_values():
    assert S.prf([2, 1, 1]) == {"precision": 2 / 3, "recall": 2 / 3, "f1": 4 / 6}
    assert S.prf([0, 0, 0])["f1"] == 1.0


def test_make_prediction_flags():
    ok = json.dumps({"requirements": []})
    assert S.make_prediction("p", "x", ok)["schema_valid"]
    fenced = S.make_prediction("p", "x", "```json\n" + ok + "\n```")
    assert fenced["json_valid"] and fenced["schema_valid"]
    bad_schema = S.make_prediction("p", "x", '{"requirements": [{"parameter": ""}]}')
    assert bad_schema["json_valid"] and not bad_schema["schema_valid"]
    assert bad_schema["requirements"] == []
    assert not S.make_prediction("p", "x", "not json")["json_valid"]


def test_verifier_drops_requirements_whose_numbers_are_not_in_the_quote():
    keep = R(value=800.0, quote="at least 800 mm")
    drop = R(value=900.0, quote="at least 800 mm")
    assert S.verified([keep, drop]) == [keep]


PASSAGE = {
    "id": "p",
    "heading": "4.2 Gauge",
    "text": "The gauge shall be at least 800 mm. Wheels shall be 4 in number.",
}


def test_score_passage_fidelity_and_quotes():
    gold = [R(quote="The gauge shall be at least 800 mm")]
    rec = S.make_prediction(
        "p",
        "x",
        json.dumps(
            {
                "requirements": [
                    R(quote="The gauge shall be at least 800 mm").model_dump(),
                    R(parameter="gauge", value=999.0, quote="made up").model_dump(),
                ]
            }
        ),
    )
    s = S.score_passage(PASSAGE, gold, rec, "match")
    assert s.fidelity_errors == 1 and s.fidelity_error
    assert s.fidelity_items == 4
    assert (s.quotes, s.quotes_verbatim) == (2, 1)
    assert not s.exact
    assert (
        S.score_passage(PASSAGE, gold, S.make_prediction("p", "x", '{"requirements": []}')).n_pred
        == 0
    )


def test_summary_false_extraction_and_exact_match():
    empty_ok = S.score_passage(
        PASSAGE, [], S.make_prediction("p", "x", '{"requirements": []}'), "none"
    )
    rec = S.make_prediction(
        "p", "x", json.dumps({"requirements": [R(quote="at least 800 mm").model_dump()]})
    )
    empty_bad = S.score_passage(PASSAGE, [], rec, "none")
    hit = S.score_passage(PASSAGE, [R(quote="at least 800 mm")], rec, "match")
    out = S.summarise([empty_ok, empty_bad, hit])
    assert out.false_extraction == 0.5 and out.false_extraction_by_stratum == {"none": 0.5}
    assert out.exact_match == 2 / 3
    assert out.empty_passages == 2 and out.json_valid == 1.0
    assert 0 < out.f1 < 1


def test_exact_gold_alignment_handles_many_requirements():
    gold = [R(parameter=f"param{i}", value=float(i)) for i in range(8)]
    pred = list(reversed(gold))
    assert len(S.align(gold, pred)) == 8
