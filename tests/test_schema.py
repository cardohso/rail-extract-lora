import json

import pytest

from rail_extract.schema import Extraction, json_schema, parse_output

GOOD = {
    "requirements": [
        {
            "parameter": "rung spacing",
            "comparator": "between",
            "value": 250,
            "value_max": 300,
            "unit": "mm",
            "conditions": ["units longer than 120 m"],
            "quote": "rung spacing between 250 mm and 300 mm",
        }
    ]
}


def test_valid_output_parses():
    assert parse_output(json.dumps(GOOD)).requirements[0].value == 250


def test_empty_list_is_valid():
    assert parse_output('{"requirements": []}') == Extraction(requirements=[])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["requirements"][0].update(quote=""),
        lambda d: d["requirements"][0].update(parameter=""),
        lambda d: d["requirements"][0].update(comparator="at least"),
        lambda d: d["requirements"][0].update(value_max=None),
        lambda d: d["requirements"][0].update(extra=1),
        lambda d: d["requirements"][0].pop("unit"),
    ],
)
def test_invalid_output_is_rejected(mutate):
    bad = json.loads(json.dumps(GOOD))
    mutate(bad)
    assert parse_output(json.dumps(bad)) is None


def test_not_json_is_rejected():
    assert parse_output("Sure! Here is the JSON") is None


def test_value_max_only_with_between():
    bad = json.loads(json.dumps(GOOD))
    bad["requirements"][0]["comparator"] = ">="
    assert parse_output(json.dumps(bad)) is None


def test_json_schema_is_closed_and_requires_all_fields():
    schema = json_schema()
    item = schema["$defs"]["Requirement"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
