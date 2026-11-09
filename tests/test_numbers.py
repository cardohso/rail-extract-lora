import pytest

from rail_extract.numbers import fidelity_errors, norm_unit, numbers_in
from rail_extract.schema import Extraction, Requirement


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a gauge of 1 435 mm", {1435.0}),
        ("tunnels of more than 0,5 km", {0.5}),
        ("between 800 mm and 1 000 mm", {800.0, 1000.0}),
        ("distance of 3,47 m", {3.47}),
        ("at least two spaces", {2.0}),
        ("5 km at 160 km/h", {5.0, 160.0}),
    ],
)
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_no_breaking_space_surprises():
    assert numbers_in("1 000 mm") == {1000.0}


def test_unit_aliases():
    assert norm_unit("m/s2") == norm_unit("m/s²") == "m/s²"
    assert norm_unit(None) is None


def _req(value, unit="mm"):
    return Requirement(
        parameter="p",
        comparator=">=",
        value=value,
        value_max=None,
        unit=unit,
        conditions=[],
        quote="q",
    )


def test_faithful_output_has_no_errors():
    out = Extraction(requirements=[_req(800)])
    assert fidelity_errors(out, "the width shall be at least 800 mm") == []


def test_invented_value_and_unit_are_flagged():
    out = Extraction(requirements=[_req(900, unit="cm")])
    assert fidelity_errors(out, "the width shall be at least 800 mm") == ["value 900", "unit cm"]


def test_unit_must_be_a_whole_token_not_a_substring():
    out = Extraction(requirements=[_req(800, unit="m")])
    assert fidelity_errors(out, "the width shall be at least 800 mm") == ["unit m"]


def test_unit_alias_is_matched():
    out = Extraction(requirements=[_req(2, unit="m/s²")])
    assert fidelity_errors(out, "acceleration of 2 m/s2") == []


def test_alias_does_not_corrupt_longer_units():
    out = Extraction(requirements=[_req(2000, unit="cm2")])
    assert fidelity_errors(out, "a free area of 2 000 cm2") == []
    out = Extraction(requirements=[_req(2000, unit="m2")])
    assert fidelity_errors(out, "a free area of 2 000 cm2") == ["unit m2"]


def test_negative_numbers_and_range_dashes():
    assert -25.0 in numbers_in("operating at – 25 °C and above")
    assert -40.0 in numbers_in("from -40 °C")
    assert 25.0 in numbers_in("– 25 °C")  # the positive reading is always allowed
    assert -21.0 not in numbers_in("between 15 – 21 Hz")  # a range dash is not a sign
    assert -10.0 not in numbers_in("5 % - 10 %")


def test_table_rows_do_not_merge_numbers():
    assert numbers_in("YTA | 70\n180 | 500") == {70.0, 180.0, 500.0}
    assert 70180.0 not in numbers_in("YTA | 70\n180")
    assert numbers_in("a gauge of 1 435 mm") == {1435.0}


def test_multi_word_units_and_slash_spacing():
    out = Extraction(requirements=[_req(2, unit="pulses per second")])
    assert fidelity_errors(out, "at least 2 pulses per second") == []
    out = Extraction(requirements=[_req(80, unit="km/h")])
    assert fidelity_errors(out, "not more than 80 km / h") == []
    assert norm_unit("Pulses  per  second") == "pulses per second"
