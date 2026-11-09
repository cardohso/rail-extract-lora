from rail_extract import training as T
from rail_extract.schema import Requirement


def test_eight_configs_fixed_and_valid():
    assert list(T.CONFIGS) == [f"C{i}" for i in range(1, 9)]
    assert T.CONFIGS["C5"].alpha == 64
    assert len({(c.r, c.lr, c.epochs, c.targets) for c in T.CONFIGS.values()}) == 8


def test_encode_masks_prompt_and_appends_eos():
    ex = T.encode_example([1, 2], [3, 4], 9)
    assert ex == {"input_ids": [1, 2, 3, 4, 9], "labels": [-100, -100, 3, 4, 9]}


def test_encode_drops_long_examples_instead_of_truncating():
    assert T.encode_example([1] * 8, [2, 3], 9, max_len=10) is None


def test_encode_rows_reports_dropped():
    req = Requirement(
        parameter="p",
        comparator=">=",
        value=1.0,
        value_max=None,
        unit=None,
        conditions=[],
        quote="q",
    )
    rows = [
        {"id": "short", "heading": "H", "text": "a", "gold": []},
        {"id": "long", "heading": "H", "text": "b " * 500, "gold": [req]},
    ]
    render = lambda messages: messages[-1]["content"]  # noqa: E731
    tokenize = lambda s: [ord(c) for c in s]  # noqa: E731
    out, dropped = T.encode_rows(rows, render, tokenize, 0, max_len=200)
    assert dropped == ["long"] and len(out) == 1
    assert out[0]["labels"][-1] == 0 and out[0]["labels"][0] == -100


def test_collate_pads_right_with_ignored_labels():
    a, b = T.encode_example([1], [2], 9), T.encode_example([1, 1, 1], [2], 9)
    c = T.collate([a, b], 0)
    assert c["input_ids"][0] == [1, 2, 9, 0, 0]
    assert c["labels"][0] == [-100, 2, 9, -100, -100]
    assert c["attention_mask"][0] == [1, 1, 1, 0, 0]


def test_batches_cover_each_epoch_and_are_deterministic():
    n, bs, seed = 10, 4, 1
    per = T.steps_per_epoch(n, bs)
    assert per == 3
    seen = [i for s in range(per) for i in T.batch_indices(s, n, bs, seed)]
    assert sorted(seen) == list(range(n))
    assert T.batch_indices(4, n, bs, seed) == T.batch_indices(4, n, bs, seed)
    assert T.batch_indices(0, n, bs, seed) != T.batch_indices(3, n, bs, seed)


def test_lr_factor_ramps_then_decays_to_near_zero():
    assert T.lr_factor(0, 100, 0.1) == 0.1
    assert T.lr_factor(9, 100, 0.1) == 1.0
    assert T.lr_factor(10, 100, 0.1) == 1.0
    assert T.lr_factor(99, 100, 0.1) < 0.01


def test_overfit_rule():
    assert T.overfit_passed({"f1": 0.9, "schema_valid": 1.0})
    assert not T.overfit_passed({"f1": 0.89, "schema_valid": 1.0})
    assert not T.overfit_passed({"f1": 1.0, "schema_valid": 0.875})
